import copy
import json

import pytest
import typer
from typer.testing import CliRunner

from scripts.export_qasper_sft_data import (
    _to_conversation,
    main,
    reviewed_candidates,
    select_candidates,
)
from scripts.generate_qasper_teacher_batch import teacher_hint
from src.eval.sft_quality import known_paper_id, split_by_paper, trajectory_hash, trajectory_issues
from src.policies.teacher_hint_policy import TeacherHintPolicy


def episode(qid="q1", doc="p1", answerability="sufficient"):
    return {
        "question_id": qid,
        "question": f'Use the known paper (doc_id: "{doc}") to answer: How many?',
        "expected_answerability": answerability,
        "trajectory": [
            {"action": f'doc_id = "{doc}"\nprint(read(doc_id)[:1000])',
             "observation": "The experiment uses two models.", "done": False},
            {"action": f'SUBMIT: Two models. CITATIONS: ["{doc}"]',
             "observation": "Submitted.", "done": True},
        ],
    }


def benchmark(rows, split="train"):
    return {"source_split": split, "corpus_hash": "fixture", "questions": [
        {"id": r["question_id"], "question": r["question"], "answer": "Two models",
         "expected_answerability": r["expected_answerability"]} for r in rows
    ]}


def test_rejects_gold_appended_after_timeout_even_within_export_budget():
    row = episode()
    row["trajectory"][0]["done"] = True
    assert "action_after_episode_done" in trajectory_issues(row)


def test_rejects_label_only_abstention_and_empty_submission():
    row = episode(answerability="insufficient")
    row["trajectory"] = [{"action": "SUBMIT: Unanswerable CITATIONS: []", "done": True}]
    assert "no_investigation" in trajectory_issues(row)
    row = episode()
    row["trajectory"][-1]["action"] = 'SUBMIT: CITATIONS: ["p1"]'
    assert "empty_answer" in trajectory_issues(row)


def test_verification_keyword_hit_is_not_a_direct_read():
    row = episode()
    row["trajectory"][0]["action"] = 'print(verify("p1", "two models"))'
    assert "no_direct_first_action" in trajectory_issues(row)
    assert not trajectory_issues(episode())


def test_reserves_whole_paper_even_for_a_different_question():
    row = episode()
    heldout = benchmark([episode("different_question", "p1")], "validation")
    accepted, rejected = select_candidates([row], benchmark([row]), [heldout], 6, 32000)
    assert not accepted
    assert "reserved_question_or_paper" in rejected[0]["reasons"]
    with pytest.raises(ValueError, match="official train"):
        select_candidates([row], benchmark([row], "test"), [], 6, 32000)


def test_group_split_never_separates_sibling_questions_and_is_deterministic():
    rows = [episode(str(i), f"p{i // 3}") for i in range(15)]
    train, val = split_by_paper(rows, .2, 42)
    assert train and val
    assert {known_paper_id(r["question"]) for r in train}.isdisjoint(
        known_paper_id(r["question"]) for r in val
    )
    assert split_by_paper(rows, .2, 42) == (train, val)
    with pytest.raises(ValueError, match="two reviewed papers"):
        split_by_paper(rows[:3], .2, 42)


def test_review_must_match_actual_trajectory_and_disclose_source_checks():
    row = episode()
    item = {"question_id": "q1", "trajectory_sha256": trajectory_hash(row), "verdict": "pass",
            "evidence_checked": True, "stopping_checked": True, "replay_verified": True,
            "notes": "Both named models are visible in the first observation."}
    review = {"reviewer": "Test reviewer", "reviewer_type": "assistant", "reviews": [item]}
    assert reviewed_candidates([row], review) == [row]
    changed = copy.deepcopy(row)
    changed["trajectory"][-1]["action"] = 'SUBMIT: Three models CITATIONS: ["p1"]'
    with pytest.raises(ValueError, match="Stale review"):
        reviewed_candidates([changed], review)
    item["evidence_checked"] = False
    with pytest.raises(ValueError, match="Incomplete"):
        reviewed_candidates([row], review)


def test_unreviewed_cli_writes_candidates_but_no_training_files(tmp_path):
    row = episode()
    paths = {"rows.json": [row], "benchmark.json": benchmark([row]),
             "excluded.json": benchmark([episode("dev", "p2")], "validation")}
    for name, data in paths.items():
        (tmp_path / name).write_text(json.dumps(data))
    app = typer.Typer()
    app.command()(main)
    output = tmp_path / "candidates"
    result = CliRunner().invoke(app, [
        "--trajectories", str(tmp_path / "rows.json"),
        "--benchmark", str(tmp_path / "benchmark.json"),
        "--exclude-benchmark", str(tmp_path / "excluded.json"), "--out-dir", str(output),
    ])
    assert result.exit_code == 0, result.output
    assert (output / "review-template.json").exists()
    assert not (output / "train.jsonl").exists()
    assert not (output / "val.jsonl").exists()


def test_teacher_reference_does_not_enter_student_conversation():
    row = episode()
    question = benchmark([row])["questions"][0]
    question["answer"] = "PRIVILEGED_ANSWER_SENTINEL"
    question["grader_notes"] = ["PRIVILEGED_PASSAGE_SENTINEL"]

    class Teacher:
        seen = ""

        def act(self, observation):
            self.seen = observation
            return row["trajectory"][0]["action"]

    inner = Teacher()
    policy = TeacherHintPolicy(inner, hint=teacher_hint(question, reference_guidance=True))
    policy.act("Question: " + question["question"])
    assert "PRIVILEGED_ANSWER_SENTINEL" in inner.seen
    assert "PRIVILEGED_PASSAGE_SENTINEL" in inner.seen
    exported = json.dumps(_to_conversation(row, 32000))
    assert "PRIVILEGED" not in exported
    assert "TEACHER-ONLY" not in exported
    assert row["trajectory"][0]["observation"] in exported
