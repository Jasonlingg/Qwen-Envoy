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
from src.env.document_env import initial_observation
from src.eval.sft_quality import known_paper_id, split_by_paper, trajectory_hash, trajectory_issues
from src.policies.code_execution import QASPER_SYSTEM_PROMPT
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


def span_episode(qid="q1", doc="p1", answerability="sufficient"):
    row = episode(qid, doc, answerability)
    row["student_protocol"] = "qasper-span-v1"
    row["initial_observation"] = "Question: " + row["question"] + "\n"
    row["trajectory"][-1]["action"] += (
        f' EVIDENCE: [{{"doc_id":"{doc}","start":0,"end":10}}]'
    )
    row["qasper_score"] = {"valid": True, "reward": 0.8, "reason": "scored"}
    return row


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


def test_strict_cli_records_candidate_hashes_without_exporting_unreviewed_data(tmp_path):
    row = span_episode()
    (tmp_path / "rows.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "benchmark.json").write_text(json.dumps(benchmark([row])))
    (tmp_path / "excluded.json").write_text(json.dumps(
        benchmark([episode("other", "p2")], "validation")
    ))
    app = typer.Typer()
    app.command()(main)
    output = tmp_path / "strict-candidates"
    result = CliRunner().invoke(app, [
        "--trajectories", str(tmp_path / "rows.jsonl"),
        "--benchmark", str(tmp_path / "benchmark.json"),
        "--exclude-benchmark", str(tmp_path / "excluded.json"),
        "--out-dir", str(output), "--require-span-protocol",
    ])
    assert result.exit_code == 0, result.output
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["require_span_protocol"] is True
    assert manifest["candidate_hashes"] == [
        {"question_id": "q1", "trajectory_sha256": trajectory_hash(row)}
    ]
    assert not (output / "train.jsonl").exists()


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


def test_span_export_requires_protocol_and_exact_submission():
    legacy = episode()
    accepted, rejected = select_candidates(
        [legacy], benchmark([legacy]), [], 6, 32000, require_span_protocol=True
    )
    assert not accepted
    assert "requires_span_protocol" in rejected[0]["reasons"]

    row = span_episode()
    accepted, rejected = select_candidates(
        [row], benchmark([row]), [], 6, 32000, require_span_protocol=True
    )
    assert accepted == [row]
    assert not rejected

    missing_evidence = copy.deepcopy(row)
    missing_evidence["trajectory"][-1]["action"] = 'SUBMIT: Two models. CITATIONS: ["p1"]'
    accepted, rejected = select_candidates(
        [missing_evidence], benchmark([row]), [], 6, 32000,
        require_span_protocol=True,
    )
    assert not accepted
    assert "invalid_span_submission" in rejected[0]["reasons"]

    mismatched_citation = copy.deepcopy(row)
    mismatched_citation["trajectory"][-1]["action"] = (
        'SUBMIT: Two models. CITATIONS: ["p1"] '
        'EVIDENCE: [{"doc_id":"p2","start":0,"end":10}]'
    )
    accepted, rejected = select_candidates(
        [mismatched_citation], benchmark([row]), [], 6, 32000,
        require_span_protocol=True,
    )
    assert not accepted
    assert "citations_evidence_disagree" in rejected[0]["reasons"]


def test_span_export_rejects_false_refusal_without_proxy_score_and_raw_mismatch():
    row = span_episode()
    refusal = copy.deepcopy(row)
    refusal.pop("qasper_score")
    refusal["trajectory"][-1]["action"] = (
        "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []"
    )
    accepted, rejected = select_candidates(
        [refusal], benchmark([row]), [], 6, 32000,
        require_span_protocol=True,
    )
    assert not accepted
    assert "false_refusal" in rejected[0]["reasons"]

    changed_raw = copy.deepcopy(row)
    changed_raw["trajectory"][0]["raw_action"] = "```python\nprint(read('p1'))\n```"
    accepted, rejected = select_candidates(
        [changed_raw], benchmark([row]), [], 6, 32000,
        require_span_protocol=True,
    )
    assert not accepted
    assert "raw_action_mismatch" in rejected[0]["reasons"]
    with pytest.raises(ValueError, match="Raw model action"):
        _to_conversation(changed_raw, 32000)

    padded_raw = copy.deepcopy(row)
    padded_raw["trajectory"][0]["raw_action"] = (
        padded_raw["trajectory"][0]["action"] + "\n"
    )
    accepted, rejected = select_candidates(
        [padded_raw], benchmark([row]), [], 6, 32000,
        require_span_protocol=True,
    )
    assert not accepted
    assert "raw_action_mismatch" in rejected[0]["reasons"]


def test_duplicate_span_candidates_select_same_hash_and_bind_review():
    row = span_episode()
    alternative = copy.deepcopy(row)
    alternative["trajectory"][0]["action"] = 'doc_id = "p1"\nprint(read(doc_id)[:900])'
    alternative["qasper_score"]["reward"] = 0.9
    source = benchmark([row])
    forward = select_candidates(
        [row, alternative], source, [], 6, 32000, require_span_protocol=True
    )
    backward = select_candidates(
        [alternative, row], source, [], 6, 32000, require_span_protocol=True
    )
    assert [trajectory_hash(item) for item in forward[0]] == [
        trajectory_hash(item) for item in backward[0]
    ] == [trajectory_hash(alternative)]
    assert forward[1][0]["selected_trajectory_sha256"] == trajectory_hash(alternative)
    stale_review = {
        "reviewer": "Test reviewer", "reviewer_type": "assistant",
        "reviews": [{
            "question_id": "q1", "trajectory_sha256": trajectory_hash(row),
            "verdict": "pass", "evidence_checked": True, "stopping_checked": True,
            "replay_verified": True, "notes": "Inspected the source and action path.",
        }],
    }
    with pytest.raises(ValueError, match="Stale review"):
        reviewed_candidates(forward[0], stale_review)


def test_benchmark_excluded_papers_apply_without_extra_cli_exclusions():
    row = span_episode()
    source = benchmark([row])
    source["excluded_doc_ids"] = ["p1"]
    accepted, rejected = select_candidates(
        [row], source, [], 6, 32000, require_span_protocol=True
    )
    assert not accepted
    assert "reserved_question_or_paper" in rejected[0]["reasons"]


def test_envoybench_replay_export_uses_exact_runtime_first_observation():
    row = span_episode()
    row["runtime_protocol"] = "envoybench-v1"
    row["initial_observation"] = initial_observation(
        row["question"], require_evidence=True, include_preamble=True
    )
    conversation = _to_conversation(row, 32000)
    assert conversation["messages"][0]["content"] == QASPER_SYSTEM_PROMPT
    assert conversation["messages"][1]["content"] == row["initial_observation"]
    accepted, rejected = select_candidates(
        [row], benchmark([row]), [], 6, 32000,
        require_span_protocol=True, required_runtime_protocol="envoybench-v1",
    )
    assert accepted == [row]
    assert not rejected

    old = copy.deepcopy(row)
    old.pop("runtime_protocol")
    old["initial_observation"] = f"Question: {row['question']}\n"
    accepted, rejected = select_candidates(
        [old], benchmark([row]), [], 6, 32000,
        require_span_protocol=True, required_runtime_protocol="envoybench-v1",
    )
    assert not accepted
    assert "requires_runtime_protocol" in rejected[0]["reasons"]

    mismatched = copy.deepcopy(row)
    mismatched["initial_observation"] = "Question only"
    with pytest.raises(ValueError, match="initial observation differs"):
        _to_conversation(mismatched, 32000)
