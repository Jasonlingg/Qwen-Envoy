"""The provisional judge remains anonymous, bounded, and clearly non-human."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from benchmarks.envoybench.judge import (
    ABSENCE_NOTE,
    MAX_OUTPUT_TOKENS,
    _anthropic_call,
    _write_new,
    enforce_input_bounds,
    judge_review,
    load_checkpoint,
    load_prepared_review,
    main,
    parse_verdicts,
    plan,
)
from benchmarks.envoybench.score import (
    KEY_VERSION,
    REVIEW_VERSION,
    RUBRIC_VERSION,
    _review_content,
    score_completed_review,
)
from src.eval.artifacts import configuration_hash


def _prepared(question_count: int = 2) -> dict:
    rows = []
    for question_index in range(question_count):
        for candidate_index in range(2):
            rows.append({
                "blind_id": f"R{len(rows) + 1:03d}",
                "question_id": f"q{question_index + 1}",
                "question": "How much did recall improve?",
                "expected_answerability": "sufficient",
                "reference_answer": "Recall improved four points.",
                "grader_notes": ["Recall improved by four points in the study."],
                "status": "submitted",
                "answer": (
                    "Recall improved four points." if candidate_index else "Recall fell."
                ),
                "citations": ["paper_a"],
                "evidence": [{
                    "doc_id": "paper_a", "start": 0, "end": 15,
                    "valid": True, "quote": "Recall improved by four points in the study.",
                }],
                "verdict": None,
                "notes": "",
                "relevant_source_passage": "",
            })
    return {
        "schema_version": REVIEW_VERSION,
        "benchmark_id": "fixture-benchmark", "benchmark_hash": "a" * 64,
        "corpus_hash": "b" * 64, "results_hash": "c" * 64,
        "reference_review_hash": None,
        "source_reference_status": "unreviewed_qasper",
        "status": "incomplete", "reviewer_kind": "model_assisted",
        "reviewer_id": "", "reviewed_at": "", "rubric_version": RUBRIC_VERSION,
        "adjudications": [], "excluded_questions": [], "instructions": {},
        "rows": rows,
    }


def _fake_call(prompt: str) -> tuple[str, dict]:
    data = json.loads(prompt)
    verdicts = []
    for candidate in data["anonymous_candidates"]:
        if "fell" in candidate["answer"]:
            verdicts.append({
                "blind_id": candidate["blind_id"], "verdict": "fail",
                "notes": "The answer contradicts the reported increase.",
                "relevant_source_passage": "Recall improved by four points",
            })
        else:
            verdicts.append({
                "blind_id": candidate["blind_id"], "verdict": "pass",
                "notes": "Correct and supported.",
                "relevant_source_passage": "",
            })
    return json.dumps({"verdicts": verdicts}), {
        "model": "claude-sonnet-5-2026", "input_tokens": 100,
        "output_tokens": 42,
    }


def test_full_judging_is_complete_model_assisted_and_preserves_blind_content():
    prepared = _prepared()
    before = {row["blind_id"]: configuration_hash(_review_content(row))
              for row in prepared["rows"]}
    judged, report = judge_review(prepared, _fake_call, model="claude-sonnet-5")
    assert all(row["verdict"] is None for row in prepared["rows"])
    assert judged["status"] == "complete"
    assert judged["reviewer_kind"] == "model_assisted"
    assert judged["source_reference_status"] == "unreviewed_qasper"
    assert judged["reference_review_hash"] is None
    assert [row["verdict"] for row in judged["rows"]] == ["fail", "pass"] * 2
    assert before == {row["blind_id"]: configuration_hash(_review_content(row))
                      for row in judged["rows"]}
    assert report["status"] == "complete"
    assert report["input_tokens"] == 200
    assert report["output_tokens"] == 84
    assert report["reported_models"] == ["claude-sonnet-5-2026"]
    assert "not human review" in report["review_provenance"]


def test_question_smoke_never_claims_completion():
    judged, report = judge_review(_prepared(), _fake_call,
                                  model="claude-sonnet-5", max_questions=1)
    assert judged["status"] == report["status"] == "incomplete"
    assert report["answer_count"] == 2
    assert sum(row["verdict"] is None for row in judged["rows"]) == 2
    assert judged["reviewed_at"] == ""
    assert plan(_prepared(), 1)["paid_request_count"] == 1
    with pytest.raises(ValueError, match="smoke must be smaller"):
        plan(_prepared(), 2)


def test_invalid_or_incomplete_model_output_is_rejected():
    rows = _prepared(1)["rows"]
    with pytest.raises(ValueError, match="every anonymous answer"):
        parse_verdicts(json.dumps({"verdicts": [{
            "blind_id": "R001", "verdict": "pass", "notes": "", "relevant_source_passage": ""
        }]}), rows)
    answer = json.loads(_fake_call(json.dumps({"anonymous_candidates": [
        {"blind_id": "R001", "answer": "Recall fell."},
        {"blind_id": "R002", "answer": "Recall improved four points."},
    ]}))[0])
    answer["verdicts"][0]["relevant_source_passage"] = "Invented supporting text"
    with pytest.raises(ValueError, match="not present"):
        parse_verdicts(json.dumps(answer), rows)
    with pytest.raises(ValueError, match="must grade every"):
        judge_review(_prepared(1), lambda _: ('{"verdicts":[]}', {
            "model": "claude-sonnet-5-2026", "input_tokens": 1, "output_tokens": 1,
        }), model="claude-sonnet-5")


def test_absence_note_is_accepted_for_unsupported_answer():
    rows = _prepared(1)["rows"]
    response = {"verdicts": [{
        "blind_id": row["blind_id"], "verdict": "fail",
        "notes": "No source supports this answer.",
        "relevant_source_passage": ABSENCE_NOTE,
    } for row in rows]}
    assert len(parse_verdicts(json.dumps(response), rows)) == 2


def test_pass_requires_explanation_and_own_cited_evidence():
    rows = _prepared(1)["rows"]
    response = json.loads(_fake_call(json.dumps({"anonymous_candidates": [
        {"blind_id": "R001", "answer": "Recall fell."},
        {"blind_id": "R002", "answer": "Recall improved four points."},
    ]}))[0])
    response["verdicts"][1]["notes"] = ""
    with pytest.raises(ValueError, match="nonempty explanation"):
        parse_verdicts(json.dumps(response), rows)
    response["verdicts"][1]["notes"] = "Correct answer."
    no_evidence = copy.deepcopy(rows)
    no_evidence[1]["evidence"] = []
    with pytest.raises(ValueError, match="own valid cited evidence"):
        parse_verdicts(json.dumps(response), no_evidence)
    uncited = copy.deepcopy(rows)
    uncited[1]["citations"] = []
    with pytest.raises(ValueError, match="own valid cited evidence"):
        parse_verdicts(json.dumps(response), uncited)
    errored = copy.deepcopy(rows)
    errored[1]["status"] = "error"
    errored[1]["answer"] = ""
    errored[1]["expected_answerability"] = "insufficient"
    with pytest.raises(ValueError, match="must be FAIL"):
        parse_verdicts(json.dumps(response), errored)


def test_input_limits_reject_before_any_call_and_atomic_output_refuses_overwrite(tmp_path):
    summary = plan(_prepared())
    with pytest.raises(ValueError, match="max-input-chars-per-request"):
        enforce_input_bounds(summary, max_request_chars=100, max_total_chars=100_000)
    with pytest.raises(ValueError, match="max-total-input-chars"):
        enforce_input_bounds(summary, max_request_chars=100_000, max_total_chars=100)
    path = tmp_path / "review.json"
    _write_new(path, {"status": "complete"})
    assert json.loads(path.read_text()) == {"status": "complete"}
    with pytest.raises(FileExistsError):
        _write_new(path, {"status": "tampered"})
    assert json.loads(path.read_text()) == {"status": "complete"}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_dry_run_makes_no_api_call_or_output_and_never_reads_key(tmp_path, monkeypatch, capsys):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    (prepared_dir / "review.json").write_text(json.dumps(_prepared()))
    (prepared_dir / "blind-key.json").write_text("THIS MUST NOT BE OPENED")
    output = tmp_path / "judged"
    monkeypatch.setattr("sys.argv", [
        "judge", "--review", str(prepared_dir / "review.json"),
        "--output-dir", str(output),
    ])
    assert main() == 0
    assert not output.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "dry_run_no_model_calls"
    assert printed["paid_request_count"] == 2


def test_execute_writes_new_review_and_report_without_identity_key(
    tmp_path, monkeypatch, capsys
):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    (prepared_dir / "review.json").write_text(json.dumps(_prepared()))
    (prepared_dir / "blind-key.json").write_text("THIS MUST NOT BE OPENED")
    output = tmp_path / "judged"
    monkeypatch.setattr("benchmarks.envoybench.judge._anthropic_call", lambda _: _fake_call)
    monkeypatch.setattr("sys.argv", [
        "judge", "--review", str(prepared_dir / "review.json"),
        "--output-dir", str(output), "--execute",
    ])
    assert main() == 0
    assert json.loads((output / "review.json").read_text())["status"] == "complete"
    assert json.loads((output / "judge-report.json").read_text())["input_tokens"] == 200
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main()  # x-mode output must never silently overwrite a prior judge result.

    (output / "review.json").unlink()  # Simulate interruption after report was saved.
    monkeypatch.setattr(
        "benchmarks.envoybench.judge._anthropic_call",
        lambda _: lambda _prompt: (_ for _ in ()).throw(AssertionError("duplicate API call")),
    )
    monkeypatch.setattr("sys.argv", [
        "judge", "--review", str(prepared_dir / "review.json"),
        "--output-dir", str(output), "--resume", "--execute",
    ])
    assert main() == 0
    assert json.loads((output / "review.json").read_text())["status"] == "complete"


def test_forged_human_or_modified_review_is_refused(tmp_path):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    file = prepared_dir / "review.json"
    review = _prepared()
    review["reviewer_kind"] = "human"
    file.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="model_assisted"):
        load_prepared_review(file)
    review = copy.deepcopy(_prepared())
    review["rows"][0]["model_key"] = "base"
    file.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="unexpected schema"):
        load_prepared_review(file)


def test_judged_review_scores_only_in_provisional_lane():
    prepared = _prepared()
    results = [{
        "model_key": "base" if index % 2 == 0 else "trained",
        "question_id": row["question_id"], "predicted_answer": row["answer"],
    } for index, row in enumerate(prepared["rows"])]
    prepared["results_hash"] = configuration_hash({"results": results})
    key = {
        "schema_version": KEY_VERSION,
        **{field: prepared[field] for field in (
            "benchmark_id", "benchmark_hash", "corpus_hash", "results_hash",
            "reference_review_hash", "source_reference_status", "excluded_questions",
        )},
        "assignments": [{
            "blind_id": row["blind_id"],
            "question_id": row["question_id"],
            "system": "base" if index % 2 == 0 else "trained",
            "result_hash": configuration_hash(results[index]),
            "review_content_hash": configuration_hash(_review_content(row)),
        } for index, row in enumerate(prepared["rows"])],
    }
    judged, _ = judge_review(prepared, _fake_call, model="claude-sonnet-5")
    score = score_completed_review(judged, key, results=results)
    assert score["human"] is None
    assert score["source_reference_status"] == "unreviewed_qasper"
    assert score["provisional_model_assisted"]["systems"]["trained"]["pass"] == 2


def test_paid_failure_checkpoint_resumes_without_repeating_completed_question(
    tmp_path, monkeypatch, capsys
):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    review_path = prepared_dir / "review.json"
    review_path.write_text(json.dumps(_prepared()))
    (prepared_dir / "blind-key.json").write_text("UNREADABLE KEY")
    output = tmp_path / "judged"
    attempts = []

    def first_run(prompt):
        attempts.append(json.loads(prompt)["question_id"])
        if len(attempts) == 2:
            raise RuntimeError("transient API outage")
        return _fake_call(prompt)

    monkeypatch.setattr("benchmarks.envoybench.judge._anthropic_call", lambda _: first_run)
    monkeypatch.setattr("sys.argv", [
        "judge", "--review", str(review_path), "--output-dir", str(output), "--execute",
    ])
    with pytest.raises(RuntimeError, match="transient API outage"):
        main()
    assert attempts == ["q1", "q2"]
    assert not (output / "review.json").exists()
    assert not (output / "judge-report.json").exists()
    assert len((output / "judgments.jsonl").read_text().splitlines()) == 2

    resumed = []

    def second_run(prompt):
        resumed.append(json.loads(prompt)["question_id"])
        return _fake_call(prompt)

    monkeypatch.setattr("benchmarks.envoybench.judge._anthropic_call", lambda _: second_run)
    monkeypatch.setattr("sys.argv", [
        "judge", "--review", str(review_path), "--output-dir", str(output),
        "--resume", "--execute",
    ])
    assert main() == 0
    assert resumed == ["q2"]
    report = json.loads((output / "judge-report.json").read_text())
    assert report["status"] == "complete"
    assert report["reused_checkpoint_questions"] == 1
    assert report["api_request_count"] == 2
    assert len((output / "judgments.jsonl").read_text().splitlines()) == 3
    capsys.readouterr()


def test_resume_validates_input_model_and_record_hash(tmp_path, monkeypatch, capsys):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    review_path = prepared_dir / "review.json"
    review = _prepared()
    review_path.write_text(json.dumps(review))
    output = tmp_path / "judged"
    calls = 0

    def fail_second(prompt):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("offline")
        return _fake_call(prompt)

    monkeypatch.setattr("benchmarks.envoybench.judge._anthropic_call", lambda _: fail_second)
    monkeypatch.setattr("sys.argv", [
        "judge", "--review", str(review_path), "--output-dir", str(output), "--execute",
    ])
    with pytest.raises(RuntimeError, match="offline"):
        main()
    checkpoint = output / "judgments.jsonl"
    with pytest.raises(ValueError, match="different input, model"):
        load_checkpoint(checkpoint, review, model="other-model")
    changed = copy.deepcopy(review)
    changed["rows"][0]["answer"] = "A changed model output."
    with pytest.raises(ValueError, match="different input, model"):
        load_checkpoint(checkpoint, changed, model="claude-sonnet-5")
    lines = checkpoint.read_text().splitlines()
    tampered = json.loads(lines[1])
    tampered["verdicts"][0]["verdict"] = "pass"
    checkpoint.write_text(lines[0] + "\n" + json.dumps(tampered) + "\n")
    with pytest.raises(ValueError, match="record hash mismatch"):
        load_checkpoint(checkpoint, review, model="claude-sonnet-5")
    capsys.readouterr()


def test_parse_error_gets_one_retry_with_both_requests_accounted():
    requests = 0

    def flaky(prompt):
        nonlocal requests
        requests += 1
        if requests == 1:
            return "invalid JSON", {
                "model": "claude-sonnet-5-2026", "input_tokens": 10, "output_tokens": 3,
            }
        return _fake_call(prompt)

    judged, report = judge_review(_prepared(1), flaky, model="claude-sonnet-5")
    assert judged["status"] == "complete"
    assert requests == 2
    assert report["api_request_count"] == 2
    assert report["input_tokens"] == 110
    assert report["output_tokens"] == 45
    assert report["calls"][0]["attempt_count"] == 2


def test_anthropic_request_uses_provider_temperature_default(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self):
            self.messages = SimpleNamespace(create=self.create)

        def with_options(self, **_kwargs):
            return self

        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                model="claude-sonnet-5-2026",
                content=[SimpleNamespace(text='{"verdicts":[]}')],
                usage=SimpleNamespace(input_tokens=17, output_tokens=8),
            )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("anthropic.Anthropic", FakeClient)
    text, metadata = _anthropic_call("claude-sonnet-5")("example prompt")
    assert text == '{"verdicts":[]}'
    assert metadata["model"] == "claude-sonnet-5-2026"
    assert captured["max_tokens"] == MAX_OUTPUT_TOKENS == 2000
    assert "temperature" not in captured
