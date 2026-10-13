"""Registered pilot diagnostics use blind verdicts and explicit refusal decisions."""

import pytest

from scripts import analyze_qasper_v6_promotion as promotion
from src.eval.artifacts import configuration_hash


def _artifacts(monkeypatch):
    keys = ("base", "v5", "candidate")
    ids = [f"Q{index:02d}" for index in range(24)]
    results, rows, assignments = [], [], []
    for qindex, qid in enumerate(ids):
        answerability = "sufficient" if qindex < 17 else "insufficient"
        for model_index, model in enumerate(keys):
            blind_id = f"R{len(rows) + 1:03d}"
            pass_limit = {"base": 8, "v5": 10, "candidate": 13}[model]
            verdict = "pass" if qindex < pass_limit or qindex >= 17 else "fail"
            rows.append({
                "blind_id": blind_id, "question_id": qid,
                "expected_answerability": answerability, "verdict": verdict,
            })
            assignments.append({
                "blind_id": blind_id, "question_id": qid, "system": model,
            })
            results.append({
                "model_key": model, "question_id": qid, "status": "submitted",
                "duration_seconds": 10.0,
                "trajectory": [{"action": "print('paper')",
                                "observation": "{'text': 'SyntaxError in a paper'}"}],
            })
    result_hash = configuration_hash({"results": results})
    review = {
        "status": "complete", "benchmark_id": "holdout", "benchmark_hash": "benchmark",
        "corpus_hash": "corpus", "results_hash": result_hash,
        "reference_review_hash": None, "source_reference_status": "unreviewed_qasper",
        "excluded_questions": [], "rows": rows, "adjudications": [],
    }
    key = {"assignments": assignments}
    models = [{
        "key": model, "serving_hourly_usd": 1.0, "serving_hardware": "same GPU",
        "serving_runtime": "vLLM", "decoding": {"temperature": 0, "top_p": 1},
    } for model in keys]
    automatic = {
        "schema_version": promotion.AUTOMATIC_VERSION,
        "benchmark_id": "holdout", "benchmark_hash": "benchmark", "corpus_hash": "corpus",
        "results_hash": result_hash, "reference_review_hash": None,
        "source_reference_status": "unreviewed_qasper", "excluded_questions": [],
        "question_ids_included": ids,
        "run_identity": {"split": "holdout", "max_steps": 15, "models": models},
        "systems": {model: {
            "question_count": 24, "timing_complete": True,
            "mean_duration_seconds": 10.0, "total_episode_seconds": 240.0,
            "estimated_episode_compute_usd": round(240 / 3600, 6),
        } for model in keys},
    }
    monkeypatch.setattr(promotion, "score_completed_review", lambda *_args, **kwargs: {
        "result_bindings_verified": kwargs.get("results") is results,
        "source_reference_status": "unreviewed_qasper",
        "review_provenance": {"kind": "model_assisted", "reviewer_id": "synthetic-test"},
    })
    return automatic, review, key, results


def _analyze(artifacts):
    return promotion.analyze(*artifacts, base_key="base", v5_key="v5",
                             candidate_key="candidate")


def test_missing_refusal_adjudications_leave_gate_indeterminate(monkeypatch) -> None:
    artifacts = _artifacts(monkeypatch)
    report = _analyze(artifacts)
    assert report["promotion"]["decision"] == "indeterminate"
    assert report["promotion"]["conditions"]["supported_answer_gain_vs_base"]["passed"]
    assert report["promotion"]["conditions"]["false_refusals_no_increase_vs_base"][
        "passed"
    ] is None
    assert report["systems"]["candidate"]["answerable_passes"] == 13
    assert len(report["systems"]["candidate"]["missing_false_refusal_blind_ids"]) == 17


def test_complete_adjudications_can_meet_rule_without_gold_claim(monkeypatch) -> None:
    automatic, review, key, results = _artifacts(monkeypatch)
    review["adjudications"] = [{
        "type": "false_refusal", "blind_id": row["blind_id"], "value": False,
        "notes": "",
    } for row in review["rows"] if row["expected_answerability"] == "sufficient"]
    report = _analyze((automatic, review, key, results))
    assert report["promotion"]["decision"] == "registered_rule_met_provisionally"
    assert report["review_provenance"]["kind"] == "model_assisted"
    assert report["systems"]["candidate"]["reviewed_false_refusals"] == 0
    assert "not infallible gold" in report["claim_boundary"]


def test_python_error_detector_ignores_paper_prose_but_counts_stderr(monkeypatch) -> None:
    automatic, review, key, results = _artifacts(monkeypatch)
    candidate = next(row for row in results if row["model_key"] == "candidate")
    candidate["trajectory"][0]["observation"] = "\nSTDERR:\nNameError: missing variable"
    automatic["results_hash"] = review["results_hash"] = configuration_hash({"results": results})
    report = _analyze((automatic, review, key, results))
    assert report["systems"]["base"]["python_error_episodes"] == 0
    assert report["systems"]["candidate"]["python_error_episodes"] == 1
    assert report["promotion"]["conditions"]["python_errors_no_increase_vs_base"][
        "passed"
    ] is False


def test_mismatched_automatic_cost_is_rejected(monkeypatch) -> None:
    automatic, review, key, results = _artifacts(monkeypatch)
    automatic["systems"]["candidate"]["estimated_episode_compute_usd"] = 0.01
    with pytest.raises(ValueError, match="automatic cost estimate differs"):
        _analyze((automatic, review, key, results))


def test_incomplete_review_is_rejected_before_identity_key(monkeypatch) -> None:
    automatic, review, key, results = _artifacts(monkeypatch)
    review["status"] = "incomplete"
    with pytest.raises(ValueError, match="Finish and lock"):
        _analyze((automatic, review, key, results))
