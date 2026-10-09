"""EnvoyBench keeps provenance checks apart from blinded answer support."""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.envoybench.score import (
    prepare_provisional_bundle,
    prepare_review_bundle,
    reference_review_template,
    score_completed_review,
)
from src.eval.artifacts import configuration_hash, content_hash


def _fixtures(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "The intervention improved recall by four points. The cost rose slightly."
    (corpus / "paper_a.json").write_text(json.dumps({
        "doc_id": "paper_a", "title": "A study", "text": text,
        "sections": [{"section": "Results", "start": 0, "end": len(text)}],
        "metadata": {"source_url": "https://example.org/a", "coverage": "full_text"},
    }))
    corpus_hash = content_hash(corpus)
    benchmark = {
        "schema_version": "research-benchmark-v1",
        "benchmark_id": "envoybench-dev-v1",
        "corpus_hash": corpus_hash,
        "reserved_doc_ids": ["paper_a"],
        "questions": [{
            "id": "q1", "split": "pilot_evaluation",
            "question": "What changed, and what was the tradeoff?",
            "answer": "Recall improved by four points, while cost rose slightly.",
            "expected_answerability": "sufficient",
            "required_doc_ids": ["paper_a"],
            "expected_citations": ["paper_a"],
            "minimum_distinct_sources": 1,
            "grader_notes": ["Check both parts and exact support."],
            "source_question_id": "source-q1",
            "source_paper_id": "paper_a",
            "reference_answers": ["Recall improved; cost rose."],
            "gold_evidence": [],
        }, {
            "id": "q2", "split": "pilot_evaluation",
            "question": "What was the deployment cost?",
            "answer": "The paper does not quantify deployment cost.",
            "expected_answerability": "insufficient",
            "required_doc_ids": ["paper_a"],
            "expected_citations": ["paper_a"],
            "minimum_distinct_sources": 1,
            "grader_notes": ["Do not invent a price."],
            "source_question_id": "source-q2",
            "source_paper_id": "paper_a",
            "reference_answers": ["Not reported."],
            "gold_evidence": [],
        }],
    }
    benchmark_hash = configuration_hash(benchmark)
    manifest = {
        "schema_version": "envoybench-run-v1", "status": "complete",
        "run_id": "fixture-run", "comparison_id": "fixture-comparison",
        "split": "dev", "split_status": "unreviewed",
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": benchmark_hash,
        "corpus_hash": corpus_hash,
        "full_split": True, "subset_smoke": False,
        "split_question_count": 2, "question_ids": ["q1", "q2"],
        "runner_hardware": {"platform": "test-host"},
        "models": [
            {"key": "base", "serving_hardware": "test-gpu", "serving_hourly_usd": 1.0},
            {"key": "trained", "serving_hardware": "test-gpu", "serving_hourly_usd": 1.0},
        ],
    }

    def row(model, question_id, answer, evidence, *, status="submitted"):
        return {
            "schema_version": "envoybench-run-v1",
            "run_id": manifest["run_id"], "comparison_id": manifest["comparison_id"],
            "split": manifest["split"], "split_status": manifest["split_status"],
            "question_id": question_id, "model_key": model,
            "question": next(item["question"] for item in benchmark["questions"]
                             if item["id"] == question_id),
            "benchmark_id": benchmark["benchmark_id"],
            "benchmark_hash": benchmark_hash,
            "corpus_hash": corpus_hash,
            "status": status,
            "predicted_answer": answer,
            "predicted_citations": ["paper_a"] if answer else [],
            "predicted_evidence": evidence,
            "trajectory": [
                {"step": 1, "action": "print(search('recall'))",
                 "observation": "found passage", "reward": 0, "done": False},
                {"step": 2, "action": "SUBMIT: ...", "observation": "done",
                 "reward": 0, "done": True},
            ],
            "duration_seconds": 1.5 if model == "trained" else 0.5,
            "verifier_events": [],
            "environment_diagnostics_not_benchmark_score": {"reward": 1.0},
        }

    exact = [{"doc_id": "paper_a", "start": 0, "end": 47,
              "quote": text[:47]}]
    wrong_quote = [{"doc_id": "paper_a", "start": 0, "end": 47,
                    "quote": "invented quotation"}]
    results = [
        row("base", "q1", "It got cheaper.", wrong_quote),
        row("trained", "q1", "Recall improved four points; cost rose.", exact),
        row("base", "q2", "It cost $7.", []),
        row("trained", "q2", "No deployment price is reported.", exact),
    ]
    references = reference_review_template(benchmark)
    references["status"] = "complete"
    references["reviewer_id"] = "paper-audit"
    references["reviewed_at"] = "2026-09-30"
    for item in references["questions"]:
        item["decision"] = "reference_valid"
    return benchmark, manifest, results, corpus, references


def test_blind_review_and_mechanical_metrics_stay_separate(tmp_path):
    benchmark, manifest, results, corpus, references = _fixtures(tmp_path)
    review, key, automatic = prepare_review_bundle(
        benchmark, manifest, results, corpus, references, seed=42
    )
    assert len(review["rows"]) == 4
    assert automatic["results_hash"] == review["results_hash"]
    assert all("model_key" not in row and "system" not in row for row in review["rows"])
    assert {item["system"] for item in key["assignments"]} == {"base", "trained"}
    assert automatic["systems"]["trained"]["valid_evidence_span_rate"] == 1.0
    assert automatic["systems"]["base"]["valid_evidence_span_rate"] == 0.0
    assert automatic["systems"]["base"]["estimated_episode_compute_usd"] == pytest.approx(
        1.0 / 3600, abs=1e-6
    )
    assert automatic["run_identity"]["runner_hardware"] == {"platform": "test-host"}
    assert automatic["run_identity"]["serving_hardware"] == {
        "base": "test-gpu", "trained": "test-gpu"
    }
    assert "supported_answer_rate" not in automatic["systems"]["trained"]
    assert "MuSiQue reward" in automatic["note"]
    base_q1_id = next(item["blind_id"] for item in key["assignments"]
                      if item["system"] == "base" and item["question_id"] == "q1")
    base_q1 = next(item for item in review["rows"] if item["blind_id"] == base_q1_id)
    assert base_q1["evidence"][0]["valid"] is False
    assert "submitted quote" in base_q1["evidence"][0]["error"]
    assert "The intervention" in base_q1["evidence"][0]["snapshot_quote"]

    review["status"] = "complete"
    review["reviewer_kind"] = "human"
    review["reviewer_id"] = "blind-reviewer"
    review["reviewed_at"] = "2026-09-30"
    by_id = {item["blind_id"]: item for item in key["assignments"]}
    for item in review["rows"]:
        item["verdict"] = "pass" if by_id[item["blind_id"]]["system"] == "trained" else "fail"
        if item["verdict"] == "fail":
            item["notes"] = "The submitted answer contradicts or invents the study result."
            item["relevant_source_passage"] = "paper_a:0-47"
    score = score_completed_review(review, key)
    assert score["human"]["systems"]["trained"]["supported_answer_rate"] == 1.0
    assert score["human"]["systems"]["base"]["supported_answer_rate"] == 0.0
    assert score["human"]["paired_wins"] == {"base": 0, "trained": 2}
    assert score["human"]["paired_pass_tests"]["base_vs_trained"] == {
        "left_only_pass": 0,
        "right_only_pass": 2,
        "pass_rate_difference_right_minus_left": 1.0,
        "exact_mcnemar_two_sided_p": 0.5,
        "scorable_question_count": 2,
    }
    assert score["provisional_model_assisted"] is None

    assisted = copy.deepcopy(review)
    assisted["reviewer_kind"] = "model_assisted"
    provisional = score_completed_review(assisted, key, results=results)
    assert provisional["human"] is None
    assert provisional["provisional_model_assisted"]["paired_wins"]["trained"] == 2
    assert provisional["provisional_model_assisted"]["schema_version"] == (
        "envoybench-provisional-model-assisted-score-v1"
    )
    assert "provisional model-graded" in provisional[
        "provisional_model_assisted"
    ]["decision_note"]
    assert provisional["result_bindings_verified"] is True
    assert score["result_bindings_verified"] is False


def test_error_diagnostics_separate_returns_runtime_and_endpoint_without_prose_hits(tmp_path):
    benchmark, manifest, results, corpus, _ = _fixtures(tmp_path)
    by_pair = {(row["model_key"], row["question_id"]): row for row in results}
    base = by_pair[("base", "q1")]
    base["trajectory"][0]["observation"] = (
        "[{'error': 'length must be between 1 and 3000'}, "
        "{'error': 'Document missing'}]\n\n[Step 1/15/15]"
    )
    base.update(status="error", error=(
        'RuntimeError: model endpoint returned HTTP 400: '
        '{"error": {"message": "maximum context length is 8192"}}'
    ))
    by_pair[("base", "q2")]["trajectory"][0]["observation"] = (
        "[{'doc_id': 'paper_a', 'text': 'SyntaxError, Traceback, and timed out "
        "are labels described by the paper.', 'metadata': {'error': 'source annotation'}}]"
    )
    by_pair[("trained", "q1")]["trajectory"][0]["observation"] = (
        "output before failure\nSTDERR:\nTraceback (most recent call last):\n"
        '  File "<string>", line 2, in <module>\nNameError: unknown is not defined'
    )
    by_pair[("trained", "q2")].update(status="error", error="RuntimeError: worker exited")
    _, _, automatic = prepare_provisional_bundle(benchmark, manifest, results, corpus)
    assert automatic["schema_version"] == "envoybench-automatic-v2"
    base_metrics = automatic["systems"]["base"]
    assert base_metrics["tool_error_return_count"] == 2
    assert base_metrics["tool_error_step_count"] == 1
    assert base_metrics["tool_error_episode_count"] == 1
    assert base_metrics["runtime_error_episode_count"] == 0
    assert base_metrics["endpoint_error_episode_count"] == 1
    assert base_metrics["context_limit_error_episode_count"] == 1
    # Context is a subset of endpoint errors, and one episode can have both
    # a recovered returned error and a later endpoint failure; count the union.
    assert base_metrics["execution_error_episode_count"] == 1
    assert base_metrics["execution_error_episode_rate"] == 0.5
    assert base_metrics["question_rows"][1]["error_categories"] == []
    trained_metrics = automatic["systems"]["trained"]
    assert trained_metrics["tool_error_return_count"] == 0
    assert trained_metrics["runtime_error_episode_count"] == 1
    assert trained_metrics["other_harness_error_episode_count"] == 1
    assert trained_metrics["execution_error_episode_count"] == 2


def test_structured_errors_parse_complete_multiline_json_and_separate_returns(tmp_path):
    benchmark, manifest, results, corpus, _ = _fixtures(tmp_path)
    results[0]["trajectory"][0]["observation"] = (
        '{"error": "invalid length"}\n'
        "{'error': 'invalid pattern'}\n\n\n[Step 1/15/15]"
    )
    results[1]["trajectory"][0]["observation"] = json.dumps(
        [{"error": "invalid pattern"}, {"doc_id": "paper_a", "text": "ERROR: source prose"}],
        indent=2,
    )
    _, _, automatic = prepare_provisional_bundle(benchmark, manifest, results, corpus)
    assert automatic["systems"]["base"]["tool_error_return_count"] == 2
    assert automatic["systems"]["trained"]["tool_error_return_count"] == 1
    assert automatic["systems"]["trained"]["runtime_error_episode_count"] == 0


def test_endpoint_connectivity_failure_is_not_a_context_or_tool_error(tmp_path):
    benchmark, manifest, results, corpus, _ = _fixtures(tmp_path)
    results[0].update(
        status="error", error="RuntimeError: could not reach model endpoint: connection refused"
    )
    _, _, automatic = prepare_provisional_bundle(benchmark, manifest, results, corpus)
    base = automatic["systems"]["base"]
    assert base["endpoint_error_episode_count"] == 1
    assert base["context_limit_error_episode_count"] == 0
    assert base["tool_error_episode_count"] == 0
    assert base["runtime_error_episode_count"] == 0
    assert base["other_harness_error_episode_count"] == 0


def test_zero_placeholder_timing_cannot_become_full_duration_or_cost(tmp_path):
    benchmark, manifest, results, corpus, _ = _fixtures(tmp_path)
    results[0].update(duration_seconds=0.0, status="error", error="load failed")
    _, _, automatic = prepare_provisional_bundle(benchmark, manifest, results, corpus)
    base = automatic["systems"]["base"]
    assert base["timing_complete"] is False
    assert base["timed_question_count"] == base["missing_duration_question_count"] == 1
    assert base["mean_duration_seconds"] is None
    assert base["total_episode_seconds"] is None
    assert base["estimated_episode_compute_usd"] is None
    assert base["observed_mean_duration_seconds"] == 0.5
    assert base["observed_total_episode_seconds"] == 0.5
    assert base["estimated_observed_episode_compute_usd"] == pytest.approx(0.5 / 3600, abs=1e-6)
    assert base["question_rows"][0]["duration_observed"] is False
    trained = automatic["systems"]["trained"]
    assert trained["timing_complete"] is True
    assert trained["missing_duration_question_count"] == 0
    assert trained["mean_duration_seconds"] == trained["observed_mean_duration_seconds"] == 1.5


def test_reference_gate_and_full_matrix_are_required(tmp_path):
    benchmark, manifest, results, corpus, references = _fixtures(tmp_path)
    incomplete = copy.deepcopy(references)
    incomplete["status"] = "incomplete"
    with pytest.raises(ValueError, match="reference review must be complete"):
        prepare_review_bundle(benchmark, manifest, results, corpus, incomplete)
    bad_scope = {**manifest, "full_split": False, "subset_smoke": True}
    with pytest.raises(ValueError, match="subset smoke"):
        prepare_review_bundle(benchmark, bad_scope, results, corpus, references)
    with pytest.raises(ValueError, match="complete paired"):
        prepare_review_bundle(benchmark, manifest, results[:-1], corpus, references)
    changed_corpus = copy.deepcopy(benchmark)
    changed_corpus["corpus_hash"] = "0" * 64
    with pytest.raises(ValueError, match="corpus hash"):
        prepare_review_bundle(changed_corpus, manifest, results, corpus, references)


def test_unreviewed_qasper_can_be_scored_only_as_provisional(tmp_path):
    benchmark, manifest, results, corpus, _ = _fixtures(tmp_path)
    review, key, automatic = prepare_provisional_bundle(
        benchmark, manifest, results, corpus
    )
    assert review["source_reference_status"] == "unreviewed_qasper"
    assert key["source_reference_status"] == "unreviewed_qasper"
    assert automatic["source_reference_status"] == "unreviewed_qasper"
    assert automatic["reference_review_hash"] is None
    assert automatic["scorable_question_ids"] is None
    assert automatic["question_ids_included"] == ["q1", "q2"]
    assert automatic["systems"]["base"]["question_count"] == 2
    assert "supported_answer_rate" not in automatic["systems"]["base"]
    assert review["reviewer_kind"] == "model_assisted"
    assert len(review["rows"]) == 4
    review["status"] = "complete"
    review["reviewer_id"] = "judge-model-revision-1"
    review["reviewed_at"] = "2026-09-30"
    assignments = {item["blind_id"]: item for item in key["assignments"]}
    for row in review["rows"]:
        row["verdict"] = "pass" if assignments[row["blind_id"]]["system"] == "trained" else "fail"
        if row["verdict"] == "fail":
            row["notes"] = "The answer does not match the supplied source."
            row["relevant_source_passage"] = "paper_a:0-47"
    with pytest.raises(ValueError, match="requires saved results"):
        score_completed_review(review, key)
    score = score_completed_review(review, key, results=results)
    assert score["human"] is None
    assert score["source_reference_status"] == "unreviewed_qasper"
    assert score["provisional_model_assisted"]["systems"]["trained"]["pass"] == 2
    assert score["provisional_model_assisted"]["schema_version"] == (
        "envoybench-provisional-model-assisted-score-v1"
    )

    relabeled = copy.deepcopy(review)
    relabeled["reviewer_kind"] = "human"
    with pytest.raises(ValueError, match="cannot yield a human-reviewed score"):
        score_completed_review(relabeled, key)
    altered = copy.deepcopy(review)
    altered["source_reference_status"] = "human_validated"
    with pytest.raises(ValueError, match="matching source reference provenance"):
        score_completed_review(altered, key)

    with pytest.raises(ValueError, match="complete paired"):
        prepare_provisional_bundle(benchmark, manifest, results[:-1], corpus)


def test_saved_results_bind_each_blind_id_to_its_actual_model(tmp_path):
    benchmark, manifest, results, corpus, _ = _fixtures(tmp_path)
    review, key, _ = prepare_provisional_bundle(benchmark, manifest, results, corpus)
    review["status"] = "complete"
    review["reviewer_id"] = "model-judge"
    review["reviewed_at"] = "2026-09-30"
    for row in review["rows"]:
        row["verdict"] = "pass"
    assert score_completed_review(review, key, results=results)["result_bindings_verified"]

    swapped = copy.deepcopy(key)
    q1 = [item for item in swapped["assignments"] if item["question_id"] == "q1"]
    q1[0]["system"], q1[1]["system"] = q1[1]["system"], q1[0]["system"]
    with pytest.raises(ValueError, match="blind assignment does not match"):
        score_completed_review(review, swapped, results=results)

    changed_hash = copy.deepcopy(key)
    changed_hash["assignments"][0]["result_hash"] = "0" * 64
    with pytest.raises(ValueError, match="blind assignment does not match"):
        score_completed_review(review, changed_hash, results=results)

    changed_result = copy.deepcopy(results)
    changed_result[0]["predicted_answer"] = "An answer not in the frozen run"
    with pytest.raises(ValueError, match="saved results do not match"):
        score_completed_review(review, key, results=changed_result)


def test_unscorable_reference_is_visible_and_excluded_from_human_pairs(tmp_path):
    benchmark, manifest, results, corpus, references = _fixtures(tmp_path)
    references["questions"][1]["decision"] = "unscorable"
    references["questions"][1]["reason"] = "Original annotation is ambiguous."
    review, key, automatic = prepare_review_bundle(
        benchmark, manifest, results, corpus, references
    )
    assert len(review["rows"]) == 2
    assert {item["question_id"] for item in key["assignments"]} == {"q1"}
    assert review["excluded_questions"] == [
        {"question_id": "q2", "reason": "Original annotation is ambiguous."}
    ]
    assert automatic["systems"]["base"]["question_count"] == 2
    assert automatic["scorable_question_ids"] == ["q1"]


def test_completed_review_requires_provenance_and_failure_explanation(tmp_path):
    benchmark, manifest, results, corpus, references = _fixtures(tmp_path)
    review, key, _ = prepare_review_bundle(
        benchmark, manifest, results, corpus, references
    )
    review["status"] = "complete"
    review["reviewer_kind"] = "human"
    review["reviewer_id"] = "reviewer"
    review["reviewed_at"] = "2026-09-30"
    for row in review["rows"]:
        row["verdict"] = "fail"
    with pytest.raises(ValueError, match="needs a reason"):
        score_completed_review(review, key)
    for row in review["rows"]:
        row["notes"] = "Unsupported."
        row["relevant_source_passage"] = "paper_a:0-47"
    review["reviewer_kind"] = None
    with pytest.raises(ValueError, match="reviewer_kind"):
        score_completed_review(review, key)
    review["reviewer_kind"] = "human"
    review["rows"][0]["answer"] = "Edited after blinding."
    with pytest.raises(ValueError, match="content changed"):
        score_completed_review(review, key)
