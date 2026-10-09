"""The frozen comparison gate must not promote provisional or partial evidence."""

from __future__ import annotations

from copy import deepcopy

import pytest

from benchmarks.envoybench.promotion import evaluate_promotion
from src.eval.artifacts import configuration_hash


def _artifacts(
    *,
    validated: int = 40,
    base_pass: int = 10,
    candidate_pass: int = 20,
    left_only: int = 0,
    right_only: int = 10,
):
    question_ids = [f"q{i:02d}" for i in range(40)]
    sheet = {
        "schema_version": "envoybench-reference-review-v1",
        "benchmark_id": "frozen-test",
        "benchmark_hash": "benchmark-hash",
        "corpus_hash": "corpus-hash",
        "status": "complete",
        "reviewer_id": "human-source-reviewer",
        "reviewed_at": "2026-09-30",
        "questions": [
            {
                "question_id": question_id,
                "decision": "reference_valid" if index < validated else "unscorable",
                "reason": "Ambiguous reference" if index >= validated else "",
            }
            for index, question_id in enumerate(question_ids)
        ],
    }
    reference_hash = configuration_hash(sheet)
    excluded = [
        {"question_id": q, "reason": "Ambiguous reference"} for q in question_ids[validated:]
    ]
    human = {
        "systems": {
            "qwen_base": {"pass": base_pass, "partial": 0, "fail": validated - base_pass},
            "qwen_v5": {"pass": candidate_pass, "partial": 0, "fail": validated - candidate_pass},
        },
        "paired_pass_tests": {
            "qwen_base_vs_qwen_v5": {
                "left_only_pass": left_only,
                "right_only_pass": right_only,
                "scorable_question_count": validated,
                "exact_mcnemar_two_sided_p": _exact_p(left_only, right_only),
            }
        },
    }
    score = {
        "schema_version": "envoybench-reviewed-score-v1",
        "benchmark_id": "frozen-test",
        "benchmark_hash": "benchmark-hash",
        "corpus_hash": "corpus-hash",
        "reference_review_hash": reference_hash,
        "results_hash": "results-hash",
        "review_provenance": {"kind": "human", "reviewer_id": "human-answer-reviewer"},
        "reviewed_at": "2026-09-30",
        "human": human,
        "provisional_model_assisted": None,
        "excluded_questions": excluded,
    }

    def rows():
        return [
            {
                "question_id": question_id,
                "execution_error_episode": False,
                "submitted": True,
                "duration_seconds": 1.0,
            }
            for question_id in question_ids
        ]

    automatic = {
        "schema_version": "envoybench-automatic-v1",
        "benchmark_id": "frozen-test",
        "benchmark_hash": "benchmark-hash",
        "corpus_hash": "corpus-hash",
        "reference_review_hash": reference_hash,
        "results_hash": "results-hash",
        "scorable_question_ids": question_ids[:validated],
        "run_identity": {
            "split": "test_candidate",
            "models": [
                {
                    "key": key,
                    "serving_hardware": "test-gpu",
                    "serving_runtime": "vllm",
                    "serving_hourly_usd": 0.9,
                    "decoding": {"max_tokens": 1024, "temperature": 0.0, "top_p": 1.0},
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                    "send_seed": False,
                }
                for key in ("qwen_base", "qwen_v5")
            ],
        },
        "systems": {
            "qwen_base": {
                "question_count": 40,
                "question_rows": rows(),
                "total_episode_seconds": 40.0,
                "estimated_episode_compute_usd": 0.01,
            },
            "qwen_v5": {
                "question_count": 40,
                "question_rows": rows(),
                "total_episode_seconds": 40.0,
                "estimated_episode_compute_usd": 0.01,
            },
        },
    }
    return score, automatic, sheet


def _refresh_cost(automatic: dict, model_key: str) -> None:
    system = automatic["systems"][model_key]
    total = round(sum(row["duration_seconds"] for row in system["question_rows"]), 4)
    price = next(
        item["serving_hourly_usd"]
        for item in automatic["run_identity"]["models"]
        if item["key"] == model_key
    )
    system["total_episode_seconds"] = total
    system["estimated_episode_compute_usd"] = round(total * price / 3600, 6)


def _exact_p(left: int, right: int) -> float:
    from math import comb

    n = left + right
    return round(min(1.0, 2 * sum(comb(n, k) for k in range(min(left, right) + 1)) / 2**n), 6)


def test_promotion_passes_only_when_all_predeclared_gates_pass():
    score, automatic, sheet = _artifacts()
    decision = evaluate_promotion(score, automatic, sheet)
    assert decision["status"] == "pass"
    assert decision["reasons"] == []
    assert decision["gates"]["pass_gain"]["observed"] == pytest.approx(0.25)
    assert decision["gates"]["mcnemar"]["observed"] < 0.05
    assert decision["gates"]["estimated_compute_cost"]["status"] == "pass"


def test_exact_15_percentage_point_gain_is_enough_when_paired_test_passes():
    score, automatic, sheet = _artifacts(base_pass=10, candidate_pass=16, left_only=0, right_only=6)
    decision = evaluate_promotion(score, automatic, sheet)
    assert decision["status"] == "pass"
    assert decision["gates"]["pass_gain"]["observed"] == pytest.approx(0.15)


def test_model_assisted_and_incomplete_reviews_are_inconclusive():
    score, automatic, sheet = _artifacts()
    assisted = deepcopy(score)
    assisted["review_provenance"]["kind"] = "model_assisted"
    assisted["human"] = None
    assisted["provisional_model_assisted"] = score["human"]
    assert evaluate_promotion(assisted, automatic, sheet)["status"] == "inconclusive"

    incomplete = deepcopy(sheet)
    incomplete["status"] = "incomplete"
    assert evaluate_promotion(score, automatic, incomplete)["status"] == "inconclusive"


def test_missing_cost_or_result_hash_is_inconclusive_even_with_quality_failure():
    score, automatic, sheet = _artifacts(
        base_pass=10, candidate_pass=16, left_only=5, right_only=11
    )
    assert evaluate_promotion(score, automatic, sheet)["status"] == "fail"
    no_cost = deepcopy(automatic)
    no_cost["systems"]["qwen_v5"]["estimated_episode_compute_usd"] = None
    decision = evaluate_promotion(score, no_cost, sheet)
    assert decision["status"] == "inconclusive"
    assert decision["gates"]["mcnemar"]["status"] == "fail"
    assert decision["gates"]["estimated_compute_cost"]["status"] == "inconclusive"

    old_artifact = deepcopy(automatic)
    old_artifact.pop("results_hash")
    assert (
        evaluate_promotion(score, old_artifact, sheet)["gates"]["artifact_binding"]["status"]
        == "inconclusive"
    )


def test_fewer_than_35_validated_references_is_inconclusive():
    score, automatic, sheet = _artifacts(validated=34)
    decision = evaluate_promotion(score, automatic, sheet)
    assert decision["status"] == "inconclusive"
    assert decision["gates"]["validated_questions"]["observed"] == 34


def test_three_extra_error_or_no_submission_episodes_fail():
    score, automatic, sheet = _artifacts()
    rows = automatic["systems"]["qwen_v5"]["question_rows"]
    rows[0]["execution_error_episode"] = True
    rows[0]["submitted"] = False  # The union counts this once.
    rows[1]["submitted"] = False
    decision = evaluate_promotion(score, automatic, sheet)
    assert decision["gates"]["execution_or_no_submission"]["status"] == "pass"
    assert decision["gates"]["execution_or_no_submission"]["delta"] == 2
    rows[2]["execution_error_episode"] = True
    decision = evaluate_promotion(score, automatic, sheet)
    assert decision["status"] == "fail"
    assert decision["gates"]["execution_or_no_submission"]["delta"] == 3


def test_duration_and_cost_limits_are_independent_and_inclusive():
    score, automatic, sheet = _artifacts()
    for row in automatic["systems"]["qwen_v5"]["question_rows"]:
        row["duration_seconds"] = 1.5
    _refresh_cost(automatic, "qwen_v5")
    assert evaluate_promotion(score, automatic, sheet)["status"] == "pass"
    automatic["systems"]["qwen_v5"]["question_rows"][0]["duration_seconds"] = 1.6
    _refresh_cost(automatic, "qwen_v5")
    assert evaluate_promotion(score, automatic, sheet)["gates"]["duration"]["status"] == "fail"
    automatic["systems"]["qwen_v5"]["question_rows"][0]["duration_seconds"] = 1.5
    automatic["run_identity"]["models"][1]["serving_hourly_usd"] = 1.8
    _refresh_cost(automatic, "qwen_v5")
    assert (
        evaluate_promotion(score, automatic, sheet)["gates"]["estimated_compute_cost"]["status"]
        == "fail"
    )


def test_mismatched_frozen_binding_and_question_matrix_are_inconclusive():
    score, automatic, sheet = _artifacts()
    mismatched = deepcopy(automatic)
    mismatched["results_hash"] = "another-run"
    assert evaluate_promotion(score, mismatched, sheet)["status"] == "inconclusive"
    missing_question = deepcopy(automatic)
    missing_question["systems"]["qwen_base"]["question_rows"].pop()
    assert (
        evaluate_promotion(score, missing_question, sheet)["gates"]["automatic_diagnostics"][
            "status"
        ]
        == "inconclusive"
    )


def test_development_split_or_missing_cost_metadata_cannot_promote():
    score, automatic, sheet = _artifacts()
    dev = deepcopy(automatic)
    dev["run_identity"]["split"] = "dev"
    assert evaluate_promotion(score, dev, sheet)["gates"]["test_split"]["status"] == "inconclusive"

    missing_runtime = deepcopy(automatic)
    missing_runtime["run_identity"]["models"][1]["serving_runtime"] = "unreported"
    assert (
        evaluate_promotion(score, missing_runtime, sheet)["gates"]["matched_setup"][
            "status"
        ]
        == "inconclusive"
    )

    missing_price = deepcopy(automatic)
    missing_price["run_identity"]["models"][1]["serving_hourly_usd"] = None
    assert (
        evaluate_promotion(score, missing_price, sheet)["gates"]["estimated_compute_cost"][
            "status"
        ]
        == "inconclusive"
    )

    forged_cost = deepcopy(automatic)
    forged_cost["systems"]["qwen_v5"]["estimated_episode_compute_usd"] = 0.0
    assert (
        evaluate_promotion(score, forged_cost, sheet)["gates"]["estimated_compute_cost"]["status"]
        == "inconclusive"
    )


def test_mismatched_decoding_cannot_support_finetune_improvement():
    score, automatic, sheet = _artifacts()
    automatic["run_identity"]["models"][1]["decoding"]["temperature"] = 0.7
    decision = evaluate_promotion(score, automatic, sheet)
    assert decision["status"] == "inconclusive"
    assert decision["gates"]["matched_setup"]["status"] == "inconclusive"


def test_v2_diagnostics_accept_complete_timing_and_refuse_missing_duration():
    score, automatic, sheet = _artifacts()
    automatic["schema_version"] = "envoybench-automatic-v2"
    for system in automatic["systems"].values():
        system.update(timing_complete=True, timed_question_count=40,
                      missing_duration_question_count=0)
        for row in system["question_rows"]:
            row["duration_observed"] = True
    assert evaluate_promotion(score, automatic, sheet)["status"] == "pass"

    base = automatic["systems"]["qwen_base"]
    base["question_rows"][0].update(duration_seconds=0.0, duration_observed=False)
    base.update(
        timing_complete=False, timed_question_count=39, missing_duration_question_count=1,
        total_episode_seconds=None, estimated_episode_compute_usd=None,
        observed_total_episode_seconds=39.0, estimated_observed_episode_compute_usd=0.00975,
    )
    decision = evaluate_promotion(score, automatic, sheet)
    assert decision["status"] == "inconclusive"
    assert decision["gates"]["human_answer_review"]["status"] == "pass"
    assert decision["gates"]["duration"]["status"] == "inconclusive"
    assert decision["gates"]["estimated_compute_cost"]["status"] == "inconclusive"
