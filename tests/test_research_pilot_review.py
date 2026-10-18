"""Meaningful gates for identity-blind research workflow review."""

from copy import deepcopy

import pytest

from src.eval.research_pilot_budget import RequestBudget
from src.eval.research_pilot_review import COST_COMPONENTS, build_review, score_review


@pytest.fixture
def run():
    questions = [{"id": f"P{i:02d}", "question": f"Question {i}", "arxiv_ids": ["2402.01030v4"]}
                 for i in range(1, 6)]
    results = []
    budget = RequestBudget()
    payload = {"model": "fixture", "messages": [{"role": "user", "content": "Question"}],
               "max_tokens": 1024}
    def dispatch(label):
        budget.transport(label, lambda _messages: 100, 8192, lambda *_args: {
            "usage": {"prompt_tokens": 80, "completion_tokens": 5}})("unused", payload, {}, 1)
        return len(budget.summary()["records"])
    for q in questions:
        for arm in ("direct", "delegated"):
            ids = [dispatch(f"host:{q['id']}:{arm}")]
            if arm == "delegated":
                ids.append(dispatch("qwen"))
            span = {"arxiv_id": "2402.01030v4", "start": 0, "end": 5}
            results.append({"question_id": q["id"], "arm": arm, "status": "submitted",
                "model": "SECRET_MODEL", "elapsed_seconds": 10 if arm == "direct" else 8,
                "submission": {"answer": "Answer", "claims": [{"text": "Claim", "evidence":
                    [span]}], "limitations": ["Scope"]},
                "provenance_checks": {"claims": [{"text": "Claim", "evidence": [
                    {**span, "quote": "Quote", "provenance_valid": True}]}]},
                "telemetry": {"host_requests": 1, "qwen_requests": int(arm == "delegated"),
                              "request_ids": ids}})
    return {"schema_version": "research-workflow-pilot-run-v1", "bundle_sha256": "frozen",
        "questions": questions, "review_strata": {q["id"]: "answerable" if i < 4
            else "deployment_insufficient" for i, q in enumerate(questions)},
        "results": results, "actual_model_requests": 15, "actual_qwen_requests": 5,
        "execution_kind": "live_models", "status": "complete", "budget": budget.summary(),
        "config": {"prompt_sha256": "prompt"}}


def human(packet):
    packet = deepcopy(packet)
    for row in packet["rows"]:
        row.update(reviewer="Human reviewer", reviewed_at="2026-10-17", reviewer_kind="human",
                   all_factual_claims_reviewed=True, completeness="complete", verdict="pass")
        for claim in row["claims"]:
            claim.update(materiality="major", support="supported", scope_correct=True)
    return packet


def costs(direct=10, delegated=10):
    return {"complete": True, "currency": "USD", "direct_total": direct,
        "delegated_total": delegated, "allocation_method": "Actual session charges allocated",
        "includes": sorted(COST_COMPONENTS)}


def test_packet_blinds_identity_preserves_full_answer_and_evidence(run):
    packet, key = build_review(run)
    assert packet == build_review(run)[0]
    assert "SECRET_MODEL" not in str(packet)
    assert all("arm" not in row for row in packet["rows"])
    assert {a["arm"] for a in key["assignments"]} == {"direct", "delegated"}
    row = packet["rows"][0]
    assert row["answer"] == "Answer" and row["limitations"] == ["Scope"]
    assert row["claims"][0]["evidence"][0]["quote"] == "Quote"
    assert row["reviewer_kind"] is None and row["claims"][0]["support"] is None
    assert score_review(run, packet, key, costs())["decision"] == "inconclusive"


@pytest.mark.parametrize("change", ["config", "answer", "evidence", "key", "duplicate"])
def test_binding_rejects_tampering(run, change):
    packet, key = build_review(run)
    if change == "config":
        run["config"]["prompt_sha256"] = "changed"
    elif change == "answer":
        packet["rows"][0]["answer"] = "changed"
    elif change == "evidence":
        packet["rows"][0]["claims"][0]["evidence"][0]["quote"] = "other"
    elif change == "key":
        key["assignments"][0]["arm"] = "other"
    else:
        packet["rows"][1] = deepcopy(packet["rows"][0])
    with pytest.raises(ValueError):
        score_review(run, packet, key)


def test_exact_latency_boundary_passes_only_with_complete_costs(run):
    packet, key = build_review(run)
    reviewed = human(packet)
    incomplete = score_review(run, reviewed, key)
    assert incomplete["decision"] == "inconclusive"
    assert incomplete["warm_latency_improvement"] == pytest.approx(0.2)
    assert score_review(run, reviewed, key, costs())["decision"] == "continue"


def test_cost_branch_can_pass_without_latency_improvement(run):
    for row in run["results"]:
        row["elapsed_seconds"] = 10
    packet, key = build_review(run)
    assert score_review(run, human(packet), key, costs(10, 8))["decision"] == "continue"
    assert score_review(run, human(packet), key, costs(10, 8.01))["decision"] == "revise"


@pytest.mark.parametrize("field,value", [("reviewer_kind", "agent"),
    ("reviewer", ""), ("reviewed_at", None), ("all_factual_claims_reviewed", False),
    ("completeness", "incomplete")])
def test_missing_or_agent_review_never_counts_as_completed_human_review(run, field, value):
    packet, key = build_review(run)
    packet = human(packet)
    packet["rows"][0][field] = value
    assert score_review(run, packet, key, costs())["decision"] == "inconclusive"


def delegated_row(packet, key, qid):
    blind_id = next(a["blind_id"] for a in key["assignments"]
                    if a["arm"] == "delegated" and a["question_id"] == qid)
    return next(row for row in packet["rows"] if row["blind_id"] == blind_id)


def test_quality_regression_and_unsupported_scope_fail_even_with_efficiency(run):
    packet, key = build_review(run)
    packet = human(packet)
    delegated_row(packet, key, "P01")["verdict"] = "partial"
    result = score_review(run, packet, key, costs())
    assert result["decision"] == "revise" and result["paired_regressions"] == ["P01"]
    packet = human(build_review(run)[0])
    delegated_row(packet, key, "P05")["claims"][0]["scope_correct"] = False
    assert score_review(run, packet, key, costs())["decision"] == "revise"


@pytest.mark.parametrize("kind,requests", [("fake_test", 2), ("live_models", 0)])
def test_fake_or_zero_qwen_cannot_pass(run, kind, requests):
    run.update(execution_kind=kind, actual_qwen_requests=requests)
    packet, key = build_review(run)
    assert score_review(run, human(packet), key, costs())["decision"] == "inconclusive"


def test_missing_pair_and_provenance_fail_honestly(run):
    del run["results"][0]
    packet, key = build_review(run)
    assert len(packet["rows"]) == 10
    assert score_review(run, human(packet), key, costs())["decision"] == "inconclusive"


@pytest.mark.parametrize("invalid", [None, 0, -1, float("nan"), float("inf"), True])
def test_costs_must_be_positive_finite_actual_totals(run, invalid):
    packet, key = build_review(run)
    assert score_review(run, human(packet), key, costs(delegated=invalid))[
        "decision"] == "inconclusive"


def test_minimum_quality_counts_and_separate_answerable_stratum(run):
    packet, key = build_review(run)
    packet = human(packet)
    for assignment in key["assignments"]:
        if assignment["question_id"] == "P01":
            next(r for r in packet["rows"] if r["blind_id"] == assignment["blind_id"])[
                "verdict"] = "partial"
    result = score_review(run, packet, key, costs())
    assert result["decision"] == "continue"
    assert result["systems"]["delegated"]["passes"] == 4
    assert result["delegated_answerable_passes"] == 3
    delegated_row(packet, key, "P02")["verdict"] = "partial"
    assert score_review(run, packet, key, costs())["decision"] == "revise"


def test_incorrect_quote_cannot_pass_provenance_or_primary_gate(run):
    run["results"][0]["provenance_checks"]["claims"][0]["evidence"][0]["quote"] = "Short?"
    packet, key = build_review(run)
    result = score_review(run, human(packet), key, costs())
    assert result["provenance_valid"] is False
    assert result["decision"] == "revise"


def test_incomplete_billing_components_are_not_complete_costs(run):
    packet, key = build_review(run)
    actual = costs()
    actual["includes"].remove("cleanup")
    assert score_review(run, human(packet), key, actual)["decision"] == "inconclusive"


def test_invalid_human_value_and_missing_warm_time_are_inconclusive(run):
    run["results"][0]["elapsed_seconds"] = None
    packet, key = build_review(run)
    packet = human(packet)
    packet["rows"][0]["verdict"] = []
    assert score_review(run, packet, key, costs())["decision"] == "inconclusive"


def test_duplicate_arm_result_is_rejected(run):
    run["results"].append(deepcopy(run["results"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        build_review(run)


def test_global_qwen_usage_does_not_hide_one_undelegated_question(run):
    next(row for row in run["results"] if row["arm"] == "delegated")["telemetry"][
        "qwen_requests"] = 0
    packet, key = build_review(run)
    result = score_review(run, human(packet), key, costs())
    assert result["decision"] == "inconclusive"
    assert "each delegated question requires an actual Qwen request" in result["reasons"]


@pytest.mark.parametrize("status", [None, "running", "incomplete", "interrupted"])
def test_execution_must_be_recorded_complete(run, status):
    run["status"] = status
    packet, key = build_review(run)
    assert score_review(run, human(packet), key, costs())["decision"] == "inconclusive"


@pytest.mark.parametrize("problem", ["missing", "blocked", "usage_unknown", "pending", "error",
    "usage_exceeded", "unknown_tokens", "negative_tokens", "actual_mismatch", "observed_mismatch",
    "reserved_mismatch", "noninteger_summary", "reservation_exceeded", "caps_exceeded",
    "ids_missing", "ids_duplicate", "noninteger_record_id",
    "wrong_role", "wrong_question", "counts_mismatch", "global_mismatch", "qwen_mismatch"])
def test_incomplete_or_inconsistent_ledger_never_passes_primary_decision(run, problem):
    budget = run["budget"]
    record = budget["records"][0]
    if problem == "missing":
        del run["budget"]
    elif problem == "blocked":
        budget["blocked_reason"] = "Missing usage"
    elif problem == "usage_unknown":
        budget["usage_complete"] = False
    elif problem in {"pending", "error", "usage_exceeded"}:
        record["status"] = problem
    elif problem == "unknown_tokens":
        record["provider_prompt_tokens"] = None
    elif problem == "negative_tokens":
        record["provider_completion_tokens"] = -1
    elif problem == "actual_mismatch":
        budget["actual_usage"]["prompt_tokens"] += 1
    elif problem == "observed_mismatch":
        budget["observed_usage"]["completion_tokens"] += 1
    elif problem == "reserved_mismatch":
        budget["reserved"]["requests"] += 1
    elif problem == "noninteger_summary":
        budget["actual_usage"]["prompt_tokens"] = float(budget["actual_usage"]["prompt_tokens"])
    elif problem == "reservation_exceeded":
        record["provider_prompt_tokens"] = record["local_input_tokens"] + 1
    elif problem == "caps_exceeded":
        budget["caps"]["max_requests"] = 97
    elif problem == "ids_missing":
        del run["results"][0]["telemetry"]["request_ids"]
    elif problem == "ids_duplicate":
        run["results"][1]["telemetry"]["request_ids"][0] = 1
    elif problem == "noninteger_record_id":
        record["request_id"] = True
    elif problem == "wrong_role":
        record["label"] = "other"
    elif problem == "wrong_question":
        record["label"] = "host:P05:direct"
    elif problem == "counts_mismatch":
        run["results"][0]["telemetry"]["host_requests"] += 1
    elif problem == "global_mismatch":
        run["actual_model_requests"] += 1
    else:
        run["actual_qwen_requests"] += 1
    packet, key = build_review(run)
    result = score_review(run, human(packet), key, costs())
    assert result["decision"] == "inconclusive"
    assert result["complete_request_ledger"] is False
