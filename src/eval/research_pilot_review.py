"""Identity-blind, human-reviewed workflow pilot; provenance is not support."""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime
from statistics import median

ARMS = {"direct", "delegated"}
VERDICTS = {"fail": 0, "partial": 1, "pass": 2}
SUPPORT = {"supported", "partly_supported", "unsupported", "contradicted", "not_assessable"}
ANSWER_REVIEW = {"reviewer", "reviewed_at", "reviewer_kind", "all_factual_claims_reviewed",
                 "completeness", "verdict", "notes"}
CLAIM_REVIEW = {"materiality", "support", "scope_correct", "notes"}
COST_COMPONENTS = {"host_inference", "worker_inference", "gpu_loading", "gpu_active",
                   "gpu_idle", "cleanup", "session_storage"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _index(run):
    questions = run.get("questions", [])
    if (run.get("schema_version") != "research-workflow-pilot-run-v1"
            or not isinstance(questions, list) or not questions):
        raise ValueError("Invalid pilot run schema or questions")
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs")
    results = {}
    for row in run.get("results", []):
        identity = row.get("question_id"), row.get("arm")
        if identity[0] not in ids or identity[1] not in ARMS or identity in results:
            raise ValueError("Unknown or duplicate question/arm result")
        results[identity] = row
    return questions, results


def _content(row):
    value = {k: v for k, v in row.items() if k not in ANSWER_REVIEW}
    value["claims"] = [{k: v for k, v in claim.items() if k not in CLAIM_REVIEW}
                       for claim in row["claims"]]
    return value


def _claims(result):
    submission = result.get("submission") or {}
    checks = result.get("provenance_checks") or result.get("checks") or {}
    checked = checks.get("claims", [])
    normalized = []
    for index, claim in enumerate(submission.get("claims", [])):
        checked_claim = checked[index] if index < len(checked) else {}
        verified = checked_claim.get("text") == claim["text"]
        evidence = []
        for span in claim.get("evidence", []):
            match = next((item for item in checked_claim.get("evidence", [])
                          if all(item.get(k) == span.get(k) for k in
                                 ("arxiv_id", "start", "end"))), {})
            quote = match.get("quote")
            exact = (verified and match.get("provenance_valid") is True
                     and isinstance(quote, str) and type(span.get("start")) is int
                     and type(span.get("end")) is int
                     and 0 <= span["start"] < span["end"]
                     and len(quote) == span["end"] - span["start"])
            evidence.append({**{k: span.get(k) for k in ("arxiv_id", "start", "end")},
                             "quote": quote, "provenance_valid": exact})
        normalized.append({"claim_id": f"C{index + 1}", "text": claim["text"],
            "evidence": evidence, "materiality": None, "support": None,
            "scope_correct": None, "notes": ""})
    return normalized


def build_review(run: dict, seed: int = 42) -> tuple[dict, dict]:
    """Return a blank human packet and separate identity key bound to the full run."""
    questions, results = _index(run)
    if type(seed) is not int:
        raise ValueError("Review seed must be an integer")
    run_hash, rng, rows, assignments = _hash(run), random.Random(seed), [], []
    for question in questions:
        order = sorted(ARMS)
        rng.shuffle(order)
        for arm in order:
            result = results.get((question["id"], arm), {})
            submission = result.get("submission") or {}
            row = {"blind_id": f"R{len(rows) + 1:03d}", "question_id": question["id"],
                "question": question["question"], "answer": submission.get("answer", ""),
                "limitations": submission.get("limitations", []), "claims": _claims(result),
                "reviewer": None, "reviewed_at": None, "reviewer_kind": None,
                "all_factual_claims_reviewed": None, "completeness": None,
                "verdict": None, "notes": ""}
            assignments.append({"blind_id": row["blind_id"], "question_id": question["id"],
                                "arm": arm, "content_sha256": _hash(_content(row))})
            rows.append(row)
    return ({"schema_version": "research-workflow-pilot-review-v1", "run_sha256": run_hash,
             "warning": "Content may reveal workflow. Exact quotes prove provenance, not support. "
                        "Human identity and review are process attestations, not authentication.",
             "rows": rows},
            {"schema_version": "research-workflow-pilot-key-v1", "run_sha256": run_hash,
             "seed": seed, "assignments": assignments})


def _validate_binding(run, packet, key):
    expected, expected_key = build_review(run, key.get("seed"))
    if key != expected_key or {k: v for k, v in packet.items() if k != "rows"} != {
            k: v for k, v in expected.items() if k != "rows"}:
        raise ValueError("Review/key no longer bind to this complete run")
    rows = packet.get("rows", [])
    by_id = {row["blind_id"]: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != {row["blind_id"] for row in expected["rows"]}:
        raise ValueError("Review rows are missing or duplicated")
    for assignment in key["assignments"]:
        if _hash(_content(by_id[assignment["blind_id"]])) != assignment["content_sha256"]:
            raise ValueError("Answer or evidence changed after review preparation")
    return by_id


def _human_complete(row):
    try:
        datetime.fromisoformat(row["reviewed_at"])
    except (ValueError, TypeError, KeyError):
        return False
    return (isinstance(row.get("reviewer"), str) and bool(row["reviewer"].strip())
        and row.get("reviewer_kind") == "human" and row.get("all_factual_claims_reviewed") is True
        and isinstance(row.get("completeness"), str)
        and row["completeness"] in {"complete", "incomplete"}
        and isinstance(row.get("verdict"), str) and row["verdict"] in VERDICTS
        and not (row["verdict"] == "pass" and row["completeness"] != "complete")
        and all(isinstance(c.get("materiality"), str) and c["materiality"] in {"major", "minor"}
                and isinstance(c.get("support"), str) and c["support"] in SUPPORT
                and type(c.get("scope_correct")) is bool for c in row["claims"]))


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _complete_ledger(run, results):
    """Require every actual attempt and its usage to reconcile across one ledger."""
    def integer(value, minimum=0):
        return type(value) is int and value >= minimum
    try:
        budget = run["budget"]
        if (budget["schema_version"] != "research-pilot-local-budget-v1"
                or budget["usage_complete"] is not True or budget["blocked_reason"] is not None):
            return False
        records, caps = budget["records"], budget["caps"]
        maxima = {"max_requests": 96, "max_input_tokens": 300_000,
                  "max_output_tokens": 98_304, "max_tokens_per_request": 1024}
        if (not isinstance(records, list) or not records or any(
                not integer(caps.get(k), 1) or caps[k] > maximum
                for k, maximum in maxima.items())):
            return False
        if (any(not integer(r["request_id"], 1) for r in records)
                or [r["request_id"] for r in records] != list(range(1, len(records) + 1))):
            return False
        for row in records:
            values = [row[k] for k in ("local_input_tokens", "reserved_output_tokens",
                                      "provider_prompt_tokens", "provider_completion_tokens")]
            if (row["status"] != "ok" or row.get("exception_class") is not None
                    or not all(integer(value) for value in values)
                    or not 0 < values[0] or not 0 < values[1] <= caps["max_tokens_per_request"]
                    or values[2] > values[0] or values[3] > values[1]):
                return False
        reserved = {"requests": len(records),
                    "input_tokens": sum(r["local_input_tokens"] for r in records),
                    "output_tokens": sum(r["reserved_output_tokens"] for r in records)}
        actual = {"prompt_tokens": sum(r["provider_prompt_tokens"] for r in records),
                  "completion_tokens": sum(r["provider_completion_tokens"] for r in records)}
        if (any(not isinstance(budget.get(k), dict) or any(not integer(value)
                for value in budget[k].values())
                for k in ("reserved", "actual_usage", "observed_usage"))
                or budget["reserved"] != reserved or budget["actual_usage"] != actual
                or budget["observed_usage"] != actual
                or any(value > caps[f"max_{key}"] for key, value in reserved.items())):
            return False
        assigned = []
        for (qid, arm), result in results.items():
            telemetry = result["telemetry"]
            ids = telemetry["request_ids"]
            if not isinstance(ids, list) or not ids or any(
                    not integer(i, 1) or i > len(records) for i in ids):
                return False
            labels = [records[i - 1]["label"] for i in ids]
            host_label = f"host:{qid}:{arm}"
            host, qwen = labels.count(host_label), labels.count("qwen")
            if (not 1 <= host <= 8 or not integer(telemetry["host_requests"])
                    or not integer(telemetry["qwen_requests"])
                    or telemetry["host_requests"] != host or telemetry["qwen_requests"] != qwen
                    or host + qwen != len(labels) or (arm == "direct" and qwen)
                    or (arm == "delegated" and not qwen)):
                return False
            assigned.extend(ids)
        return (sorted(assigned) == list(range(1, len(records) + 1))
                and type(run["actual_model_requests"]) is int
                and run["actual_model_requests"] == len(records)
                and type(run["actual_qwen_requests"]) is int
                and run["actual_qwen_requests"] == sum(r["label"] == "qwen" for r in records))
    except (KeyError, TypeError, IndexError):
        return False


def score_review(run: dict, packet: dict, key: dict, costs: dict | None = None) -> dict:
    """Score complete human pairs; costs require actual totals and component attestations.

    costs: complete=True, currency, direct_total, delegated_total, allocation_method,
    includes=[all COST_COMPONENTS]. Costs are external declarations, never inferred.
    """
    by_id = _validate_binding(run, packet, key)
    questions, results = _index(run)
    pairs = {(a["question_id"], a["arm"]): by_id[a["blind_id"]] for a in key["assignments"]}
    reasons, latency = [], {}
    for arm in sorted(ARMS):
        times = [results.get((q["id"], arm), {}).get("elapsed_seconds") for q in questions]
        latency[arm] = median(times) if all(_positive(t) for t in times) else None
    latency_gain = (1 - latency["delegated"] / latency["direct"]
                    if all(value is not None for value in latency.values()) else None)
    complete_cost = (isinstance(costs, dict) and costs.get("complete") is True
        and all(_positive(costs.get(f"{arm}_total")) for arm in ARMS)
        and isinstance(costs.get("currency"), str) and bool(costs["currency"].strip())
        and isinstance(costs.get("allocation_method"), str)
        and bool(costs["allocation_method"].strip())
        and isinstance(costs.get("includes"), list)
        and all(isinstance(item, str) for item in costs["includes"])
        and COST_COMPONENTS <= set(costs["includes"]))
    cost_gain = 1 - costs["delegated_total"] / costs["direct_total"] if complete_cost else None
    strata = run.get("review_strata", {})
    if run.get("status") != "complete":
        reasons.append("the recorded pilot execution must be complete")
    ledger_complete = _complete_ledger(run, results)
    if not ledger_complete:
        reasons.append("complete unblocked budget usage must reconcile with every actual request")
    if (len(questions) != 5 or set(strata) != {q["id"] for q in questions}
            or list(strata.values()).count("answerable") != 4
            or list(strata.values()).count("deployment_insufficient") != 1):
        reasons.append("question strata do not match the declared five-question pilot")
    if any(not _human_complete(row) for row in pairs.values()):
        reasons.append("complete paired human claim review is required")
    if any(results.get(identity, {}).get("status") != "submitted" or not row["answer"]
           for identity, row in pairs.items()):
        reasons.append("all paired live answers must be submitted")
    if (run.get("execution_kind") != "live_models"
            or type(run.get("actual_qwen_requests")) is not int
            or type(run.get("actual_model_requests")) is not int
            or not 0 < run["actual_qwen_requests"] <= run["actual_model_requests"]):
        reasons.append("fake, unexecuted, or zero-Qwen runs cannot support the primary decision")
    if any(type(results.get((q["id"], "delegated"), {}).get("telemetry", {}).get(
            "qwen_requests")) is not int or results.get((q["id"], "delegated"), {}).get(
            "telemetry", {}).get("qwen_requests", 0) <= 0 for q in questions):
        reasons.append("each delegated question requires an actual Qwen request")
    if not complete_cost:
        reasons.append("complete positive actual per-arm costs are required")
    if latency_gain is None:
        reasons.append("complete positive warm durations are required")
    summaries = {arm: {"passes": sum(row.get("verdict") == "pass" for (qid, a), row in
                                    pairs.items() if a == arm)} for arm in sorted(ARMS)}
    answerable = sum(row.get("verdict") == "pass" for (qid, arm), row in pairs.items()
                     if arm == "delegated" and strata.get(qid) == "answerable")
    regressions = [q["id"] for q in questions if all(isinstance(
        pairs[(q["id"], a)].get("verdict"), str) and pairs[(q["id"], a)]["verdict"]
        in VERDICTS for a in ARMS) and VERDICTS[pairs[(q["id"], "delegated")]["verdict"]]
        < VERDICTS[pairs[(q["id"], "direct")]["verdict"]]]
    major = [row["blind_id"] for (qid, arm), row in pairs.items() if arm == "delegated"
             and any(c.get("materiality") == "major" and (c.get("support") != "supported"
                     or c.get("scope_correct") is not True) for c in row["claims"])]
    provenance = all(c["evidence"] and all(e["provenance_valid"] for e in c["evidence"])
                     for row in pairs.values() for c in row["claims"])
    quality = summaries["delegated"]["passes"] >= 4 and answerable >= 3 and not (
        regressions or major) and provenance
    efficiency = any(gain is not None and gain >= 0.2 - 1e-12
                     for gain in (cost_gain, latency_gain))
    decision = "inconclusive" if reasons else "continue" if quality and efficiency else "revise"
    return {"schema_version": "research-workflow-pilot-score-v1", "decision": decision,
        "reasons": reasons, "run_sha256": key["run_sha256"], "systems": summaries,
        "delegated_answerable_passes": answerable, "paired_regressions": regressions,
        "major_unsupported_answers": major, "provenance_valid": provenance,
        "quality_gate_passed": quality if not reasons else None,
        "efficiency_gate_passed": efficiency if not reasons else None,
        "median_warm_seconds": latency, "warm_latency_improvement": latency_gain,
        "complete_costs": bool(complete_cost), "cost_improvement": cost_gain,
        "complete_request_ledger": ledger_complete,
        "costs": costs if complete_cost else None,
        "limitation": "Exploratory workflow pilot; no fine-tuning or general superiority claim."}
