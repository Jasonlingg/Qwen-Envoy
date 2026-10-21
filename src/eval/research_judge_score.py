"""Compare saved judge judgments with frozen provisional labels, never train."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from src.eval.research_judge_checks import check_bundle, sha256
from src.eval.research_judge_plan import MODEL, encoded
from src.eval.research_judge_run import validate_preflight


def judgment_score(row, question):
    if set(row) != {"case_id", "fact_judgments", "all_material_assertions_supported", "reason"}:
        raise ValueError("wrong judgment schema")
    facts = row["fact_judgments"]
    ids = [f["fact_id"] for f in question["requested_facts"]]
    if (not isinstance(facts, list) or len(facts) != len(ids)
            or {f["fact_id"] for f in facts} != set(ids)
            or any(set(f) != {"fact_id", "correct_and_supported"} for f in facts)
            or not isinstance(row["reason"], str) or not 0 < len(row["reason"]) <= 500):
        raise ValueError("incomplete facts or unbounded reason")
    values = [f["correct_and_supported"] for f in facts]
    gate = row["all_material_assertions_supported"]
    if any(v is not None and type(v) is not bool for v in [*values, gate]):
        raise ValueError("fact and gate judgments must be boolean or null")
    if any(v is None for v in [*values, gate]):
        return None
    c = sum(values) / len(values)
    return (int(c == 1) if question["task"] == "target_gap" else c) if gate else 0


def compare(root, response, receipt, *, bindings_verified=False):
    root = Path(root)
    check_bundle(root)
    cases, labels, questions = [json.loads((root / f"{n}.json").read_text())
                                for n in ("cases", "labels", "questions")]
    error = None
    try:
        choices = response["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
            raise ValueError("incomplete provider response")
        content = json.loads(choices[0]["message"]["content"])
        if set(content) != {"rows"} or not isinstance(content["rows"], list):
            raise ValueError("wrong top-level judgment schema")
        rows = content["rows"]
        valid_ids = [r for r in rows if isinstance(r, dict)
                     and isinstance(r.get("case_id"), str)]
        if len(valid_ids) != len(rows):
            error = "malformed row without a case ID"
        rows = valid_ids
        counts = Counter(r["case_id"] for r in rows)
        known = {r["case_id"] for r in cases}
        if set(counts) - known:
            error = "unknown case IDs in response"
        indexed = {r["case_id"]: r for r in rows if counts[r["case_id"]] == 1}
    except (ValueError, KeyError, TypeError) as exc:
        error, counts, indexed = type(exc).__name__, Counter(), {}
    metrics = {"critical_false_acceptances": 0, "incorrect_complete_or_paraphrase_scores": 0,
        "incorrect_partial_C_scores": 0, "incorrect_valid_gap_scores": 0,
        "rewarded_answerable_blanket_abstentions": 0,
        "unknown_labels_or_judgments_on_clear_validation_cases": 0, "missing_cases": 0}
    results = []
    quality_usage = (receipt.get("usage_complete_and_within_reservation") is True
                     and receipt.get("served_model_matches_request") is True)
    for case, label in zip(cases, labels):
        cid, q = case["case_id"], questions[case["question_id"]]
        row, score, issue = indexed.get(cid), None, None
        if not row:
            metrics["missing_cases"] += 1
            issue = "duplicate_case" if counts[cid] else "missing_case"
        else:
            try:
                score = judgment_score(row, q)
                order = {f["fact_id"]: i for i, f in enumerate(q["requested_facts"])}
                row = {**row, "fact_judgments": sorted(row["fact_judgments"],
                                                     key=lambda f: order[f["fact_id"]])}
                if score is None:
                    issue = "unknown_judgment"
            except (ValueError, KeyError, TypeError):
                issue = "invalid_judgment"
        if issue:
            metrics["unknown_labels_or_judgments_on_clear_validation_cases"] += 1
        expected, group = label["expected_score"], label["group"]
        if score is not None:
            if group == "critical_negative" and score > 0:
                metrics["critical_false_acceptances"] += 1
            if group == "blanket_abstention" and score > 0:
                metrics["rewarded_answerable_blanket_abstentions"] += 1
            metric = {"complete": "incorrect_complete_or_paraphrase_scores",
                      "partial": "incorrect_partial_C_scores",
                      "valid_gap": "incorrect_valid_gap_scores"}.get(group)
            if metric and score != expected:
                metrics[metric] += 1
        results.append({"case_id": cid, "group": group, "expected_score": expected,
            "judge_score": score, "score_agrees": score == expected if score is not None else False,
            "issue": issue, "judgment": row, "expected_fact_judgments": label["fact_judgments"],
            "fact_labels_agree": row is not None
                and row.get("fact_judgments") == label["fact_judgments"]})
    passed = not any(metrics.values()) and error is None and quality_usage
    status = "sentinel_gate_failed"
    if passed:
        status = ("alias_sentinel_scores_passed_not_training_readiness" if bindings_verified
                  else "scores_match_unattested_not_a_gate_result")
    return {"status": status, "bindings_verified": bindings_verified,
        "score_thresholds_met": passed,
        "original_immutable_judge_readiness_condition_met": False,
        "metrics": metrics, "all_zero_error_limits": True, "provider_or_schema_error": error,
        "usage_complete_and_within_reservation": quality_usage,
        "scores_agree": sum(r["score_agrees"] for r in results),
        "fact_labels_agree": sum(r["fact_labels_agree"] for r in results), "total_cases": 32,
        "results": results, "human_review": "not_performed", "training_approved": False,
        "interpretation": (
            "One batch judge versus provisional model-assisted labels on clear inspected "
            "development fixtures. Not human correctness, a production verifier, "
            "statistical reliability or Qwen improvement."
        )}


def score_saved(directory, root, repo):
    """Only a fully bound saved run may report the named judge's sentinel result."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    response, receipt = [json.loads((directory / f"{name}.json").read_text())
                         for name in ("response", "receipt")]
    validate_preflight(directory, root, repo, [manifest["catalog_model"]])
    bindings = {
        "request_sha256": manifest["request_sha256"],
        "fixture_lock_sha256": sha256(Path(root) / "freeze-lock.json"),
        "manifest_sha256": sha256(directory / "manifest.json"),
        "response_sha256": hashlib.sha256(encoded(response)).hexdigest()}
    data = response if isinstance(response, dict) else {}
    usage = data.get("usage")
    prompt = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
    verified = (all(receipt.get(k) == v for k, v in bindings.items())
                and data.get("model") == receipt.get("served_model") == MODEL
                and receipt.get("usage") == usage
                and type(prompt) is int and type(completion) is int
                and 0 <= prompt <= manifest["input_tokens_reserved"]
                and 0 <= completion <= 16_384)
    if not verified:
        receipt = {**receipt, "usage_complete_and_within_reservation": False}
    result = compare(root, response, receipt, bindings_verified=verified)
    result["artifact_bindings"] = bindings
    if not verified:
        result["status"] = "inconclusive_binding_or_identity_failure"
    return result
