"""Apply the registered v6 pilot decision rule to a locked three-arm review.

Inputs are saved EnvoyBench automatic diagnostics, a *completed* blinded
answer review, its identity key, and the exact run results. This script does
not open the holdout benchmark or papers. False refusals require explicit
blind-ID adjudications in ``review['adjudications']``::

    {"type": "false_refusal", "blind_id": "R001", "value": false,
     "notes": "The answer attempts the question rather than refusing."}

Each of the 17 answerable questions needs one such decision for each model.
Missing decisions keep that promotion condition indeterminate. QASPER
answerability labels and literal refusal wording are diagnostics, not gold
semantic judgments. Compute cost is a declared-rate episode estimate, not a bill.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.envoybench.score import AUTOMATIC_VERSION, score_completed_review
from src.eval.artifacts import configuration_hash
from src.eval.hashing import sha256

SCHEMA = "qasper-v6-promotion-diagnostics-v1"
PYTHON_EXCEPTION = re.compile(
    r"^(?:Traceback \(most recent call last\):|[\w.]+(?:Error|Exception):)", re.M,
)
EXPECTED_QUESTIONS = 24
EXPECTED_SUFFICIENT = 17
EXPECTED_INSUFFICIENT = 7


def _python_error_episode(row: dict) -> bool:
    """Count Python exceptions in structured stderr or the pre-execution syntax reply."""
    for step in row["trajectory"]:
        if not isinstance(step, dict):
            continue
        observation = step.get("observation", "")
        if not isinstance(observation, str):
            continue
        if observation.startswith("SyntaxError: "):
            return True
        if "\nSTDERR:\n" in observation:
            stderr = observation.split("\nSTDERR:\n", 1)[1]
            if PYTHON_EXCEPTION.search(stderr):
                return True
    return False


def _false_refusal_adjudications(review: dict, answerable_blind_ids: set[str]) -> dict[str, bool]:
    decisions: dict[str, bool] = {}
    for item in review.get("adjudications", []):
        if not isinstance(item, dict) or item.get("type") != "false_refusal":
            continue
        blind_id = item.get("blind_id")
        if blind_id not in answerable_blind_ids or blind_id in decisions:
            raise ValueError("False-refusal adjudication has an unknown or duplicate blind ID")
        if type(item.get("value")) is not bool:
            raise ValueError(f"{blind_id} needs a boolean false-refusal decision")
        if item["value"] and not str(item.get("notes", "")).strip():
            raise ValueError(f"{blind_id} needs a reason for a false-refusal decision")
        decisions[blind_id] = item["value"]
    return decisions


def _condition(passed: bool | None, observed: object, rule: str) -> dict:
    return {"passed": passed, "observed": observed, "rule": rule}


def _ratio(candidate: float | None, base: float | None) -> float | None:
    if candidate is None or base is None or base <= 0:
        return None
    return candidate / base


def analyze(
    automatic: dict, review: dict, key: dict, results: list[dict], *,
    base_key: str, v5_key: str, candidate_key: str,
) -> dict:
    """Return ID-only diagnostics after verifying saved-run and blind-key bindings."""
    if review.get("status") != "complete":
        raise ValueError("Finish and lock the blind review before opening model identities")
    scored = score_completed_review(review, key, results=results)
    if scored.get("result_bindings_verified") is not True:
        raise ValueError("Blind assignments were not verified against saved results")
    if automatic.get("schema_version") != AUTOMATIC_VERSION:
        raise ValueError("Expected EnvoyBench automatic diagnostics, not QASPER F1")
    for field in ("benchmark_id", "benchmark_hash", "corpus_hash", "results_hash",
                  "reference_review_hash"):
        if automatic.get(field) != review.get(field):
            raise ValueError(f"Automatic diagnostics and locked review differ on {field}")
    if automatic.get("source_reference_status") != review.get(
        "source_reference_status", "human_validated",
    ):
        raise ValueError("Automatic diagnostics and locked review differ on source provenance")
    if automatic.get("excluded_questions") or review.get("excluded_questions"):
        raise ValueError("The registered 24-question decision cannot exclude questions")
    identity = automatic.get("run_identity") or {}
    if identity.get("split") != "holdout" or identity.get("max_steps") != 15:
        raise ValueError("Expected the pinned 15-step holdout protocol")

    keys = (base_key, v5_key, candidate_key)
    if len(set(keys)) != 3 or set(automatic.get("systems", {})) != set(keys):
        raise ValueError("Provide three distinct keys matching the automatic score")
    models = identity.get("models")
    if not isinstance(models, list) or {m.get("key") for m in models} != set(keys):
        raise ValueError("Automatic score lacks the exact three declared models")
    specs = {item["key"]: item for item in models}
    question_ids = automatic.get("question_ids_included")
    if (not isinstance(question_ids, list) or len(question_ids) != EXPECTED_QUESTIONS
            or len(set(question_ids)) != EXPECTED_QUESTIONS):
        raise ValueError("Automatic score is not the complete 24-question holdout")
    if configuration_hash({"results": results}) != automatic["results_hash"]:
        raise ValueError("Automatic score does not bind to the supplied results")

    assignments = key.get("assignments")
    review_rows = review.get("rows")
    if not isinstance(assignments, list) or not isinstance(review_rows, list):
        raise ValueError("Blind key and review need row lists")
    by_blind = {item["blind_id"]: item for item in assignments}
    if len(by_blind) != EXPECTED_QUESTIONS * 3 or len(review_rows) != len(by_blind):
        raise ValueError("Blind review is not a 24-by-3 matrix")
    result_by_pair = {(row["model_key"], row["question_id"]): row for row in results}
    expected_pairs = {(model, qid) for model in keys for qid in question_ids}
    if len(result_by_pair) != len(results) or set(result_by_pair) != expected_pairs:
        raise ValueError("Saved results are not a complete 24-by-3 matrix")

    by_pair: dict[tuple[str, str], dict] = {}
    labels: dict[str, str] = {}
    answerable_blind_ids: set[str] = set()
    for row in review_rows:
        blind_id, qid = row["blind_id"], row["question_id"]
        assignment = by_blind[blind_id]
        model = assignment["system"]
        if assignment["question_id"] != qid or (model, qid) in by_pair:
            raise ValueError("Blind review assignment matrix changed")
        label = row["expected_answerability"]
        if label not in {"sufficient", "insufficient"}:
            raise ValueError("Unknown selected QASPER answerability stratum")
        if qid in labels and labels[qid] != label:
            raise ValueError("Model rows disagree on question answerability")
        labels[qid] = label
        if label == "sufficient":
            answerable_blind_ids.add(blind_id)
        by_pair[(model, qid)] = row
    if set(by_pair) != expected_pairs or set(labels) != set(question_ids):
        raise ValueError("Blind review does not cover all model/question pairs")
    if (sum(label == "sufficient" for label in labels.values()) != EXPECTED_SUFFICIENT
            or sum(label == "insufficient" for label in labels.values())
            != EXPECTED_INSUFFICIENT):
        raise ValueError("Blind review strata differ from the pre-registered 17/7 split")
    adjudications = _false_refusal_adjudications(review, answerable_blind_ids)

    summaries = {}
    total_duration_by_model: dict[str, float | None] = {}
    total_cost_by_model: dict[str, float | None] = {}
    for model in keys:
        system = automatic["systems"][model]
        saved = [result_by_pair[(model, qid)] for qid in question_ids]
        times = [row["duration_seconds"] for row in saved]
        if any(type(value) not in {int, float} or not math.isfinite(value) or value < 0
               for value in times):
            raise ValueError(f"{model} has invalid saved episode duration")
        timing_complete = all(value > 0 for value in times)
        mean_seconds = round(statistics.mean(times), 4) if timing_complete else None
        total_seconds = round(sum(times), 4) if timing_complete else None
        total_duration_by_model[model] = sum(times) if timing_complete else None
        if (system.get("question_count") != EXPECTED_QUESTIONS
                or system.get("timing_complete") is not timing_complete
                or system.get("mean_duration_seconds") != mean_seconds
                or system.get("total_episode_seconds") != total_seconds):
            raise ValueError(f"{model} automatic duration diagnostics differ from results")
        price = specs[model].get("serving_hourly_usd")
        if price is not None and (type(price) not in {int, float} or
                                  not math.isfinite(price) or price < 0):
            raise ValueError(f"{model} declared hourly compute price is invalid")
        total_cost = (
            round(total_seconds * price / 3600, 6)
            if timing_complete and price is not None else None
        )
        total_cost_by_model[model] = (
            sum(times) * price / 3600 if timing_complete and price is not None else None
        )
        if system.get("estimated_episode_compute_usd") != total_cost:
            raise ValueError(f"{model} automatic cost estimate differs from saved results")
        answerable = [qid for qid in question_ids if labels[qid] == "sufficient"]
        insufficient = [qid for qid in question_ids if labels[qid] == "insufficient"]
        false_refusal_ids = [
            by_pair[(model, qid)]["blind_id"] for qid in answerable
        ]
        missing = [blind_id for blind_id in false_refusal_ids
                   if blind_id not in adjudications]
        summaries[model] = {
            "answerable_passes": sum(by_pair[(model, qid)]["verdict"] == "pass"
                                     for qid in answerable),
            "answerable_questions": EXPECTED_SUFFICIENT,
            "insufficient_passes": sum(by_pair[(model, qid)]["verdict"] == "pass"
                                       for qid in insufficient),
            "insufficient_questions": EXPECTED_INSUFFICIENT,
            "reviewed_false_refusals": (
                sum(adjudications[blind_id] for blind_id in false_refusal_ids)
                if not missing else None
            ),
            "false_refusal_adjudicated": len(false_refusal_ids) - len(missing),
            "missing_false_refusal_blind_ids": missing,
            "python_error_episodes": sum(_python_error_episode(row) for row in saved),
            "mean_episode_seconds": mean_seconds,
            "estimated_mean_episode_compute_usd": (
                round(total_cost / EXPECTED_QUESTIONS, 8)
                if total_cost is not None else None
            ),
            "serving_hourly_usd": price,
            "serving_hardware": specs[model].get("serving_hardware"),
            "serving_runtime": specs[model].get("serving_runtime"),
        }

    base = summaries[base_key]
    v5 = summaries[v5_key]
    candidate = summaries[candidate_key]
    latency_ratio = _ratio(
        total_duration_by_model[candidate_key], total_duration_by_model[base_key],
    )
    cost_ratio = _ratio(total_cost_by_model[candidate_key], total_cost_by_model[base_key])
    hardware = {summaries[model]["serving_hardware"] for model in keys}
    decodings = {json.dumps(specs[model].get("decoding"), sort_keys=True) for model in keys}
    hardware_matched = len(hardware) == 1 and "unreported" not in hardware and None not in hardware
    decoding_matched = len(decodings) == 1 and "null" not in decodings
    false_refusal_pass = (
        candidate["reviewed_false_refusals"] <= base["reviewed_false_refusals"]
        if candidate["reviewed_false_refusals"] is not None
        and base["reviewed_false_refusals"] is not None else None
    )
    conditions = {
        "supported_answer_gain_vs_base": _condition(
            candidate["answerable_passes"] >= base["answerable_passes"] + 4,
            candidate["answerable_passes"] - base["answerable_passes"], ">= 4",
        ),
        "supported_answer_gain_vs_v5": _condition(
            candidate["answerable_passes"] > v5["answerable_passes"],
            candidate["answerable_passes"] - v5["answerable_passes"], "> 0",
        ),
        "false_refusals_no_increase_vs_base": _condition(
            false_refusal_pass,
            {"base": base["reviewed_false_refusals"],
             "candidate": candidate["reviewed_false_refusals"]}, "candidate <= base",
        ),
        "python_errors_no_increase_vs_base": _condition(
            candidate["python_error_episodes"] <= base["python_error_episodes"],
            {"base": base["python_error_episodes"],
             "candidate": candidate["python_error_episodes"]}, "candidate <= base",
        ),
        "latency_within_2x_base": _condition(
            latency_ratio <= 2 if latency_ratio is not None else None,
            latency_ratio, "<= 2x",
        ),
        "estimated_compute_within_2x_base": _condition(
            cost_ratio <= 2 if cost_ratio is not None else None,
            cost_ratio, "<= 2x",
        ),
        "matched_hardware": _condition(hardware_matched, sorted(str(x) for x in hardware),
                                       "same reported GPU class"),
        "matched_decoding": _condition(decoding_matched, sorted(decodings),
                                       "same decoding settings"),
    }
    passed = [item["passed"] for item in conditions.values()]
    if False in passed:
        decision = "does_not_meet_registered_rule"
    elif None in passed:
        decision = "indeterminate"
    else:
        decision = "registered_rule_met_provisionally"

    paired = []
    for qid in question_ids:
        paired.append({
            "question_id": qid,
            "expected_answerability": labels[qid],
            "systems": {
                model: {
                    "verdict": by_pair[(model, qid)]["verdict"],
                    "status": result_by_pair[(model, qid)]["status"],
                    "false_refusal": adjudications.get(by_pair[(model, qid)]["blind_id"]),
                    "python_error_episode": _python_error_episode(result_by_pair[(model, qid)]),
                    "duration_seconds": result_by_pair[(model, qid)]["duration_seconds"],
                } for model in keys
            },
        })
    return {
        "schema_version": SCHEMA,
        "benchmark_id": review["benchmark_id"],
        "benchmark_hash": review["benchmark_hash"],
        "corpus_hash": review["corpus_hash"],
        "results_hash": review["results_hash"],
        "source_reference_status": scored["source_reference_status"],
        "review_provenance": scored["review_provenance"],
        "question_count": EXPECTED_QUESTIONS,
        "answerable_count": EXPECTED_SUFFICIENT,
        "insufficient_count": EXPECTED_INSUFFICIENT,
        "roles": {"base": base_key, "v5": v5_key, "candidate": candidate_key},
        "systems": summaries,
        "paired_question_diagnostics": paired,
        "promotion": {"decision": decision, "conditions": conditions},
        "claim_boundary": (
            "Selected QASPER answerability labels are not infallible gold. PASS verdicts "
            "inherit the completed review's human/model-assisted provenance. Missing "
            "false-refusal adjudications make that condition indeterminate. Python errors "
            "are detected only in saved syntax replies and structured stderr. Episode "
            "compute cost is a declared GPU-rate estimate, not an invoice. A rule pass "
            "is provisional until independent review or replication."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automatic", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--base-key", required=True)
    parser.add_argument("--v5-key", required=True)
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    review = json.loads(args.review.read_text())
    if review.get("status") != "complete":
        parser.error("Finish and lock the blind review before opening model identities")
    automatic = json.loads(args.automatic.read_text())
    key = json.loads(args.blind_key.read_text())
    results = json.loads(args.results.read_text())
    report = analyze(
        automatic, review, key, results,
        base_key=args.base_key, v5_key=args.v5_key, candidate_key=args.candidate_key,
    )
    report["inputs"] = {
        "automatic_sha256": sha256(args.automatic),
        "completed_review_sha256": sha256(args.review),
        "blind_key_sha256": sha256(args.blind_key),
        "results_sha256": sha256(args.results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["promotion"]["decision"],
        "answerable_passes": {key: row["answerable_passes"]
                              for key, row in report["systems"].items()},
    }))


if __name__ == "__main__":
    main()
