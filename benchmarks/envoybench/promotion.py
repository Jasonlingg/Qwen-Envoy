"""Apply the predeclared base-versus-v5 EnvoyBench promotion rule.

This consumes artifacts produced by ``score prepare`` and ``score score``. It
does not turn model-assisted reviews or automatic evidence checks into human
supported-answer judgments. Artifact hashes must bind the two scores to the
same frozen reference sheet and run results.
"""

from __future__ import annotations

from datetime import date
from math import comb

from src.eval.artifacts import configuration_hash

RULE_VERSION = "envoybench-base-v5-promotion-v1"
FULL_QUESTION_COUNT = 40
MIN_VALIDATED_QUESTIONS = 35


def _gate(status: str, detail: str, **values: object) -> dict:
    return {"status": status, "detail": detail, **values}


def _result(base_key: str, candidate_key: str, gates: dict[str, dict]) -> dict:
    reasons = [
        f"{name}: {gate['detail']}" for name, gate in gates.items() if gate["status"] != "pass"
    ]
    status = (
        "inconclusive"
        if any(gate["status"] == "inconclusive" for gate in gates.values())
        else "fail"
        if any(gate["status"] == "fail" for gate in gates.values())
        else "pass"
    )
    return {
        "schema_version": RULE_VERSION,
        "base_model": base_key,
        "candidate_model": candidate_key,
        "status": status,
        "gates": gates,
        "reasons": reasons,
    }


def _nonnegative_number(value: object) -> bool:
    return type(value) in (int, float) and 0 <= value < float("inf")


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _iso_date(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    tail = min(left_only, right_only)
    return min(1.0, 2 * sum(comb(discordant, k) for k in range(tail + 1)) / 2**discordant)


def _reference_ids(sheet: dict) -> tuple[set[str], set[str]] | None:
    rows = sheet.get("questions")
    if not isinstance(rows, list) or len(rows) != FULL_QUESTION_COUNT:
        return None
    all_ids: set[str] = set()
    valid_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("question_id"), str):
            return None
        question_id = row["question_id"]
        if not question_id or question_id in all_ids:
            return None
        all_ids.add(question_id)
        if row.get("decision") == "reference_valid":
            valid_ids.add(question_id)
        elif row.get("decision") == "unscorable":
            if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                return None
        else:
            return None
    return all_ids, valid_ids


def _automatic_rows(automatic: dict, model_key: str, all_ids: set[str]) -> list[dict] | None:
    systems = automatic.get("systems")
    if not isinstance(systems, dict):
        return None
    system = systems.get(model_key)
    if not isinstance(system, dict) or system.get("question_count") != FULL_QUESTION_COUNT:
        return None
    rows = system.get("question_rows")
    if not isinstance(rows, list) or len(rows) != FULL_QUESTION_COUNT:
        return None
    ids = [row.get("question_id") for row in rows if isinstance(row, dict)]
    if len(ids) != FULL_QUESTION_COUNT or set(ids) != all_ids or len(set(ids)) != len(ids):
        return None
    if any(
        type(row.get("execution_error_episode")) is not bool
        or type(row.get("submitted")) is not bool
        or not _nonnegative_number(row.get("duration_seconds"))
        for row in rows
    ):
        return None
    return rows


def _declared_cost(automatic: dict, model_key: str, rows: list[dict]) -> float | None:
    identity = automatic.get("run_identity")
    models = identity.get("models") if isinstance(identity, dict) else None
    if not isinstance(models, list):
        return None
    matches = [item for item in models if isinstance(item, dict) and item.get("key") == model_key]
    if len(matches) != 1:
        return None
    model = matches[0]
    if any(
        not isinstance(model.get(field), str)
        or not model[field].strip()
        or model[field] == "unreported"
        for field in ("serving_hardware", "serving_runtime")
    ):
        return None
    price = model.get("serving_hourly_usd")
    if not _nonnegative_number(price) or price == 0:
        return None
    system = automatic["systems"][model_key]
    total = system.get("total_episode_seconds")
    recorded_cost = system.get("estimated_episode_compute_usd")
    measured_total = round(sum(row["duration_seconds"] for row in rows), 4)
    if (
        not _nonnegative_number(total)
        or not _nonnegative_number(recorded_cost)
        or abs(total - measured_total) > 0.0000001
        or abs(recorded_cost - round(total * price / 3600, 6)) > 0.000000001
    ):
        return None
    return total * price / 3600


def evaluate_promotion(
    reviewed_score: dict,
    automatic: dict,
    reference_sheet: dict,
    *,
    base_key: str = "qwen_base",
    candidate_key: str = "qwen_v5",
) -> dict:
    """Return a transparent pass/fail/inconclusive decision for the frozen 40.

    ``pass`` requires every gate. A missing prerequisite, including cost, makes
    the whole decision inconclusive even if another measured gate fails. The
    caller must supply files from one scoring bundle; this function verifies
    their recorded benchmark, reference, and results hashes.
    """
    gates: dict[str, dict] = {}
    if (
        not isinstance(reviewed_score, dict)
        or not isinstance(automatic, dict)
        or not isinstance(reference_sheet, dict)
        or base_key == candidate_key
    ):
        gates["artifacts"] = _gate(
            "inconclusive", "Expected three artifact objects and distinct model keys."
        )
        return _result(base_key, candidate_key, gates)

    if (
        reference_sheet.get("schema_version") != "envoybench-reference-review-v1"
        or reference_sheet.get("status") != "complete"
        or not isinstance(reference_sheet.get("reviewer_id"), str)
        or not reference_sheet["reviewer_id"].strip()
        or not _iso_date(reference_sheet.get("reviewed_at"))
    ):
        gates["reference_review"] = _gate(
            "inconclusive", "Frozen source-reference sheet needs completed human review metadata."
        )
        return _result(base_key, candidate_key, gates)
    ids = _reference_ids(reference_sheet)
    if ids is None:
        gates["reference_review"] = _gate(
            "inconclusive", "Reference sheet must decide each of the 40 unique questions."
        )
        return _result(base_key, candidate_key, gates)
    all_ids, valid_ids = ids
    gates["reference_review"] = _gate("pass", "All 40 source references have review decisions.")

    provenance = reviewed_score.get("review_provenance")
    if (
        reviewed_score.get("schema_version") != "envoybench-reviewed-score-v1"
        or automatic.get("schema_version") not in {
            "envoybench-automatic-v1", "envoybench-automatic-v2"
        }
        or not isinstance(provenance, dict)
        or provenance.get("kind") != "human"
        or not isinstance(provenance.get("reviewer_id"), str)
        or not provenance["reviewer_id"].strip()
        or not _iso_date(reviewed_score.get("reviewed_at"))
        or not isinstance(reviewed_score.get("human"), dict)
        or reviewed_score.get("provisional_model_assisted") is not None
    ):
        gates["human_answer_review"] = _gate(
            "inconclusive",
            "A completed human-reviewed answer score is required; "
            "model-assisted scores are provisional.",
        )
        return _result(base_key, candidate_key, gates)
    gates["human_answer_review"] = _gate(
        "pass", "Supported-answer verdicts are from the human score."
    )

    reference_hash = configuration_hash(reference_sheet)
    identity_fields = ("benchmark_id", "benchmark_hash", "corpus_hash")
    if (
        any(
            not reference_sheet.get(field)
            or reviewed_score.get(field) != reference_sheet[field]
            or automatic.get(field) != reference_sheet[field]
            for field in identity_fields
        )
        or reviewed_score.get("reference_review_hash") != reference_hash
        or automatic.get("reference_review_hash") != reference_hash
        or not reviewed_score.get("results_hash")
        or automatic.get("results_hash") != reviewed_score["results_hash"]
    ):
        gates["artifact_binding"] = _gate(
            "inconclusive",
            "Benchmark, corpus, reference, or run-results hashes are missing or differ.",
        )
        return _result(base_key, candidate_key, gates)
    gates["artifact_binding"] = _gate(
        "pass", "Scores match the frozen references and same run results."
    )

    run_identity = automatic.get("run_identity")
    if not isinstance(run_identity, dict) or run_identity.get("split") != "test_candidate":
        gates["test_split"] = _gate(
            "inconclusive", "Promotion requires a full test_candidate run, not a development split."
        )
        return _result(base_key, candidate_key, gates)
    gates["test_split"] = _gate("pass", "Automatic diagnostics are from test_candidate.")

    models = run_identity.get("models")
    declared = {
        item.get("key"): item for item in models if isinstance(item, dict)
    } if isinstance(models, list) else {}
    comparable = ("decoding", "extra_body", "send_seed", "serving_hardware", "serving_runtime")
    base_declared, candidate_declared = declared.get(base_key), declared.get(candidate_key)
    if (
        not isinstance(base_declared, dict)
        or not isinstance(candidate_declared, dict)
        or not isinstance(models, list)
        or len(declared) != len(models)
        or any(
            field not in base_declared
            or field not in candidate_declared
            or base_declared[field] != candidate_declared[field]
            for field in comparable
        )
        or any(
            not isinstance(base_declared.get(field), str)
            or base_declared[field] == "unreported"
            or not base_declared[field].strip()
            for field in ("serving_hardware", "serving_runtime")
        )
    ):
        gates["matched_setup"] = _gate(
            "inconclusive",
            "Base and candidate need identical decoding, thinking settings, seed behavior, "
            "serving hardware, and runtime.",
        )
        return _result(base_key, candidate_key, gates)
    gates["matched_setup"] = _gate("pass", "Both models used the same evaluation setup.")

    scorable_ids = automatic.get("scorable_question_ids")
    excluded = reviewed_score.get("excluded_questions")
    if (
        not isinstance(scorable_ids, list)
        or not isinstance(excluded, list)
        or any(not isinstance(item, str) for item in scorable_ids)
        or any(not isinstance(item, dict) for item in excluded)
        or set(scorable_ids) != valid_ids
        or len(scorable_ids) != len(valid_ids)
        or {item.get("question_id") for item in excluded} != all_ids - valid_ids
        or len(excluded) != len(all_ids - valid_ids)
    ):
        gates["question_binding"] = _gate(
            "inconclusive",
            "Scorable or excluded question IDs do not match the frozen review sheet.",
        )
        return _result(base_key, candidate_key, gates)
    gates["question_binding"] = _gate(
        "pass", "Scored question IDs match the frozen review decisions."
    )

    count = len(valid_ids)
    gates["validated_questions"] = _gate(
        "pass" if count >= MIN_VALIDATED_QUESTIONS else "inconclusive",
        f"{count}/{FULL_QUESTION_COUNT} references validated; "
        f"at least {MIN_VALIDATED_QUESTIONS} required.",
        observed=count,
        minimum=MIN_VALIDATED_QUESTIONS,
    )

    human = reviewed_score["human"]
    systems = human.get("systems")
    paired = human.get("paired_pass_tests")
    if not isinstance(systems, dict) or not isinstance(paired, dict):
        gates["paired_verdicts"] = _gate(
            "inconclusive", "Human score lacks paired system verdicts."
        )
        return _result(base_key, candidate_key, gates)
    base = systems.get(base_key)
    candidate = systems.get(candidate_key)
    if not isinstance(base, dict) or not isinstance(candidate, dict):
        gates["paired_verdicts"] = _gate(
            "inconclusive", "Human score lacks the named base or candidate model."
        )
        return _result(base_key, candidate_key, gates)
    for system in (base, candidate):
        if (
            any(
                not _nonnegative_int(system.get(verdict)) for verdict in ("pass", "partial", "fail")
            )
            or sum(system[verdict] for verdict in ("pass", "partial", "fail")) != count
        ):
            gates["paired_verdicts"] = _gate(
                "inconclusive", "Human verdict totals do not equal the validated question count."
            )
            return _result(base_key, candidate_key, gates)

    left, right = sorted((base_key, candidate_key))
    pair = paired.get(f"{left}_vs_{right}")
    if not isinstance(pair, dict):
        gates["paired_verdicts"] = _gate(
            "inconclusive", "Human score lacks the exact paired pass test."
        )
        return _result(base_key, candidate_key, gates)
    left_only, right_only = pair.get("left_only_pass"), pair.get("right_only_pass")
    expected_difference = systems[right]["pass"] - systems[left]["pass"]
    if (
        not _nonnegative_int(left_only)
        or not _nonnegative_int(right_only)
        or left_only + right_only > count
        or right_only - left_only != expected_difference
        or pair.get("scorable_question_count") != count
    ):
        gates["paired_verdicts"] = _gate(
            "inconclusive", "Discordant pass counts conflict with human verdict totals."
        )
        return _result(base_key, candidate_key, gates)
    p_value = _exact_mcnemar(left_only, right_only)
    reported_p = pair.get("exact_mcnemar_two_sided_p")
    if not _nonnegative_number(reported_p) or abs(reported_p - round(p_value, 6)) > 0.000001:
        gates["paired_verdicts"] = _gate(
            "inconclusive", "Reported McNemar p-value conflicts with paired counts."
        )
        return _result(base_key, candidate_key, gates)
    gates["paired_verdicts"] = _gate(
        "pass", "Human verdict totals and paired counts are consistent."
    )

    gain = (candidate["pass"] - base["pass"]) / count if count else 0
    gates["pass_gain"] = _gate(
        "pass" if (candidate["pass"] - base["pass"]) * 100 >= 15 * count else "fail",
        f"Pass-rate gain is {gain:+.1%}; required at least +15.0 percentage points.",
        observed=gain,
        minimum=0.15,
    )
    gates["mcnemar"] = _gate(
        "pass" if p_value < 0.05 else "fail",
        f"Two-sided exact McNemar p={p_value:.6g}; required p<0.05.",
        observed=p_value,
        maximum_exclusive=0.05,
    )

    base_rows = _automatic_rows(automatic, base_key, all_ids)
    candidate_rows = _automatic_rows(automatic, candidate_key, all_ids)
    if base_rows is None or candidate_rows is None:
        gates["automatic_diagnostics"] = _gate(
            "inconclusive",
            "Complete 40-question automatic diagnostics are required for both models.",
        )
        return _result(base_key, candidate_key, gates)
    gates["automatic_diagnostics"] = _gate(
        "pass", "Both models have complete 40-question diagnostics."
    )

    def failure_episodes(rows: list[dict]) -> int:
        return sum(row["execution_error_episode"] or not row["submitted"] for row in rows)

    failures_base = failure_episodes(base_rows)
    failures_candidate = failure_episodes(candidate_rows)
    failure_delta = failures_candidate - failures_base
    gates["execution_or_no_submission"] = _gate(
        "pass" if failure_delta <= 2 else "fail",
        f"Candidate has {failure_delta:+d} execution-error or no-submission episodes; maximum +2.",
        base=failures_base,
        candidate=failures_candidate,
        delta=failure_delta,
        maximum=2,
    )

    base_duration = sum(row["duration_seconds"] for row in base_rows) / FULL_QUESTION_COUNT
    candidate_duration = (
        sum(row["duration_seconds"] for row in candidate_rows) / FULL_QUESTION_COUNT
    )
    if any(row["duration_seconds"] <= 0 for row in base_rows + candidate_rows):
        gates["duration"] = _gate(
            "inconclusive", "Every episode needs a positive measured duration; missing "
            "timings cannot be treated as zero-cost or instant attempts."
        )
    else:
        gates["duration"] = _gate(
            "pass" if candidate_duration <= 1.5 * base_duration else "fail",
            f"Mean duration is {candidate_duration:.4g}s versus {base_duration:.4g}s base; "
            "maximum 1.5× base.",
            base=base_duration,
            candidate=candidate_duration,
            maximum_ratio=1.5,
        )

    base_cost = _declared_cost(automatic, base_key, base_rows)
    candidate_cost = _declared_cost(automatic, candidate_key, candidate_rows)
    if base_cost is None or candidate_cost is None:
        gates["estimated_compute_cost"] = _gate(
            "inconclusive",
            "Serving hardware, runtime, hourly price, duration, or estimated cost "
            "is missing or inconsistent for at least one model.",
        )
    else:
        gates["estimated_compute_cost"] = _gate(
            "pass" if candidate_cost <= 1.5 * base_cost else "fail",
            f"Estimated compute cost is ${candidate_cost:.6g} versus ${base_cost:.6g} base; "
            "maximum 1.5× base.",
            base=base_cost,
            candidate=candidate_cost,
            maximum_ratio=1.5,
        )
    return _result(base_key, candidate_key, gates)
