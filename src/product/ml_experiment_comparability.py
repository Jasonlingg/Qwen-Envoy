"""Eligibility checks for comparing paired measurements against a frozen plan."""

from __future__ import annotations

import hashlib
import json
import math


def _sample_hash(evaluation):
    if evaluation.sample_ids:
        raw = json.dumps(sorted(evaluation.sample_ids), ensure_ascii=False,
                         separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()
    return evaluation.sample_set_sha256


def _known(values, key):
    item = values.get(key)
    return item.value if item is not None else None


def comparison_problem(brief, result, metric, provenance) -> str | None:
    if result.deviations:
        return "Declared deviations from the frozen plan need review"
    if metric.measurement_kind == "unspecified":
        return "Frozen metric does not declare its measurement kind; create a successor plan"
    required = ["model_revision", "workload"]
    if metric.measurement_kind in {"latency", "cost"}:
        required += ["hardware", "software_revision"]
    if metric.measurement_kind == "quality":
        required += ["prompt_sha256", "decoding"]
    if metric.measurement_kind == "latency" and metric.timing_scope == "end_to_end":
        required += ["session_allocation"]
    arms = (result.baseline, result.candidate)
    dataset_revisions = [arm.evaluation.dataset_revision.value for arm in arms]
    if None in dataset_revisions or len(set(dataset_revisions)) != 1:
        return "Dataset revisions are unknown or differ between arms"
    planned_revision = brief.evaluation.dataset_revision.value
    if planned_revision is not None and dataset_revisions[0] != planned_revision:
        return "Dataset revision differs from the frozen plan"
    for label, arm in zip(("baseline", "candidate"), arms, strict=True):
        evaluation = arm.evaluation
        expected = brief.evaluation
        if (evaluation.population != expected.population
                or evaluation.expected_cases != expected.expected_cases
                or evaluation.purpose != expected.purpose
                or _sample_hash(evaluation) != _sample_hash(expected)):
            return f"{label}: evaluation samples, purpose, or population differ from the plan"
        if arm.missing_case_ids:
            return f"{label}: some scheduled cases are missing"
        if any(_known(arm.identity, key) is None for key in required):
            return f"{label}: relevant model/workload/runtime identity is unknown"
        for key, value in getattr(brief, label).identity.items():
            if value.value is not None and _known(arm.identity, key) != value.value:
                return f"{label}: actual identity differs from the declared arm"
        for condition in brief.conditions:
            actual = _known(arm.conditions, condition.name)
            planned = getattr(condition, label).value
            if actual is None or planned is None or actual != planned:
                return f"{label}: a planned condition is unknown or mismatched"
            if (condition.name in arm.identity
                    and _known(arm.identity, condition.name) != actual):
                return f"{label}: identity and condition declarations conflict"
        measurement = next((m for m in arm.measurements if m.metric_id == metric.metric_id), None)
        if measurement is None or measurement.value is None:
            return f"{label}: measurement is missing or unknown"
        if (measurement.unit != metric.unit or measurement.aggregation != metric.aggregation
                or measurement.population != metric.population):
            return f"{label}: metric units, aggregation, or population differ from the plan"
        if (measurement.coverage != "complete"
                or measurement.sample_count != expected.expected_cases):
            return f"{label}: metric coverage is incomplete or unknown"
        if (arm.failed_case_ids and metric.failure_policy != "include"):
            return f"{label}: failed cases have no planned inclusion policy"
        if metric.requires_semantic_review:
            if (measurement.review.kind != "human_declared"
                    or provenance.get(f"{label}.{metric.metric_id}")
                    != "artifact_measurement_verified"):
                return f"{label}: semantic human-review declaration is absent or not artifact-bound"
        if metric.measurement_kind == "latency":
            if metric.timing_scope is None or measurement.timing_scope != metric.timing_scope:
                return f"{label}: latency scope is unknown or differs from the plan"
            if measurement.value < 0:
                return f"{label}: latency is negative"
        if metric.measurement_kind == "cost":
            breakdown = measurement.cost_breakdown
            if breakdown is None or breakdown.known_total() is None:
                return f"{label}: all-in cost components are missing or unknown"
            if measurement.unit != "USD" or not math.isclose(
                    measurement.value, breakdown.known_total(), rel_tol=1e-9, abs_tol=1e-12):
                return f"{label}: USD total disagrees with the component breakdown"
    for key in required:
        if _known(arms[0].identity, key) != _known(arms[1].identity, key):
            planned = [_known(getattr(brief, label).identity, key)
                       for label in ("baseline", "candidate")]
            if None in planned or planned[0] == planned[1]:
                return "Relevant arm identities differ without a predeclared difference"
    for condition in brief.conditions:
        if condition.role == "held_constant" and (
                _known(arms[0].conditions, condition.name)
                != _known(arms[1].conditions, condition.name)):
            return "A held-constant condition changed between arms"
    return None
