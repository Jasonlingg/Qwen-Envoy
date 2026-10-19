"""Mechanical criteria checks on supplied measurements; no scientific or adoption verdict."""

from __future__ import annotations

import math
from datetime import datetime

from src.product.ml_experiment_comparability import comparison_problem
from src.product.ml_experiment_result_evidence import measurement_provenance
from src.product.ml_experiment_results import CriterionAssessment, ResultAssessment


def _timing(plan, result):
    if (result.timing == "retrospective" or plan.brief.candidate_outcomes_already_seen == "yes"
            or result.started_at is not None
            and datetime.fromisoformat(result.started_at) < datetime.fromisoformat(plan.frozen_at)):
        return "retrospective"
    if (result.timing == "prospective" and result.started_at is not None
            and plan.brief.candidate_outcomes_already_seen == "no"):
        return "prospective_declared"
    return "unknown"


def assess_result(plan, result, receipts=()) -> ResultAssessment:
    provenance = measurement_provenance(result, receipts)
    metrics = {m.metric_id: m for m in plan.brief.metrics}
    unknown_ids = {m.metric_id for arm in (result.baseline, result.candidate)
                   for m in arm.measurements} - set(metrics)
    if unknown_ids:
        raise ValueError("Result includes metrics not defined in the frozen plan")
    rows = []
    for criterion in plan.brief.criteria:
        metric = metrics[criterion.metric_id]
        problem = comparison_problem(plan.brief, result, metric, provenance)
        value = None
        unit = metric.unit
        if not problem:
            baseline, candidate = [next(m.value for m in arm.measurements
                                       if m.metric_id == metric.metric_id)
                                   for arm in (result.baseline, result.candidate)]
            if criterion.operation.startswith("absolute"):
                value = candidate
            elif criterion.operation.startswith("delta"):
                value = candidate - baseline
                if metric.unit == "percent":
                    unit = "percentage_points"
            elif baseline == 0:
                problem = "Relative improvement is undefined for a zero baseline"
            else:
                change = (candidate - baseline if metric.direction == "higher"
                          else baseline - candidate)
                value = change / abs(baseline) * 100
                unit = "percent_improvement"
            if value is not None and not math.isfinite(value):
                value, problem = None, "Arithmetic exceeded finite-number limits"
        status = "unknown"
        if problem:
            value = None
        else:
            met = (value <= criterion.threshold if criterion.operation.endswith("lte")
                   else value >= criterion.threshold)
            status = "met" if met else "not_met"
        rows.append(CriterionAssessment(criterion_id=criterion.criterion_id,
            metric_id=metric.metric_id, required=criterion.required, status=status,
            observed_value=value, unit=unit,
            reason=problem or "Compared against the frozen threshold"))
    required = {row.status for row in rows if row.required}
    overall = ("criteria_not_met" if "not_met" in required else
               "inconclusive" if "unknown" in required or not required else "criteria_met")
    warnings = ["Criteria compare supplied measurements; they prove no causality or adoption.",
                "Runtime identity, coverage, and controlled conditions remain caller declarations.",
                "Reviewer identity and timestamps are declarations, not authenticated attestations."
                ]
    if result.data_kind == "synthetic":
        warnings.append("Synthetic data: this is not a measured project outcome.")
    if any(value == "caller_reported" for value in provenance.values()):
        warnings.append("Some measurements are caller-reported without source verification.")
    if any(value == "artifact_value_only" for value in provenance.values()):
        warnings.append("Some artifacts verify only values; other metadata remains declared.")
    return ResultAssessment(overall=overall, effective_timing=_timing(plan, result),
                            criteria=rows, measurement_provenance=provenance, warnings=warnings)
