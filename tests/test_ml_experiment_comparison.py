from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.product.ml_experiment_compare import assess_result
from src.product.ml_experiment_measurements import CostBreakdown
from src.product.ml_experiment_models import FrozenPlan
from src.product.ml_experiment_results import ExperimentResult
from src.product.ml_experiment_store import ExperimentStore
from tests.ml_experiment_result_fixtures import paired_result, planned_brief


def compare(tmp_path, brief=None, result=None):
    brief = brief or planned_brief()
    store = ExperimentStore(tmp_path / "records")
    saved = store.save_brief(brief)
    frozen = store.freeze_plan(saved["experiment_id"], 1)
    return assess_result(FrozenPlan.model_validate(frozen["frozen_plan"]),
                         ExperimentResult.model_validate(result or paired_result(brief)))


@pytest.mark.parametrize("operation,threshold,expected,status", [
    ("absolute_lte", 75.0, 75.0, "met"), ("absolute_gte", 80.0, 75.0, "not_met"),
    ("delta_lte", -25.0, -25.0, "met"), ("delta_gte", 0.0, -25.0, "not_met"),
    ("relative_improvement_gte", 20.0, 25.0, "met"),
])
def test_frozen_operations(tmp_path, operation, threshold, expected, status):
    brief = planned_brief()
    brief["criteria"][0].update(operation=operation, threshold=threshold)
    checked = compare(tmp_path, brief)
    assert checked.criteria[0].observed_value == expected
    assert checked.criteria[0].status == status
    assert checked.overall == ("criteria_met" if status == "met" else "criteria_not_met")
    assert checked.effective_timing == "retrospective"
    assert any("Synthetic" in warning for warning in checked.warnings)
    assert checked.measurement_provenance["baseline.latency"] == "caller_reported"


@pytest.mark.parametrize("section,key,value,reason", [
    ("measurement", "unit", "seconds", "units"),
    ("measurement", "aggregation", "p95", "aggregation"),
    ("measurement", "population", "Successful questions", "population"),
    ("measurement", "coverage", "incomplete", "coverage"),
    ("measurement", "sample_count", 1, "coverage"),
    ("measurement", "timing_scope", "cold", "scope"),
    ("measurement", "value", -1.0, "negative"),
    ("evaluation", "sample_ids", ["synthetic-3", "synthetic-4"], "samples"),
    ("evaluation", "purpose", "confirmation", "purpose"),
    ("evaluation", "dataset_revision", {"unknown_reason": "Missing revision"}, "revisions"),
    ("identity", "hardware", {"value": "other GPU"}, "identity"),
    ("identity", "workload", {"unknown_reason": "Missing workload"}, "identity"),
    ("conditions", "reader", {"value": "other-reader"}, "condition"),
    ("arm", "missing_case_ids", ["synthetic-2"], "missing"),
    ("arm", "failed_case_ids", ["synthetic-1"], "inclusion"),
    ("arm", "measurements", [], "missing"),
])
def test_incompatible_or_incomplete_measurements_are_unknown(tmp_path, section, key, value, reason):
    result = paired_result()
    arm = result["candidate"]
    target = arm["measurements"][0] if section == "measurement" else (
        arm if section == "arm" else arm[section])
    target[key] = value
    checked = compare(tmp_path, result=result)
    assert checked.overall == "inconclusive"
    assert checked.criteria[0].observed_value is None
    assert reason in checked.criteria[0].reason


def test_unknown_measurement_and_legacy_metric_and_deviations(tmp_path):
    result = paired_result()
    result["candidate"]["measurements"][0].update(value=None, unknown_reason="No timestamp")
    assert compare(tmp_path, result=result).overall == "inconclusive"
    brief = planned_brief()
    del brief["metrics"][0]["measurement_kind"]
    assert "successor" in compare(tmp_path, brief).criteria[0].reason
    result = paired_result()
    result["deviations"] = ["Changed data loader"]
    assert "deviations" in compare(tmp_path, result=result).criteria[0].reason


def test_zero_baseline_and_overflow_do_not_produce_success(tmp_path):
    result = paired_result()
    result["baseline"]["measurements"][0]["value"] = 0.0
    assert "zero baseline" in compare(tmp_path, result=result).criteria[0].reason
    result["baseline"]["measurements"][0]["value"] = 1e-300
    result["candidate"]["measurements"][0]["value"] = 1e300
    assert "finite" in compare(tmp_path, result=result).criteria[0].reason


def test_percentage_points_higher_direction_and_optional_gate(tmp_path):
    brief = planned_brief()
    brief["metrics"][0].update(measurement_kind="other", unit="percent", direction="higher")
    brief["criteria"][0].update(operation="delta_gte", threshold=-25.0)
    brief["criteria"].append({**brief["criteria"][0], "criterion_id": "optional",
                              "threshold": 0.0, "required": False})
    result = paired_result(brief)
    for arm in (result["baseline"], result["candidate"]):
        arm["measurements"][0]["unit"] = "percent"
    checked = compare(tmp_path, brief, result)
    assert checked.criteria[0].unit == "percentage_points"
    assert checked.criteria[1].status == "not_met" and checked.overall == "criteria_met"
    brief["criteria"] = [dict(brief["criteria"][0], operation="relative_improvement_gte")]
    assert compare(tmp_path, brief, result).criteria[0].observed_value == -25.0


def test_failed_cases_require_explicit_inclusion_and_end_to_end_scope(tmp_path):
    brief = planned_brief()
    brief["metrics"][0]["failure_policy"] = "include"
    result = paired_result(brief)
    result["candidate"]["failed_case_ids"] = ["synthetic-1"]
    assert compare(tmp_path, brief, result).overall == "criteria_met"
    brief["metrics"][0]["timing_scope"] = "end_to_end"
    result = paired_result(brief)
    for arm in (result["baseline"], result["candidate"]):
        arm["measurements"][0]["timing_scope"] = "end_to_end"
    assert "identity" in compare(tmp_path, brief, result).criteria[0].reason


def test_cost_unknown_is_not_free_and_total_is_checked(tmp_path):
    brief = planned_brief()
    brief["metrics"][0].update(measurement_kind="cost", unit="USD", timing_scope=None)
    result = paired_result(brief)
    zero = {name: {"value": 0.0} for name in CostBreakdown.model_fields if name != "accounting"}
    for arm, cost in ((result["baseline"], 100.0), (result["candidate"], 75.0)):
        arm["measurements"][0].update(unit="USD", cost_breakdown={**deepcopy(zero),
            "host_inference": {"value": cost}, "accounting": "Synthetic session total"})
    assert compare(tmp_path, brief, result).overall == "criteria_met"
    result["candidate"]["measurements"][0]["cost_breakdown"]["gpu_idle"] = {
        "unknown_reason": "No idle billing record"}
    assert "unknown" in compare(tmp_path, brief, result).criteria[0].reason
    result["candidate"]["measurements"][0]["cost_breakdown"]["gpu_idle"] = {"value": 1.0}
    assert "disagrees" in compare(tmp_path, brief, result).criteria[0].reason


def test_historical_timing_cannot_be_promoted(tmp_path):
    brief = planned_brief()
    result = paired_result(brief)
    result.update(timing="prospective", started_at="2030-01-01T00:00:00+00:00")
    assert compare(tmp_path, brief, result).effective_timing == "retrospective"
    brief["candidate_outcomes_already_seen"] = "no"
    assert compare(tmp_path, brief, result).effective_timing == "prospective_declared"
    result["started_at"] = "2000-01-01T00:00:00+00:00"
    assert compare(tmp_path, brief, result).effective_timing == "retrospective"


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(timing="prospective"),
    lambda r: r.update(started_at="2026-10-01T00:00:00"),
    lambda r: r.update(started_at="2026-10-02T00:00:00Z", completed_at="2026-10-01T00:00:00Z"),
    lambda r: r["candidate"]["measurements"][0].update(value=True),
    lambda r: r["candidate"]["measurements"][0].update(value=float("inf")),
    lambda r: r["candidate"]["measurements"][0].update(value=None),
    lambda r: r["candidate"]["measurements"][0].update(numerator=3, denominator=2),
    lambda r: r["candidate"]["measurements"][0].update(sample_count=1, denominator=2, numerator=1),
    lambda r: r["candidate"]["measurements"].append(deepcopy(r["candidate"]["measurements"][0])),
    lambda r: r["candidate"].update(failed_case_ids=["unknown"]),
    lambda r: r["candidate"].update(failed_case_ids=["synthetic-1"],
                                    missing_case_ids=["synthetic-1"]),
])
def test_invalid_results_rejected(mutation):
    result = paired_result()
    mutation(result)
    with pytest.raises(ValidationError):
        ExperimentResult.model_validate(result)
