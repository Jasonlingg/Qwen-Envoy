import copy
import hashlib
import json

import pytest
from pydantic import ValidationError

from src.product.ml_experiment_io import decode, encode
from src.product.ml_experiment_models import ExperimentBrief
from src.product.ml_experiment_types import KnownText, LocalArtifact, PaperSpan, ResourceBudget
from tests.ml_experiment_fixtures import complete_brief


def test_question_alone_is_a_valid_draft_with_explicit_unknowns():
    brief = ExperimentBrief(question="Should I try a smaller model?")
    assert brief.current_setup.hardware.value is None
    assert brief.current_setup.hardware.unknown_reason
    assert "hypothesis" in brief.missing_fields()
    assert "current_setup.model" in brief.missing_fields()
    assert brief.candidate_outcomes_already_seen == "unknown"


def test_complete_brief_can_freeze_with_hardware_explicitly_unknown():
    brief = ExperimentBrief.model_validate(complete_brief())
    assert brief.missing_fields() == []
    assert brief.current_setup.hardware.unknown_reason
    assert brief.budget.max_cost_usd == 0


@pytest.mark.parametrize("value", [{}, {"value": "GPU", "unknown_reason": "No GPU"},
                                  {"value": " "}, {"unknown_reason": " "}])
def test_unknowns_cannot_lose_their_reason_or_conflict_with_a_value(value):
    with pytest.raises(ValidationError):
        KnownText.model_validate(value)


@pytest.mark.parametrize("key,value", [("surprise", "extra field"), ("question", "  "),
                                      ("question", 42), ("author_kind", "human_reviewed"),
                                      ("candidate_outcomes_already_seen", False)])
def test_brief_rejects_extra_fields_coercion_and_claimed_human_review(key, value):
    with pytest.raises(ValidationError):
        ExperimentBrief.model_validate({"question": "Try a reader?", key: value})


@pytest.mark.parametrize("field", ["metrics", "criteria", "conditions", "references"])
def test_duplicate_identities_are_rejected(field):
    data = complete_brief()
    data["references"] = [{"kind": "external_reference", "reference_id": "paper",
                           "url": "https://example.org/paper", "description": "Declared only"}]
    data[field].append(copy.deepcopy(data[field][0]))
    with pytest.raises(ValidationError, match="Duplicate"):
        ExperimentBrief.model_validate(data)


def test_dangling_metric_and_evidence_references_are_rejected():
    for key, value in [("criteria", [{"criterion_id": "speed", "metric_id": "missing",
                                    "operation": "absolute_lte", "threshold": 10.0}]),
                       ("observations", [{"text": "Observed failure", "reference_ids": ["bad"]}])]:
        with pytest.raises(ValidationError, match="unknown"):
            ExperimentBrief.model_validate({**complete_brief(), key: value})


@pytest.mark.parametrize("field,value", [("max_model_requests", True),
                                        ("max_gpu_seconds", float("inf")),
                                        ("max_elapsed_seconds", 0.0),
                                        ("max_cost_usd", -1.0), ("max_input_tokens", 10),
                                        ("cost_unknown_reason", "Unknown")])
def test_budget_rejects_invalid_or_inconsistent_limits(field, value):
    budget = {**complete_brief()["budget"], field: value}
    with pytest.raises(ValidationError):
        ResourceBudget.model_validate(budget)


def test_samples_and_thresholds_are_strict():
    data = complete_brief()
    data["evaluation"]["sample_ids"] = ["same", "same"]
    with pytest.raises(ValidationError, match="unique"):
        ExperimentBrief.model_validate(data)
    data = complete_brief()
    data["criteria"][0]["threshold"] = float("nan")
    with pytest.raises(ValidationError):
        ExperimentBrief.model_validate(data)


def test_sample_hash_binds_the_declared_ids_independent_of_order():
    data = complete_brief()
    samples = sorted(data["evaluation"]["sample_ids"])
    sha = hashlib.sha256(json.dumps(samples, separators=(",", ":")).encode()).hexdigest()
    data["evaluation"]["sample_set_sha256"] = sha
    data["evaluation"]["sample_ids"].reverse()
    assert ExperimentBrief.model_validate(data).evaluation.sample_set_sha256 == sha
    data["evaluation"]["sample_ids"][0] = "different"
    with pytest.raises(ValidationError, match="hash"):
        ExperimentBrief.model_validate(data)


def test_mismatched_constants_and_optional_only_criteria_block_freezing():
    data = complete_brief()
    data["conditions"].append({"name": "inputs", "role": "held_constant",
                               "baseline": {"value": "first"}, "candidate": {"value": "second"}})
    data["criteria"][0]["required"] = False
    missing = ExperimentBrief.model_validate(data).missing_fields()
    assert "conditions.inputs.held_constant_mismatch" in missing
    assert "criteria.at_least_one_required" in missing


@pytest.mark.parametrize("updates", [{"arxiv_id": "2402.01030"}, {"start": True},
                                      {"end": 1_201}, {"quote": "wrong length"}])
def test_paper_reference_is_versioned_and_bounded_but_not_source_verified(updates):
    reference = {"kind": "paper_span", "reference_id": "paper", "arxiv_id": "2402.01030v4",
                 "start": 0, "end": 10, **updates}
    with pytest.raises(ValidationError):
        PaperSpan.model_validate(reference)


@pytest.mark.parametrize("updates", [{"relative_path": "../secret.json"},
                                      {"relative_path": "/tmp/secret.json"},
                                      {"relative_path": "data\\secret.json"},
                                      {"relative_path": ".env"}, {"json_pointer": "value"},
                                      {"json_pointer": "/value~2"}])
def test_local_reference_rejects_unsafe_names_and_bad_pointers(updates):
    with pytest.raises(ValidationError):
        LocalArtifact.model_validate({"kind": "local_artifact", "reference_id": "run",
                                      "relative_path": "run.json", "sha256": "a" * 64, **updates})


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'\xff',
                                b'[' * 30 + b'0' + b']' * 30, b' ' * (256 * 1024 + 1)])
def test_disk_json_rejects_ambiguous_nonfinite_deep_and_oversized_values(raw):
    with pytest.raises(ValueError):
        decode(raw)


@pytest.mark.parametrize("value", [{"x": float("inf")}, {"x": "a" * (256 * 1024)},
                                  {1: "not a JSON key"}])
def test_in_memory_json_has_equivalent_bounds(value):
    with pytest.raises(ValueError):
        encode(value)
