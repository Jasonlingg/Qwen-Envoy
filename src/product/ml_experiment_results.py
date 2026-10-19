"""Immutable experiment outcomes, their declared timing, and mechanical assessments."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from src.product.ml_experiment_measurements import MeasuredArm
from src.product.ml_experiment_models import Timestamped
from src.product.ml_experiment_receipts import EvidenceReceipt, check_receipts
from src.product.ml_experiment_types import (
    Digest,
    ExperimentID,
    Key,
    Name,
    Number,
    RecordModel,
    Reference,
    Text,
)


class ExperimentResult(RecordModel):
    schema_version: Literal["ml-experiment-result-v1"] = "ml-experiment-result-v1"
    run_id: Key
    timing: Literal["prospective", "retrospective", "unknown"] = "unknown"
    started_at: Name | None = None
    completed_at: Name | None = None
    data_kind: Literal["recorded", "synthetic"] = "recorded"
    baseline: MeasuredArm
    candidate: MeasuredArm
    references: list[Reference] = Field(default_factory=list, max_length=20)
    deviations: list[Text] = Field(default_factory=list, max_length=20)
    interpretation: Text | None = None
    limitations: list[Text] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def valid_timing_and_refs(self):
        dates = []
        for value in (self.started_at, self.completed_at):
            if value is not None:
                try:
                    parsed = datetime.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError("Invalid result timestamp") from exc
                if parsed.utcoffset() is None:
                    raise ValueError("Result timestamps need timezones")
                dates.append(parsed)
        if len(dates) == 2 and dates[0] > dates[1]:
            raise ValueError("Result completion predates its start")
        if self.timing == "prospective" and self.started_at is None:
            raise ValueError("Prospective timing needs a declared start timestamp")
        ids = [r.reference_id for r in self.references]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate result reference ID")
        if any(m.reference_id is not None and m.reference_id not in ids
               for arm in (self.baseline, self.candidate) for m in arm.measurements):
            raise ValueError("Measurement names an unknown result reference")
        return self


class CriterionAssessment(RecordModel):
    criterion_id: Key
    metric_id: Key
    required: bool
    status: Literal["met", "not_met", "unknown"]
    observed_value: Number | None
    unit: Name
    reason: Text


class ResultAssessment(RecordModel):
    schema_version: Literal["ml-experiment-assessment-v1"] = "ml-experiment-assessment-v1"
    overall: Literal["criteria_met", "criteria_not_met", "inconclusive"]
    effective_timing: Literal["prospective_declared", "retrospective", "unknown"]
    criteria: list[CriterionAssessment] = Field(min_length=1, max_length=20)
    measurement_provenance: dict[str, Literal[
        "caller_reported", "artifact_value_only", "artifact_measurement_verified"
    ]] = Field(max_length=40)
    warnings: list[Text] = Field(default_factory=list, max_length=20)


class ResultRecord(Timestamped):
    schema_version: Literal["ml-experiment-outcome-v1"] = "ml-experiment-outcome-v1"
    experiment_id: ExperimentID
    plan_sha256: Digest
    recorded_at: str
    result: ExperimentResult
    evidence_receipts: list[EvidenceReceipt] = Field(default_factory=list, max_length=20)
    assessment: ResultAssessment

    @model_validator(mode="after")
    def bound_evidence(self):
        check_receipts(self.result.references, self.evidence_receipts)
        return self
