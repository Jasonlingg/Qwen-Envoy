"""Recorded measurements and coverage; declarations are not scientific validation."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from src.product.ml_experiment_types import (
    Count,
    Evaluation,
    Key,
    KnownText,
    Name,
    Number,
    RecordModel,
    Text,
)


class KnownNumber(RecordModel):
    value: Number | None = None
    unknown_reason: Text | None = None

    @model_validator(mode="after")
    def explicit_unknown(self):
        if (self.value is None) != (self.unknown_reason is not None):
            raise ValueError("Supply a measurement or an unknown_reason, exclusively")
        return self


class ReviewDeclaration(RecordModel):
    kind: Literal["not_reviewed", "model_assisted", "human_declared"] = "not_reviewed"
    reviewer: Name | None = None
    reviewed_at: Name | None = None

    @model_validator(mode="after")
    def declared_human(self):
        if self.kind == "human_declared":
            if not self.reviewer or not self.reviewed_at:
                raise ValueError("Human review needs its declared reviewer and timestamp")
            try:
                timestamp = datetime.fromisoformat(self.reviewed_at)
            except ValueError as exc:
                raise ValueError("Invalid review timestamp") from exc
            if timestamp.utcoffset() is None:
                raise ValueError("Review timestamp needs a timezone")
        return self


class CostBreakdown(RecordModel):
    host_inference: KnownNumber
    worker_inference: KnownNumber
    gpu_loading: KnownNumber
    gpu_active: KnownNumber
    gpu_idle: KnownNumber
    cleanup: KnownNumber
    session_storage: KnownNumber
    accounting: Text

    @model_validator(mode="after")
    def nonnegative_costs(self):
        for name in type(self).model_fields:
            if name != "accounting":
                value = getattr(self, name).value
                if value is not None and value < 0:
                    raise ValueError("Cost components must be nonnegative")
        return self

    def known_total(self) -> float | None:
        values = [getattr(self, name).value for name in type(self).model_fields
                  if name != "accounting"]
        if any(value is None for value in values):
            return None
        total = sum(values)
        return total if math.isfinite(total) else None


class Measurement(KnownNumber):
    metric_id: Key
    unit: Name
    aggregation: Name
    population: Name
    sample_count: Count | None = None
    coverage: Literal["complete", "incomplete", "unknown"] = "unknown"
    numerator: Count | None = None
    denominator: Annotated[int, Field(ge=1)] | None = None
    reference_id: Key | None = None
    review: ReviewDeclaration = Field(default_factory=ReviewDeclaration)
    timing_scope: Literal["warm", "cold", "end_to_end"] | None = None
    cost_breakdown: CostBreakdown | None = None

    @model_validator(mode="after")
    def consistent_counts(self):
        if self.value is not None and self.unit in {"fraction", "percent"}:
            if not 0 <= self.value <= (100 if self.unit == "percent" else 1):
                raise ValueError("Fraction or percent measurement is outside its valid range")
        if (self.numerator is None) != (self.denominator is None):
            raise ValueError("Supply numerator and denominator together")
        if self.denominator is not None:
            if self.numerator > self.denominator:
                raise ValueError("Numerator exceeds denominator")
            if self.sample_count != self.denominator:
                raise ValueError("Denominator must match the declared sample count")
            if self.value is not None and self.unit in {"fraction", "percent"}:
                ratio = self.numerator / self.denominator * (100 if self.unit == "percent" else 1)
                if not math.isclose(self.value, ratio, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError("Measurement disagrees with its counts")
        return self


class MeasuredArm(RecordModel):
    identity: dict[Key, KnownText] = Field(default_factory=dict, max_length=30)
    conditions: dict[Key, KnownText] = Field(default_factory=dict, max_length=30)
    evaluation: Evaluation
    measurements: list[Measurement] = Field(default_factory=list, max_length=20)
    failed_case_ids: list[Name] = Field(default_factory=list, max_length=1_000)
    missing_case_ids: list[Name] = Field(default_factory=list, max_length=1_000)

    @model_validator(mode="after")
    def unique_cases_and_metrics(self):
        metrics = [m.metric_id for m in self.measurements]
        if len(set(metrics)) != len(metrics):
            raise ValueError("Duplicate measured metric ID")
        for ids in (self.failed_case_ids, self.missing_case_ids):
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate failed/missing case ID")
            if self.evaluation.sample_ids and not set(ids) <= set(self.evaluation.sample_ids):
                raise ValueError("Failed/missing cases must belong to the declared sample")
        if set(self.failed_case_ids) & set(self.missing_case_ids):
            raise ValueError("A case cannot be both failed and missing")
        if len(self.failed_case_ids) + len(self.missing_case_ids) > self.evaluation.expected_cases:
            raise ValueError("Failed/missing cases exceed the sample size")
        if any(m.sample_count is not None and m.sample_count > self.evaluation.expected_cases
               for m in self.measurements):
            raise ValueError("Measured sample count exceeds the declared sample")
        return self
