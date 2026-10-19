"""Question-first briefs and frozen-plan envelopes, without model or execution calls."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from src.product.ml_experiment_receipts import EvidenceReceipt, check_receipts
from src.product.ml_experiment_types import (
    Condition,
    Criterion,
    CurrentSetup,
    Digest,
    Evaluation,
    ExperimentArm,
    ExperimentID,
    Key,
    Metric,
    Observation,
    RecordModel,
    Reference,
    ResourceBudget,
    Text,
)

SOURCE_WARNING = "References are declarations; source and semantic verification are separate."
FREEZE_WARNING = ("Freezing records a plan; it proves neither execution permission "
                  "nor preregistration.")
PositiveInt = Annotated[int, Field(ge=1)]


class ExperimentBrief(RecordModel):
    schema_version: Literal["ml-experiment-brief-v1"] = "ml-experiment-brief-v1"
    question: Text
    goal: Literal["model_comparison", "fine_tuning", "latency"] | None = None
    current_setup: CurrentSetup = Field(default_factory=CurrentSetup)
    constraints: list[Text] = Field(default_factory=list, max_length=20)
    observations: list[Observation] = Field(default_factory=list, max_length=20)
    alternatives: list[Text] = Field(default_factory=list, max_length=10)
    proposed_change: Text | None = None
    hypothesis: Text | None = None
    expected_signal: Text | None = None
    applicability: Text | None = None
    differences: list[Text] = Field(default_factory=list, max_length=20)
    baseline: ExperimentArm | None = None
    candidate: ExperimentArm | None = None
    evaluation: Evaluation | None = None
    conditions: list[Condition] = Field(default_factory=list, max_length=30)
    metrics: list[Metric] = Field(default_factory=list, max_length=20)
    criteria: list[Criterion] = Field(default_factory=list, max_length=20)
    budget: ResourceBudget | None = None
    procedure: list[Text] = Field(default_factory=list, max_length=30)
    references: list[Reference] = Field(default_factory=list, max_length=20)
    unknowns: list[Text] = Field(default_factory=list, max_length=30)
    limitations: list[Text] = Field(default_factory=list, max_length=20)
    candidate_outcomes_already_seen: Literal["yes", "no", "unknown"] = "unknown"
    author_kind: Literal["user_supplied", "agent_authored"] = "agent_authored"
    supersedes_experiment_id: ExperimentID | None = None

    @model_validator(mode="after")
    def consistent_references(self):
        for items, attribute in ((self.references, "reference_id"), (self.metrics, "metric_id"),
                                 (self.criteria, "criterion_id"), (self.conditions, "name")):
            ids = [getattr(item, attribute) for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {attribute}")
        references = {item.reference_id for item in self.references}
        if any(set(item.reference_ids) - references for item in self.observations):
            raise ValueError("An observation names an unknown reference")
        metrics = {item.metric_id for item in self.metrics}
        if any(item.metric_id not in metrics for item in self.criteria):
            raise ValueError("A criterion names an unknown metric")
        return self

    def missing_fields(self) -> list[str]:
        """Drafts are useful from one question; freezing needs a repeatable test."""
        missing = [name for name in (
            "goal", "proposed_change", "hypothesis", "expected_signal", "applicability",
            "baseline", "candidate", "evaluation", "budget", "procedure", "metrics", "criteria",
        ) if not getattr(self, name)]
        for name in ("task", "model", "workload"):
            if getattr(self.current_setup, name).value is None:
                missing.append(f"current_setup.{name}")
        if len([c for c in self.conditions if c.role == "proposed_change"]) != 1:
            missing.append("conditions.one_proposed_change")
        for condition in self.conditions:
            if (condition.role == "held_constant" and condition.baseline.value is not None
                    and condition.candidate.value is not None
                    and condition.baseline.value != condition.candidate.value):
                missing.append(f"conditions.{condition.name}.held_constant_mismatch")
        if self.criteria and not any(c.required for c in self.criteria):
            missing.append("criteria.at_least_one_required")
        return missing


class Timestamped(RecordModel):
    @field_validator("recorded_at", "frozen_at", check_fields=False)
    @classmethod
    def utc_timestamp(cls, value: str) -> str:
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("Invalid record timestamp") from exc
        if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError("Record timestamp must use UTC")
        return value


class BriefRevision(Timestamped):
    schema_version: Literal["ml-experiment-revision-v1"] = "ml-experiment-revision-v1"
    experiment_id: ExperimentID
    revision: PositiveInt
    previous_sha256: Digest | None
    recorded_at: str
    brief: ExperimentBrief
    evidence_receipts: list[EvidenceReceipt] = Field(default_factory=list, max_length=20)
    reference_warning: Literal[SOURCE_WARNING] = SOURCE_WARNING

    @model_validator(mode="after")
    def bound_evidence(self):
        if self.evidence_receipts:
            check_receipts(self.brief.references, self.evidence_receipts)
        return self


class FrozenPlan(Timestamped):
    schema_version: Literal["ml-experiment-plan-v1"] = "ml-experiment-plan-v1"
    experiment_id: ExperimentID
    revision: PositiveInt
    revision_sha256: Digest
    frozen_at: str
    brief: ExperimentBrief
    evidence_receipts: list[EvidenceReceipt] = Field(default_factory=list, max_length=20)
    warning: Literal[FREEZE_WARNING] = FREEZE_WARNING

    @model_validator(mode="after")
    def complete_brief(self):
        if self.brief.missing_fields():
            raise ValueError("A frozen plan contains an incomplete brief")
        if self.evidence_receipts:
            check_receipts(self.brief.references, self.evidence_receipts)
        return self


class RevisionPointer(RecordModel):
    revision: PositiveInt
    sha256: Digest


class ResultPointer(RecordModel):
    run_id: Key
    sha256: Digest


class ExperimentIndex(RecordModel):
    schema_version: Literal["ml-experiment-index-v1"] = "ml-experiment-index-v1"
    experiment_id: ExperimentID
    revisions: list[RevisionPointer] = Field(min_length=1, max_length=100)
    frozen_plan_sha256: Digest | None = None
    results: list[ResultPointer] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def ordered_revisions(self):
        if [p.revision for p in self.revisions] != list(range(1, len(self.revisions) + 1)):
            raise ValueError("Experiment revisions must be consecutive")
        if len({p.sha256 for p in self.revisions}) != len(self.revisions):
            raise ValueError("Duplicate revision digest")
        if len({p.run_id for p in self.results}) != len(self.results):
            raise ValueError("Duplicate result run ID")
        if self.results and self.frozen_plan_sha256 is None:
            raise ValueError("Recorded results require a frozen plan")
        return self


class IndexEnvelope(RecordModel):
    schema_version: Literal["ml-experiment-index-envelope-v1"] = "ml-experiment-index-envelope-v1"
    sha256: Digest
    index: ExperimentIndex
