"""Strict, bounded inputs shared by local ML experiment records."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.product.arxiv_ids import validate_arxiv_id

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000)]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Key = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ExperimentID = Annotated[str, StringConstraints(pattern=r"^experiment_[0-9a-f]{32}$")]
Count = Annotated[int, Field(ge=0)]
Number = Annotated[float, Field(allow_inf_nan=False)]


class RecordModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, validate_default=True,
                              allow_inf_nan=False)


class KnownText(RecordModel):
    value: Text | None = None
    unknown_reason: Text | None = None

    @model_validator(mode="after")
    def explicit_unknown(self):
        if (self.value is None) != (self.unknown_reason is not None):
            raise ValueError("Supply a value or an unknown_reason, exclusively")
        return self


def unknown_text() -> KnownText:
    return KnownText(unknown_reason="Not provided yet")


class CurrentSetup(RecordModel):
    task: KnownText = Field(default_factory=unknown_text)
    model: KnownText = Field(default_factory=unknown_text)
    checkpoint: KnownText = Field(default_factory=unknown_text)
    revision: KnownText = Field(default_factory=unknown_text)
    hardware: KnownText = Field(default_factory=unknown_text)
    workload: KnownText = Field(default_factory=unknown_text)
    software: dict[Key, KnownText] = Field(default_factory=dict, max_length=30)


class PaperSpan(RecordModel):
    kind: Literal["paper_span"]
    reference_id: Key
    arxiv_id: Name
    start: Count
    end: Count
    quote: Annotated[str, Field(min_length=1, max_length=1_200)] | None = None
    document_sha256: Digest | None = None

    @model_validator(mode="after")
    def version_and_range(self):
        if "v" not in validate_arxiv_id(self.arxiv_id):
            raise ValueError("Paper evidence needs a versioned arXiv ID")
        if not 0 < self.end - self.start <= 1_200:
            raise ValueError("Paper span must cover 1..1200 characters")
        if self.quote is not None and len(self.quote) != self.end - self.start:
            raise ValueError("Quote length must match offsets; source verification is separate")
        return self


class LocalArtifact(RecordModel):
    kind: Literal["local_artifact"]
    reference_id: Key
    relative_path: Name
    sha256: Digest
    json_pointer: Annotated[str, Field(max_length=1_000)] = ""

    @model_validator(mode="after")
    def safe_reference(self):
        path = self.relative_path
        if (not path.endswith(".json") or path.startswith("/") or "\\" in path
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or any(ord(char) < 32 for char in path)):
            raise ValueError("Artifact must name a relative JSON file inside the export root")
        pointer = self.json_pointer
        if pointer and not pointer.startswith("/"):
            raise ValueError("JSON Pointer must be empty or start with /")
        for part in pointer.split("~")[1:]:
            if not part.startswith(("0", "1")):
                raise ValueError("Invalid JSON Pointer escape")
        return self


class ExternalReference(RecordModel):
    kind: Literal["external_reference"]
    reference_id: Key
    url: Annotated[str, StringConstraints(pattern=r"^https://", max_length=2_000)]
    description: Text


Reference = Annotated[PaperSpan | LocalArtifact | ExternalReference, Field(discriminator="kind")]


class Observation(RecordModel):
    text: Text
    reference_ids: list[Key] = Field(default_factory=list, max_length=20)
    basis: Literal["user_reported", "artifact_reported", "agent_interpretation"] = "user_reported"


class ExperimentArm(RecordModel):
    description: Text
    identity: dict[Key, KnownText] = Field(default_factory=dict, max_length=30)


class Condition(RecordModel):
    name: Key
    role: Literal["held_constant", "proposed_change", "declared_difference"]
    baseline: KnownText
    candidate: KnownText


class Evaluation(RecordModel):
    population: Text
    purpose: Literal["development", "confirmation", "historical"]
    expected_cases: Annotated[int, Field(ge=1)]
    sample_ids: list[Name] = Field(default_factory=list, max_length=1_000)
    sample_set_sha256: Digest | None = None
    dataset_revision: KnownText = Field(default_factory=unknown_text)

    @model_validator(mode="after")
    def declared_samples(self):
        if not self.sample_ids and self.sample_set_sha256 is None:
            raise ValueError("Supply sample IDs or an explicit sample-set hash")
        if self.sample_ids and (len(self.sample_ids) != self.expected_cases
                                or len(set(self.sample_ids)) != len(self.sample_ids)):
            raise ValueError("Sample IDs must be unique and match expected_cases")
        if self.sample_ids and self.sample_set_sha256 is not None:
            raw = json.dumps(sorted(self.sample_ids), ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != self.sample_set_sha256:
                raise ValueError("Sample-set hash does not match the sorted sample IDs")
        return self


class Metric(RecordModel):
    metric_id: Key
    unit: Name
    aggregation: Name
    population: Name
    direction: Literal["higher", "lower"]
    measurement_kind: Literal["quality", "latency", "cost", "other", "unspecified"] = "unspecified"
    timing_scope: Literal["warm", "cold", "end_to_end"] | None = None
    failure_policy: Literal["include", "unspecified"] = "unspecified"
    requires_semantic_review: bool = False


class Criterion(RecordModel):
    criterion_id: Key
    metric_id: Key
    operation: Literal["absolute_lte", "absolute_gte", "delta_lte", "delta_gte",
                       "relative_improvement_gte"]
    threshold: Number
    required: bool = True


class ResourceBudget(RecordModel):
    max_model_requests: Count
    max_input_tokens: Count
    max_output_tokens: Count
    max_gpu_seconds: Annotated[float, Field(ge=0)]
    max_elapsed_seconds: Annotated[float, Field(gt=0)]
    max_cost_usd: Annotated[float, Field(ge=0)] | None = None
    cost_unknown_reason: Text | None = None
    stop_conditions: list[Text] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def consistent_limits(self):
        if (self.max_cost_usd is None) != (self.cost_unknown_reason is not None):
            raise ValueError("Supply a cost ceiling or a reason it is unknown")
        tokens = (self.max_input_tokens, self.max_output_tokens)
        if (self.max_model_requests == 0 and any(tokens)
                or self.max_model_requests > 0 and not all(tokens)):
            raise ValueError("Request and token ceilings disagree")
        return self
