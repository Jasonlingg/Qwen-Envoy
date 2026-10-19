"""Internal source receipts; callers cannot supply these through the brief schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from src.product.ml_experiment_io import digest, encode
from src.product.ml_experiment_types import Digest, Key, RecordModel


class EvidenceReceipt(RecordModel):
    reference_id: Key
    reference_sha256: Digest
    kind: Literal["paper_span", "local_artifact", "external_reference"]
    verification: Literal["paper_span_verified", "artifact_value_verified", "reference_only"]
    source_sha256: Digest | None
    value: JsonValue
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    semantic_support: Literal["not_reviewed"] = "not_reviewed"

    @model_validator(mode="after")
    def appropriate_verification(self):
        expected = {"paper_span": "paper_span_verified",
                    "local_artifact": "artifact_value_verified",
                    "external_reference": "reference_only"}[self.kind]
        if self.verification != expected:
            raise ValueError("Receipt verification does not match reference kind")
        if (self.kind == "external_reference") != (self.source_sha256 is None):
            raise ValueError("Verified source receipts need a source hash")
        if len(encode(self.value).decode("utf-8")) > 4_000:
            raise ValueError("Selected evidence value exceeds 4000 characters; narrow the pointer")
        return self


def reference_hash(reference) -> str:
    return digest(encode(reference.model_dump(mode="json")))


def check_receipts(references, receipts) -> None:
    if (len(references) != len(receipts) or any(
            ref.reference_id != receipt.reference_id or ref.kind != receipt.kind
            or reference_hash(ref) != receipt.reference_sha256
            for ref, receipt in zip(references, receipts, strict=True))):
        raise ValueError("Evidence receipts do not bind to the exact brief references")
