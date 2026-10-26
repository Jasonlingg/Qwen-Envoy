"""Append-only research linked to brief revisions, without editing frozen plans."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, model_validator

from src.product.ml_experiment_io import decode, encode
from src.product.ml_experiment_models import ResearchPointer, Timestamped
from src.product.ml_experiment_receipts import EvidenceReceipt, check_receipts
from src.product.ml_experiment_types import Digest, ExperimentID, Key, RecordModel, Reference, Text
from src.product.research_justification import ResearchJustification, validate_paper_comparison


class ResearchFinding(RecordModel):
    claim: Text
    reference_ids: list[Key] = Field(min_length=1, max_length=5)


class ResearchNote(RecordModel):
    note_id: Key
    question: Text
    findings: list[ResearchFinding] = Field(default_factory=list, max_length=6)
    references: list[Reference] = Field(default_factory=list, max_length=20)
    uncertainties: list[Text] = Field(min_length=1, max_length=8)
    proposed_test: Text | None = None
    origin: Literal["host_research", "qwen_research"] = "host_research"
    data_kind: Literal["recorded", "synthetic"] = "recorded"
    semantic_support: Literal["not_reviewed"] = "not_reviewed"
    proposal_status: Literal["not_executed_or_validated"] = "not_executed_or_validated"
    justification: ResearchJustification | None = None

    @model_validator(mode="after")
    def valid_references(self):
        ids = [ref.reference_id for ref in self.references]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate research reference ID")
        if any(set(item.reference_ids) - set(ids) for item in self.findings):
            raise ValueError("Research finding names an unknown reference")
        if self.justification is not None:
            validate_paper_comparison(self.justification, self.references)
        return self


class ResearchRecord(Timestamped):
    schema_version: Literal["ml-experiment-research-v1"] = "ml-experiment-research-v1"
    experiment_id: ExperimentID
    brief_revision: int = Field(ge=1)
    brief_sha256: Digest
    recorded_at: str
    note: ResearchNote
    evidence_receipts: list[EvidenceReceipt] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def bound_evidence(self):
        check_receipts(self.note.references, self.evidence_receipts)
        return self


class ExperimentResearchNotes:
    def _read_research(self, directory, index):
        records = []
        for pointer in index.research:
            record = ResearchRecord.model_validate(
                self.files.read_object(directory, pointer.sha256))
            if (record.experiment_id != index.experiment_id
                    or record.note.note_id != pointer.note_id
                    or not 1 <= record.brief_revision <= len(index.revisions)
                    or record.brief_sha256 != index.revisions[record.brief_revision - 1].sha256):
                raise ValueError("Research note identity or brief binding mismatch")
            records.append((pointer.sha256, record))
        return records

    @staticmethod
    def _research_view(records, note_id=None):
        if note_id is not None and not any(r.note.note_id == note_id for _, r in records):
            raise ValueError("No research note has this note ID")
        selected = next(((sha, r) for sha, r in records if r.note.note_id == note_id),
                        records[-1] if records else None)
        return {"research_notes": [{"note_id": r.note.note_id, "sha256": sha,
                                   "brief_revision": r.brief_revision,
                                   "question": r.note.question, "recorded_at": r.recorded_at,
                                   "data_kind": r.note.data_kind}
                                  for sha, r in records],
                "selected_research_note": ({**selected[1].model_dump(mode="json"),
                                            "sha256": selected[0]} if selected else None)}

    def record_research(self, experiment_id, expected_revision, note):
        parsed = ResearchNote.model_validate(decode(encode(note)))
        with self.files.locked() as directory:
            index, revisions, plan, results, research = self._load(directory, experiment_id)
            existing = next((r for _, r in research if r.note.note_id == parsed.note_id), None)
            if existing:
                if (parsed != existing.note or type(expected_revision) is not int
                        or expected_revision != existing.brief_revision):
                    raise ValueError("Note ID already has different immutable research")
                return self._view(index, revisions, plan, results, research,
                                  research_note_id=parsed.note_id)
            if type(expected_revision) is not int or expected_revision != len(revisions):
                raise ValueError("Stale or missing expected_revision; resume the experiment")
            if len(research) >= 100:
                raise ValueError("Experiment has reached its 100-research-note limit")
            if parsed.references and self.evidence_resolver is None:
                raise ValueError("Research references require configured source verification")
            receipts = self.evidence_resolver(parsed.references) if self.evidence_resolver else []
            record = ResearchRecord(experiment_id=experiment_id, brief_revision=expected_revision,
                brief_sha256=index.revisions[-1].sha256,
                recorded_at=datetime.now(timezone.utc).isoformat(), note=parsed,
                evidence_receipts=receipts)
            sha = self.files.object(directory, record.model_dump(mode="json"))
            index = index.model_copy(update={"research": [*index.research,
                ResearchPointer(note_id=parsed.note_id, sha256=sha)]})
            self._write_index(directory, index)
            return self._view(index, revisions, plan, results, [*research, (sha, record)],
                              research_note_id=parsed.note_id)
