"""Versioned question-first experiment drafts and immutable frozen plans."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.product.ml_experiment_io import LocalFiles, decode, digest, encode
from src.product.ml_experiment_models import (
    BriefRevision,
    ExperimentBrief,
    ExperimentIndex,
    FrozenPlan,
    IndexEnvelope,
    RevisionPointer,
)
from src.product.ml_experiment_outcomes import ExperimentOutcomes
from src.product.ml_experiment_research_notes import ExperimentResearchNotes

_ID = re.compile(r"experiment_[0-9a-f]{32}")


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid experiment ID")
    return value


def _expected(value: int, actual: int) -> None:
    if type(value) is not int or value != actual:
        raise ValueError("Stale or missing expected_revision; retrieve the current experiment")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentStore(ExperimentOutcomes, ExperimentResearchNotes):
    def __init__(self, root: Path, *, evidence_resolver=None):
        self.files = LocalFiles(root)
        self.evidence_resolver = evidence_resolver

    def _load(self, directory, experiment_id):
        name = f"{_identifier(experiment_id)}.json"
        try:
            raw = decode(self.files.read(directory, name))
            envelope = IndexEnvelope.model_validate(raw)
            if digest(encode(raw["index"])) != envelope.sha256:
                raise ValueError("Experiment index hash mismatch")
            index = envelope.index
            if index.experiment_id != experiment_id:
                raise ValueError("Experiment index identity mismatch")
            revisions, previous = [], None
            for pointer in index.revisions:
                record = BriefRevision.model_validate(
                    self.files.read_object(directory, pointer.sha256))
                if (record.experiment_id != experiment_id or record.revision != pointer.revision
                        or record.previous_sha256 != previous):
                    raise ValueError("Experiment revision chain mismatch")
                revisions.append(record)
                previous = pointer.sha256
            plan = None
            if index.frozen_plan_sha256:
                plan = FrozenPlan.model_validate(
                    self.files.read_object(directory, index.frozen_plan_sha256))
                if (plan.experiment_id != experiment_id or plan.revision != len(revisions)
                        or plan.revision_sha256 != previous or plan.brief != revisions[-1].brief
                        or plan.evidence_receipts != revisions[-1].evidence_receipts):
                    raise ValueError("Frozen plan binding mismatch")
            return (index, revisions, plan, self._read_results(directory, index, plan),
                    self._read_research(directory, index))
        except ValueError as exc:
            raise ValueError("Experiment is missing, unsafe, or corrupt") from exc

    def _write_index(self, directory, index):
        sha = digest(encode(index.model_dump(mode="json")))
        envelope = IndexEnvelope(sha256=sha, index=index)
        self.files.write(directory, f"{index.experiment_id}.json",
                         encode(envelope.model_dump(mode="json")))

    def _view(self, index, revisions, plan, records=(), research=(), run_id=None,
              research_note_id=None):
        return {"experiment_id": index.experiment_id, "revision": len(revisions),
                "status": "frozen" if plan else "draft",
                "brief": revisions[-1].brief.model_dump(mode="json"),
                "missing_fields": revisions[-1].brief.missing_fields(),
                "revisions": [{"revision": r.revision, "sha256": p.sha256,
                               "recorded_at": r.recorded_at}
                              for r, p in zip(revisions, index.revisions, strict=True)],
                "plan_sha256": index.frozen_plan_sha256,
                "frozen_plan": plan.model_dump(mode="json") if plan else None,
                "evidence_receipts": [r.model_dump(mode="json")
                                      for r in revisions[-1].evidence_receipts],
                "reference_warning": revisions[-1].reference_warning,
                **self._result_view(records, run_id),
                **self._research_view(research, research_note_id)}

    def save_brief(self, brief: dict, experiment_id: str | None = None,
                   expected_revision: int | None = None) -> dict:
        # Copy through bounded JSON: caller-owned dictionaries cannot mutate saved state.
        parsed = ExperimentBrief.model_validate(decode(encode(brief)))
        receipts = self.evidence_resolver(parsed.references) if self.evidence_resolver else []
        with self.files.locked() as directory:
            if experiment_id is None:
                if expected_revision is not None:
                    raise ValueError("A new experiment must not supply expected_revision")
                experiment_id = f"experiment_{uuid4().hex}"
                try:
                    os.stat(f"{experiment_id}.json", dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise ValueError("Generated experiment ID already exists; retry creation")
                revisions, plan, pointers, research, research_pointers = [], None, [], [], []
            else:
                index, revisions, plan, _, research = self._load(directory, experiment_id)
                _expected(expected_revision, len(revisions))
                if plan:
                    raise ValueError("Frozen plan cannot be edited; create a successor experiment")
                if len(revisions) >= 100:
                    raise ValueError("Experiment has reached its 100-revision limit")
                pointers = index.revisions
                research_pointers = index.research
            parent = parsed.supersedes_experiment_id
            if parent:
                if parent == experiment_id:
                    raise ValueError("An experiment cannot supersede itself")
                self._load(directory, parent)
            record = BriefRevision(experiment_id=experiment_id, revision=len(revisions) + 1,
                                   previous_sha256=pointers[-1].sha256 if pointers else None,
                                   recorded_at=_now(), brief=parsed, evidence_receipts=receipts)
            sha = self.files.object(directory, record.model_dump(mode="json"))
            index = ExperimentIndex(experiment_id=experiment_id, revisions=[*pointers,
                RevisionPointer(revision=record.revision, sha256=sha)], research=research_pointers)
            self._write_index(directory, index)
            return self._view(index, [*revisions, record], plan, research=research)

    def get(self, experiment_id: str, run_id: str | None = None,
            research_note_id: str | None = None) -> dict:
        with self.files.locked() as directory:
            return self._view(*self._load(directory, experiment_id), run_id=run_id,
                              research_note_id=research_note_id)

    def freeze_plan(self, experiment_id: str, expected_revision: int) -> dict:
        with self.files.locked() as directory:
            index, revisions, plan, records, research = self._load(directory, experiment_id)
            _expected(expected_revision, len(revisions))
            if plan:
                return self._view(index, revisions, plan, records, research)
            brief = revisions[-1].brief
            if (self.evidence_resolver is not None
                    and len(revisions[-1].evidence_receipts) != len(brief.references)):
                raise ValueError("Re-save the brief to verify its references before freezing")
            missing = brief.missing_fields()
            if missing:
                raise ValueError("Incomplete experiment plan: " + ", ".join(missing))
            plan = FrozenPlan(experiment_id=experiment_id, revision=len(revisions),
                              revision_sha256=index.revisions[-1].sha256,
                              frozen_at=_now(), brief=brief,
                              evidence_receipts=revisions[-1].evidence_receipts)
            sha = self.files.object(directory, plan.model_dump(mode="json"))
            index = index.model_copy(update={"frozen_plan_sha256": sha})
            self._write_index(directory, index)
            return self._view(index, revisions, plan, research=research)

    def list(self, limit: int = 20) -> dict:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer in 1..100")
        summaries = []
        with self.files.locked() as directory:
            for name in os.listdir(directory):
                if not name.startswith("experiment_") or not name.endswith(".json"):
                    continue
                experiment_id = _identifier(name[:-5])
                index, revisions, plan, records, research = self._load(directory, experiment_id)
                summaries.append({"experiment_id": experiment_id,
                                  "question": revisions[-1].brief.question,
                                  "revision": len(revisions),
                                  "status": "frozen" if plan else "draft",
                                  "result_count": len(records),
                                  "research_note_count": len(research),
                                  "updated_at": max([revisions[-1].recorded_at,
                                      *([plan.frozen_at] if plan else []),
                                      *[r.recorded_at for _, r in records],
                                      *[r.recorded_at for _, r in research]])})
        summaries.sort(key=lambda item: (item["updated_at"], item["experiment_id"]), reverse=True)
        return {"experiments": summaries[:limit], "truncated": len(summaries) > limit}
