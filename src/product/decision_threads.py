"""Local decision threads linking immutable experiment attempts and stated choices."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, StringConstraints, TypeAdapter, model_validator

from src.product.ml_experiment_io import LocalFiles, decode, digest, encode
from src.product.ml_experiment_models import Timestamped
from src.product.ml_experiment_store import ExperimentStore, _terms
from src.product.ml_experiment_types import Digest, ExperimentID, Key, RecordModel, Text

_ID = re.compile(r"decision_[0-9a-f]{32}")
DecisionThreadID = Annotated[str, StringConstraints(pattern=r"^decision_[0-9a-f]{32}$")]


class DecisionBasis(RecordModel):
    experiment_id: ExperimentID
    run_id: Key | None = None
    research_note_id: Key | None = None


class DecisionChoice(RecordModel):
    decision_id: Key
    outcome: Literal["adopt", "reject", "defer"]
    rationale: Text
    revisit_when: Text | None = None
    author_kind: Literal["user_reported", "agent_proposed"]
    basis: list[DecisionBasis] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def clear_deferral(self):
        if self.outcome == "defer" and self.revisit_when is None:
            raise ValueError("A deferred decision needs a revisit condition")
        if len({(item.experiment_id, item.run_id, item.research_note_id)
                for item in self.basis}) != len(self.basis):
            raise ValueError("Duplicate decision basis")
        return self


class BasisReceipt(DecisionBasis):
    brief_sha256: Digest
    plan_sha256: Digest | None
    result_sha256: Digest | None = None
    research_sha256: Digest | None = None


class ThreadEvent(Timestamped):
    schema_version: Literal["decision-thread-event-v1"] = "decision-thread-event-v1"
    thread_id: DecisionThreadID
    revision: int = Field(ge=1)
    previous_sha256: Digest | None
    recorded_at: str
    kind: Literal["experiment_linked", "decision_recorded"]
    experiment_id: ExperimentID | None = None
    decision: DecisionChoice | None = None
    basis_receipts: list[BasisReceipt] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def one_event_kind(self):
        if self.kind == "experiment_linked":
            if self.experiment_id is None or self.decision is not None or self.basis_receipts:
                raise ValueError("A link event needs only an experiment ID")
        elif (self.experiment_id is not None or self.decision is None
              or len(self.basis_receipts) != len(self.decision.basis)
              or any(receipt.experiment_id != basis.experiment_id
                     or receipt.run_id != basis.run_id
                     or receipt.research_note_id != basis.research_note_id
                     for receipt, basis in zip(self.basis_receipts, self.decision.basis,
                                               strict=True))):
            raise ValueError("A decision event needs matching basis receipts")
        return self


class ThreadIndex(RecordModel):
    schema_version: Literal["decision-thread-v1"] = "decision-thread-v1"
    thread_id: DecisionThreadID
    question: Text
    events: list[Digest] = Field(min_length=1, max_length=100)


class ThreadEnvelope(RecordModel):
    sha256: Digest
    index: ThreadIndex


class DecisionThreads:
    """Append-only thread events; experiments remain the source of evidence."""

    def __init__(self, root: Path, experiments: ExperimentStore):
        self.files = LocalFiles(root)
        self.experiments = experiments

    @staticmethod
    def _id(thread_id: str) -> str:
        if not isinstance(thread_id, str) or not _ID.fullmatch(thread_id):
            raise ValueError("Invalid decision thread ID")
        return thread_id

    @staticmethod
    def _revision(expected_revision: int, actual: int) -> None:
        if type(expected_revision) is not int or expected_revision != actual:
            raise ValueError("Stale decision thread revision; retrieve the current thread")

    def _load(self, directory: int, thread_id: str):
        try:
            raw = decode(self.files.read(directory, f"{self._id(thread_id)}.json"))
            envelope = ThreadEnvelope.model_validate(raw)
            if digest(encode(raw["index"])) != envelope.sha256:
                raise ValueError("Decision thread index hash mismatch")
            index = envelope.index
            if index.thread_id != thread_id:
                raise ValueError("Decision thread identity mismatch")
            events, previous = [], None
            for revision, sha in enumerate(index.events, 1):
                event = ThreadEvent.model_validate(self.files.read_object(directory, sha))
                if (event.thread_id != thread_id or event.revision != revision
                        or event.previous_sha256 != previous):
                    raise ValueError("Decision thread event chain mismatch")
                events.append(event)
                previous = sha
            linked = set()
            decisions = set()
            for event in events:
                if event.kind == "experiment_linked":
                    if event.experiment_id in linked:
                        raise ValueError("Duplicate linked experiment")
                    linked.add(event.experiment_id)
                else:
                    if (event.decision.decision_id in decisions
                            or any(b.experiment_id not in linked
                                   for b in event.decision.basis)):
                        raise ValueError("Invalid decision history")
                    decisions.add(event.decision.decision_id)
            return index, events
        except ValueError as exc:
            raise ValueError("Decision thread is missing, unsafe, or corrupt") from exc

    def _write(self, directory: int, index: ThreadIndex):
        raw = index.model_dump(mode="json")
        envelope = ThreadEnvelope(sha256=digest(encode(raw)), index=index)
        self.files.write(directory, f"{index.thread_id}.json",
                         encode(envelope.model_dump(mode="json")))

    @staticmethod
    def _view(index: ThreadIndex, events: list[ThreadEvent]):
        linked = [e.experiment_id for e in events if e.kind == "experiment_linked"]
        decisions = [{"revision": e.revision, "recorded_at": e.recorded_at,
                      "choice": e.decision.model_dump(mode="json"),
                      "basis_receipts": [r.model_dump(mode="json")
                                         for r in e.basis_receipts]}
                     for e in events if e.kind == "decision_recorded"]
        return {"thread_id": index.thread_id, "question": index.question,
                "revision": len(events), "experiment_ids": linked,
                "decisions": decisions, "latest_decision": decisions[-1] if decisions else None,
                "warning": "Author kind is caller-declared, not proof of human review. "
                           "Hashes and source receipts prove provenance, not semantic support. "
                           "Inspect the linked experiment and selected run before reuse."}

    def create(self, question: str, experiment_id: str):
        question = TypeAdapter(Text).validate_python(question, strict=True)
        self.experiments.get(experiment_id)
        with self.files.locked() as directory:
            thread_id = f"decision_{uuid4().hex}"
            try:
                os.stat(f"{thread_id}.json", dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError("Generated decision thread ID already exists")
            event = ThreadEvent(thread_id=thread_id, revision=1, previous_sha256=None,
                                recorded_at=datetime.now(timezone.utc).isoformat(),
                                kind="experiment_linked", experiment_id=experiment_id)
            sha = self.files.object(directory, event.model_dump(mode="json"))
            index = ThreadIndex(thread_id=thread_id, question=question, events=[sha])
            self._write(directory, index)
            return self._view(index, [event])

    def get(self, thread_id: str):
        with self.files.locked() as directory:
            return self._view(*self._load(directory, thread_id))

    def find(self, question: str, limit: int = 5):
        if not isinstance(question, str) or not 3 <= len(question.strip()) <= 500:
            raise ValueError("Question must contain 3..500 characters")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("Limit must be an integer in 1..10")
        query = _terms(question)
        if not query:
            raise ValueError("Question needs a searchable term")
        matches = []
        with self.files.locked() as directory:
            for name in os.listdir(directory):
                if not _ID.fullmatch(name.removesuffix(".json")) or not name.endswith(".json"):
                    continue
                index, events = self._load(directory, name[:-5])
                hits = query & _terms(index.question)
                if hits:
                    view = self._view(index, events)
                    matches.append({"thread_id": index.thread_id, "question": index.question,
                                    "matched_terms": sorted(hits), "score": len(hits),
                                    "revision": len(events),
                                    "updated_at": events[-1].recorded_at,
                                    "experiment_count": len(view["experiment_ids"]),
                                    "latest_decision": view["latest_decision"]["choice"]["outcome"]
                                    if view["latest_decision"] else None})
        matches.sort(key=lambda item: (item["score"], item["updated_at"],
                                       item["thread_id"]), reverse=True)
        return {"query": question, "match_kind": "local_word_overlap",
                "matches": matches[:limit], "truncated": len(matches) > limit,
                "warning": "Word overlap suggests candidates; inspect the thread and its "
                           "linked experiments before using a decision."}

    def link(self, thread_id: str, expected_revision: int, experiment_id: str):
        self.experiments.get(experiment_id)
        with self.files.locked() as directory:
            index, events = self._load(directory, thread_id)
            self._revision(expected_revision, len(events))
            if any(e.experiment_id == experiment_id for e in events
                   if e.kind == "experiment_linked"):
                return self._view(index, events)
            if len(events) >= 100:
                raise ValueError("Decision thread has reached its 100-event limit")
            event = ThreadEvent(thread_id=thread_id, revision=len(events) + 1,
                                previous_sha256=index.events[-1],
                                recorded_at=datetime.now(timezone.utc).isoformat(),
                                kind="experiment_linked", experiment_id=experiment_id)
            sha = self.files.object(directory, event.model_dump(mode="json"))
            index = index.model_copy(update={"events": [*index.events, sha]})
            self._write(directory, index)
            return self._view(index, [*events, event])

    def _receipt(self, basis: DecisionBasis):
        view = self.experiments.get(basis.experiment_id, basis.run_id,
                                    basis.research_note_id)
        result = view["selected_result"] if basis.run_id is not None else None
        research = (view["selected_research_note"]
                    if basis.research_note_id is not None else None)
        return BasisReceipt(**basis.model_dump(), brief_sha256=view["revisions"][-1]["sha256"],
                            plan_sha256=view["plan_sha256"],
                            result_sha256=result["sha256"] if result else None,
                            research_sha256=research["sha256"] if research else None)

    def record(self, thread_id: str, expected_revision: int, decision: dict):
        choice = DecisionChoice.model_validate(decode(encode(decision)))
        receipts = [self._receipt(basis) for basis in choice.basis]
        with self.files.locked() as directory:
            index, events = self._load(directory, thread_id)
            existing = next((e for e in events if e.kind == "decision_recorded"
                             and e.decision.decision_id == choice.decision_id), None)
            if existing:
                if existing.decision != choice:
                    raise ValueError("Decision ID already has different immutable content")
                return self._view(index, events)
            self._revision(expected_revision, len(events))
            linked = {e.experiment_id for e in events if e.kind == "experiment_linked"}
            if any(b.experiment_id not in linked for b in choice.basis):
                raise ValueError("Decision basis must name a linked experiment")
            if len(events) >= 100:
                raise ValueError("Decision thread has reached its 100-event limit")
            event = ThreadEvent(thread_id=thread_id, revision=len(events) + 1,
                                previous_sha256=index.events[-1],
                                recorded_at=datetime.now(timezone.utc).isoformat(),
                                kind="decision_recorded", decision=choice,
                                basis_receipts=receipts)
            sha = self.files.object(directory, event.model_dump(mode="json"))
            index = index.model_copy(update={"events": [*index.events, sha]})
            self._write(directory, index)
            return self._view(index, [*events, event])
