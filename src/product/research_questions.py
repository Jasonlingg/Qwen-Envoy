"""Question-first, append-only links from paper evidence to attempted experiments."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, StringConstraints, TypeAdapter, model_validator

from src.product.ml_experiment_io import LocalFiles, decode, digest, encode
from src.product.ml_experiment_store import ExperimentStore, _terms
from src.product.ml_experiment_types import Digest, ExperimentID, Key, RecordModel, Text

QuestionID = Annotated[str, StringConstraints(pattern=r"^question_[0-9a-f]{32}$")]
_QUESTION = re.compile(r"question_[0-9a-f]{32}")


class PaperBasis(RecordModel):
    experiment_id: ExperimentID
    research_note_id: Key
    reference_id: Key


class ResultBasis(RecordModel):
    experiment_id: ExperimentID
    run_id: Key


class ConsideredPaper(RecordModel):
    paper_id: Key
    source: PaperBasis
    relevance: Text
    limitation: Text
    author_kind: Literal["user_reported", "agent_proposed"]


class PaperReceipt(RecordModel):
    note_sha256: Digest
    note_data_kind: str
    document_sha256: Digest
    source_url: Text
    quote: Text


class PaperToResult(RecordModel):
    link_id: Key
    paper: PaperBasis
    result: ResultBasis
    method_tried: Text
    observed_outcome: Text
    interpretation: Text
    next_step: Text | None = None
    author_kind: Literal["user_reported", "agent_proposed"]


class LinkReceipt(RecordModel):
    paper_note_sha256: Digest
    paper_note_data_kind: str
    paper_document_sha256: Digest
    paper_source_url: Text
    paper_quote: Text
    result_sha256: Digest
    result_data_kind: str
    result_assessment: str


class ViewRevision(RecordModel):
    view_id: Key
    statement: Text
    reason: Text
    next_question: Text
    basis_link_ids: list[Key] = Field(default_factory=list, max_length=20)
    author_kind: Literal["user_reported", "agent_proposed"]

    @model_validator(mode="after")
    def unique_basis(self):
        if len(set(self.basis_link_ids)) != len(self.basis_link_ids):
            raise ValueError("Duplicate view basis link")
        return self


class QuestionEvent(RecordModel):
    schema_version: Literal["research-question-event-v1"] = "research-question-event-v1"
    question_id: QuestionID
    revision: int = Field(ge=1)
    previous_sha256: Digest | None
    recorded_at: str
    kind: Literal["opened", "paper_considered", "paper_to_result", "view_recorded"]
    paper: ConsideredPaper | None = None
    paper_receipt: PaperReceipt | None = None
    link: PaperToResult | None = None
    receipt: LinkReceipt | None = None
    view: ViewRevision | None = None

    @model_validator(mode="after")
    def matching_payload(self):
        valid = {
            "opened": all(value is None for value in (
                self.paper, self.paper_receipt, self.link, self.receipt, self.view)),
            "paper_considered": self.paper is not None and self.paper_receipt is not None
            and self.link is None and self.receipt is None and self.view is None,
            "paper_to_result": self.link is not None and self.receipt is not None
            and self.paper is None and self.paper_receipt is None and self.view is None,
            "view_recorded": self.view is not None and self.link is None
            and self.receipt is None and self.paper is None and self.paper_receipt is None,
        }
        if not valid[self.kind]:
            raise ValueError("Question event kind and payload disagree")
        return self


class QuestionIndex(RecordModel):
    schema_version: Literal["research-question-v1"] = "research-question-v1"
    question_id: QuestionID
    question: Text
    events: list[Digest] = Field(min_length=1, max_length=100)


class QuestionEnvelope(RecordModel):
    sha256: Digest
    index: QuestionIndex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchQuestions:
    """Durable question trail; evidence verification stays in ExperimentStore."""

    def __init__(self, root: Path, experiments: ExperimentStore):
        self.files = LocalFiles(root)
        self.experiments = experiments

    @staticmethod
    def _id(question_id: str) -> str:
        if not isinstance(question_id, str) or not _QUESTION.fullmatch(question_id):
            raise ValueError("Invalid research question ID")
        return question_id

    @staticmethod
    def _revision(expected: int, actual: int) -> None:
        if type(expected) is not int or expected != actual:
            raise ValueError("Stale research question revision; retrieve it again")

    def _load(self, directory: int, question_id: str):
        try:
            raw = decode(self.files.read(directory, f"{self._id(question_id)}.json"))
            envelope = QuestionEnvelope.model_validate(raw)
            if digest(encode(raw["index"])) != envelope.sha256:
                raise ValueError("Question index hash mismatch")
            index = envelope.index
            if index.question_id != question_id:
                raise ValueError("Question identity mismatch")
            events, previous = [], None
            paper_ids, link_ids, view_ids = set(), set(), set()
            for revision, sha in enumerate(index.events, 1):
                event = QuestionEvent.model_validate(self.files.read_object(directory, sha))
                if (event.question_id != question_id or event.revision != revision
                        or event.previous_sha256 != previous
                        or (revision == 1) != (event.kind == "opened")):
                    raise ValueError("Question event chain mismatch")
                if event.paper:
                    if event.paper.paper_id in paper_ids:
                        raise ValueError("Duplicate considered paper")
                    paper_ids.add(event.paper.paper_id)
                if event.link:
                    if event.link.link_id in link_ids:
                        raise ValueError("Duplicate paper-to-result link")
                    link_ids.add(event.link.link_id)
                if event.view:
                    if (event.view.view_id in view_ids
                            or any(key not in link_ids for key in event.view.basis_link_ids)):
                        raise ValueError("Invalid view history")
                    view_ids.add(event.view.view_id)
                events.append(event)
                previous = sha
            return index, events
        except ValueError as exc:
            raise ValueError("Research question is missing, unsafe, or corrupt") from exc

    def _write(self, directory: int, index: QuestionIndex) -> None:
        raw = index.model_dump(mode="json")
        envelope = QuestionEnvelope(sha256=digest(encode(raw)), index=index)
        self.files.write(directory, f"{index.question_id}.json",
                         encode(envelope.model_dump(mode="json")))

    @staticmethod
    def _view(index: QuestionIndex, events: list[QuestionEvent]) -> dict:
        papers = [{"recorded_at": event.recorded_at,
                   "paper": event.paper.model_dump(mode="json"),
                   "receipt": event.paper_receipt.model_dump(mode="json")}
                  for event in events if event.paper]
        links = [{"recorded_at": event.recorded_at,
                  "link": event.link.model_dump(mode="json"),
                  "receipt": event.receipt.model_dump(mode="json")}
                 for event in events if event.link]
        views = [{"recorded_at": event.recorded_at,
                  "view": event.view.model_dump(mode="json")}
                 for event in events if event.view]
        return {"question_id": index.question_id, "question": index.question,
                "revision": len(events), "created_at": events[0].recorded_at,
                "updated_at": events[-1].recorded_at, "considered_papers": papers,
                "paper_to_result_links": links,
                "view_history": views, "current_view": views[-1] if views else None,
                "warning": "Paper text and result hashes establish provenance, not causation "
                           "or semantic support. Link interpretations and author_kind are "
                           "caller-declared. Inspect the underlying note and run."}

    def start(self, question: str) -> dict:
        question = TypeAdapter(Text).validate_python(question, strict=True)
        question_id = f"question_{uuid4().hex}"
        index = QuestionIndex(question_id=question_id, question=question,
                              events=["0" * 64])
        with self.files.locked() as directory:
            try:
                os.stat(f"{question_id}.json", dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError("Generated question ID already exists")
            event = QuestionEvent(question_id=question_id, revision=1,
                                  previous_sha256=None, recorded_at=_now(), kind="opened")
            sha = self.files.object(directory, event.model_dump(mode="json"))
            index = index.model_copy(update={"events": [sha]})
            self._write(directory, index)
            return self._view(index, [event])

    def get(self, question_id: str) -> dict:
        with self.files.locked() as directory:
            return self._view(*self._load(directory, question_id))

    def find(self, query: str | None = None, limit: int = 20) -> dict:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Limit must be an integer in 1..100")
        if query is not None and (not isinstance(query, str)
                                  or not 3 <= len(query.strip()) <= 500):
            raise ValueError("Query must contain 3..500 characters")
        terms = _terms(query) if query else set()
        if query and not terms:
            raise ValueError("Query needs a searchable term")
        matches = []
        with self.files.locked() as directory:
            for name in os.listdir(directory):
                if not name.endswith(".json") or not _QUESTION.fullmatch(name[:-5]):
                    continue
                index, events = self._load(directory, name[:-5])
                view = self._view(index, events)
                corpus = " ".join([index.question,
                    *[row["paper"]["relevance"] for row in view["considered_papers"]],
                    *[row["link"]["method_tried"] for row in view["paper_to_result_links"]],
                    *[row["view"]["statement"] for row in view["view_history"]]])
                hits = sorted(terms & _terms(corpus)) if query else []
                if query and not hits:
                    continue
                matches.append({"question_id": index.question_id, "question": index.question,
                                "revision": len(events), "updated_at": events[-1].recorded_at,
                                "paper_count": len(view["considered_papers"]),
                                "link_count": len(view["paper_to_result_links"]),
                                "current_view": (view["current_view"]["view"]["statement"]
                                                 if view["current_view"] else None),
                                "matched_terms": hits, "score": len(hits)})
        matches.sort(key=lambda row: (row["score"], row["updated_at"],
                                       row["question_id"]), reverse=True)
        return {"query": query, "match_kind": "local_word_overlap" if query else "recent",
                "matches": matches[:limit], "truncated": len(matches) > limit}

    def _paper_receipt(self, basis: PaperBasis) -> PaperReceipt:
        source = self.experiments.get(basis.experiment_id,
                                      research_note_id=basis.research_note_id)
        note = source["selected_research_note"]
        if note is None:
            raise ValueError("Paper research note does not exist")
        passage = next((row for row in note["evidence_receipts"]
                        if row["reference_id"] == basis.reference_id
                        and row["verification"] == "paper_span_verified"), None)
        if passage is None:
            raise ValueError("Source reference is not a verified paper span")
        return PaperReceipt(note_sha256=note["sha256"],
                            note_data_kind=note["note"]["data_kind"],
                            document_sha256=passage["source_sha256"],
                            source_url=passage["metadata"]["source_url"],
                            quote=passage["value"])

    def _receipt(self, link: PaperToResult) -> LinkReceipt:
        paper = self._paper_receipt(link.paper)
        target = self.experiments.get(link.result.experiment_id,
                                      run_id=link.result.run_id)
        result = target["selected_result"]
        if result is None:
            raise ValueError("Linked attempt has no recorded result")
        return LinkReceipt(paper_note_sha256=paper.note_sha256,
                           paper_note_data_kind=paper.note_data_kind,
                           paper_document_sha256=paper.document_sha256,
                           paper_source_url=paper.source_url,
                           paper_quote=paper.quote,
                           result_sha256=result["sha256"],
                           result_data_kind=result["result"]["data_kind"],
                           result_assessment=result["assessment"]["overall"])

    def consider_paper(self, question_id: str, expected_revision: int,
                       paper: dict) -> dict:
        parsed = ConsideredPaper.model_validate(decode(encode(paper)))
        with self.files.locked() as directory:
            index, events = self._load(directory, question_id)
            existing = next((event for event in events if event.paper
                             and event.paper.paper_id == parsed.paper_id), None)
            if existing:
                if existing.paper != parsed:
                    raise ValueError("Paper ID already has different immutable content")
                return self._view(index, events)
            self._revision(expected_revision, len(events))
        receipt = self._paper_receipt(parsed.source)
        with self.files.locked() as directory:
            index, events = self._load(directory, question_id)
            existing = next((event for event in events if event.paper
                             and event.paper.paper_id == parsed.paper_id), None)
            if existing:
                if existing.paper != parsed:
                    raise ValueError("Paper ID already has different immutable content")
                return self._view(index, events)
            self._revision(expected_revision, len(events))
            if len(events) >= 100:
                raise ValueError("Question has reached its 100-event limit")
            event = QuestionEvent(question_id=question_id, revision=len(events) + 1,
                                  previous_sha256=index.events[-1], recorded_at=_now(),
                                  kind="paper_considered", paper=parsed,
                                  paper_receipt=receipt)
            sha = self.files.object(directory, event.model_dump(mode="json"))
            index = index.model_copy(update={"events": [*index.events, sha]})
            self._write(directory, index)
            return self._view(index, [*events, event])

    def link(self, question_id: str, expected_revision: int, link: dict) -> dict:
        parsed = PaperToResult.model_validate(decode(encode(link)))
        with self.files.locked() as directory:
            index, events = self._load(directory, question_id)
            existing = next((event for event in events if event.link
                             and event.link.link_id == parsed.link_id), None)
            if existing:
                if existing.link != parsed:
                    raise ValueError("Link ID already has different immutable content")
                return self._view(index, events)
            self._revision(expected_revision, len(events))
        receipt = self._receipt(parsed)
        with self.files.locked() as directory:
            index, events = self._load(directory, question_id)
            existing = next((event for event in events if event.link
                             and event.link.link_id == parsed.link_id), None)
            if existing:
                if existing.link != parsed:
                    raise ValueError("Link ID already has different immutable content")
                return self._view(index, events)
            self._revision(expected_revision, len(events))
            if len(events) >= 100:
                raise ValueError("Question has reached its 100-event limit")
            event = QuestionEvent(question_id=question_id, revision=len(events) + 1,
                                  previous_sha256=index.events[-1], recorded_at=_now(),
                                  kind="paper_to_result", link=parsed, receipt=receipt)
            sha = self.files.object(directory, event.model_dump(mode="json"))
            index = index.model_copy(update={"events": [*index.events, sha]})
            self._write(directory, index)
            return self._view(index, [*events, event])

    def record_view(self, question_id: str, expected_revision: int, view: dict) -> dict:
        parsed = ViewRevision.model_validate(decode(encode(view)))
        with self.files.locked() as directory:
            index, events = self._load(directory, question_id)
            existing = next((event for event in events if event.view
                             and event.view.view_id == parsed.view_id), None)
            if existing:
                if existing.view != parsed:
                    raise ValueError("View ID already has different immutable content")
                return self._view(index, events)
            self._revision(expected_revision, len(events))
            links = {event.link.link_id for event in events if event.link}
            if any(key not in links for key in parsed.basis_link_ids):
                raise ValueError("View basis must name a link in this question")
            if len(events) >= 100:
                raise ValueError("Question has reached its 100-event limit")
            event = QuestionEvent(question_id=question_id, revision=len(events) + 1,
                                  previous_sha256=index.events[-1], recorded_at=_now(),
                                  kind="view_recorded", view=parsed)
            sha = self.files.object(directory, event.model_dump(mode="json"))
            index = index.model_copy(update={"events": [*index.events, sha]})
            self._write(directory, index)
            return self._view(index, [*events, event])
