"""Local research-thread routes over frozen web sources and an Obsidian inbox.

This is a working, reviewable product path. Importing a URL never grants a
model network access, and neither source import nor thread acceptance promotes
an agent-authored draft into the reviewed Obsidian library.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal
from uuid import uuid4

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from src.harness.staging import stage_drafts
from src.product.chat import _verified_review
from src.product.learning_thread import (
    approve_revision,
    create_thread,
    discard_pending_proposal,
    get_thread,
    list_threads,
    propose_revision,
    record_staged_draft,
    render_proposal_markdown,
)
from src.product.memory import _record_context, inspect_evidence
from src.product.system_map import get_map, save_map
from src.research.agent import load_snapshot
from src.research.file_sources import (
    MAX_FILE_BYTES,
    build_file_snapshot,
    load_file_snapshot,
)
from src.research.file_sources import (
    PARSER_VERSION as FILE_PARSER_VERSION,
)
from src.research.tools_runtime import ResearchTools
from src.research.web_sources import (
    PARSER_VERSION as WEB_PARSER_VERSION,
)
from src.research.web_sources import (
    build_web_snapshot,
    load_web_snapshot,
)

_SOURCE_ID = re.compile(r"[0-9a-f]{32}\Z")
_THREAD_ID = re.compile(r"thread_[0-9a-f]{32}\Z")
_FRONTMATTER = re.compile(r"\A---\n.*?\n---(?:\n|$)", re.S)
_WEB_REF = re.compile(r"web:([0-9a-f]{32}):(\d+):(\d+):([0-9a-f]{16})\Z")
_FILE_REF = re.compile(r"file:([0-9a-f]{32}):(\d+):(\d+):([0-9a-f]{16})\Z")
_VAULT_REF = re.compile(
    r"vault:([0-9a-f]{16}):(vault_[0-9a-f]{24}):(\d+):(\d+):([0-9a-f]{16})\Z"
)
_MAX_EVIDENCE = 5
_MAX_QUOTE = 1_200


class ThreadInput(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    current_view: str = Field(min_length=1, max_length=4_000)
    effective_date: str


class SourceInput(BaseModel):
    url: str = Field(min_length=1, max_length=2_000)


class EvidenceInput(BaseModel):
    source_id: str
    start: int
    end: int
    relation: Literal["supports", "challenges", "context", "unclear"] = "unclear"


class ObservationInput(BaseModel):
    doc_id: str
    start: int
    end: int
    corpus_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    relation: Literal["supports", "challenges", "context", "unclear"] = "unclear"


class ProposalInput(BaseModel):
    revised_view: str = Field(min_length=1, max_length=4_000)
    reason: str = Field(min_length=1, max_length=4_000)
    effective_date: str
    evidence: list[EvidenceInput] = Field(min_length=1, max_length=_MAX_EVIDENCE)
    observation_refs: list[str] = Field(default_factory=list, max_length=_MAX_EVIDENCE)
    observations: list[ObservationInput] = Field(default_factory=list, max_length=_MAX_EVIDENCE)


class AcceptInput(BaseModel):
    confirmed: bool = False


class SuggestionInput(BaseModel):
    evidence: list[EvidenceInput] = Field(min_length=1, max_length=_MAX_EVIDENCE - 1)
    observations: list[ObservationInput] = Field(default_factory=list, max_length=_MAX_EVIDENCE - 1)


class MapNodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    type: Literal["component", "process", "assumption", "outcome", "stock", "flow"]
    label: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1_000)
    x: float = Field(ge=-10_000, le=10_000)
    y: float = Field(ge=-10_000, le=10_000)


class MapEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    relation: Literal["supports", "challenges", "context", "unclear"] = "unclear"
    reference: str | None = Field(default=None, max_length=200, exclude=True)


class MapEdgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    source: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    target: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    type: Literal["flows_to", "depends_on", "may_affect", "tests"]
    label: str = Field(default="", max_length=120)
    explanation: str = Field(default="", max_length=1_000)
    basis: Literal["hypothesis", "documented", "measured"]
    evidence: list[MapEvidenceInput] = Field(default_factory=list, max_length=5)
    receipts: list[dict] = Field(default_factory=list, max_length=5, exclude=True)


class MapInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[MapNodeInput] = Field(max_length=30)
    edges: list[MapEdgeInput] = Field(max_length=50)
    summary: str = Field(min_length=1, max_length=1_000)


class MapProposalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[MapNodeInput] = Field(max_length=30)
    edges: list[MapEdgeInput] = Field(max_length=50)


class CuratedMapProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(pattern=r"^thread_[0-9a-f]{32}$")
    authorship: Literal["curated_demo"]
    title: str = Field(min_length=1, max_length=120)
    rationale: str = Field(min_length=1, max_length=2_000)
    summary: str = Field(min_length=1, max_length=1_000)
    base_revision_id: str | None = Field(pattern=r"^maprev_[0-9a-f]{32}$")
    proposed_map: MapProposalInput


def _source_dir(root: Path, thread_id: str, source_id: str) -> Path:
    if _THREAD_ID.fullmatch(thread_id) is None or _SOURCE_ID.fullmatch(source_id) is None:
        raise ValueError("invalid thread or source ID")
    return root / "sources" / thread_id / source_id


def _source(root: Path, thread_id: str, source_id: str) -> tuple[dict, dict]:
    snapshot = _source_dir(root, thread_id, source_id)
    summary = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    parser = summary.get("parser_version")
    if parser == WEB_PARSER_VERSION:
        manifest, docs = load_web_snapshot(snapshot)
    elif parser == FILE_PARSER_VERSION:
        manifest, docs = load_file_snapshot(snapshot)
    else:
        raise ValueError("unsupported research source snapshot")
    if manifest.get("status") != "complete" or len(docs) != 1:
        raise ValueError("research source snapshot is incomplete")
    return manifest, next(iter(docs.values()))


def _span_pages(doc: dict, start: int, end: int) -> list[int]:
    return sorted({section["page"] for section in doc.get("sections", [])
                   if isinstance(section.get("page"), int)
                   and section["start"] < end and section["end"] > start})


def _passages(snapshot: Path, doc: dict, query: str) -> list[dict]:
    tools = ResearchTools(snapshot / "corpus")
    hits = tools.search_paper(doc["doc_id"], query, top_k=3)
    if not hits:
        length = min(_MAX_QUOTE, len(doc["text"]))
        hits = [tools.passage(doc["doc_id"], 0, length)]
    return [
        {"start": hit["start"], "end": min(hit["start"] + _MAX_QUOTE, hit["end"]),
         "quote": doc["text"][hit["start"]:min(hit["start"] + _MAX_QUOTE, hit["end"])],
         "pages": _span_pages(doc, hit["start"],
                              min(hit["start"] + _MAX_QUOTE, hit["end"]))}
        for hit in hits
    ]


def _source_card(vault: Path, root: Path, thread_id: str, source_id: str,
                 question: str) -> dict:
    manifest, doc = _source(root, thread_id, source_id)
    metadata = doc["metadata"]
    is_file = metadata["source_kind"] == "uploaded_file"
    excerpt = doc["text"][:30_000]
    truncated = len(excerpt) < len(doc["text"])
    frontmatter = {
        "kind": "source",
        "review_status": "agent_authored_draft",
        "source_kind": metadata["source_kind"],
        "source_snapshot_hash": manifest["corpus_hash"],
        "source_doc_id": doc["doc_id"],
        "source_revision": metadata["source_revision"],
        "coverage": metadata["coverage"],
        "thread_id": thread_id,
    }
    if is_file:
        frontmatter.update({
            "original_filename": metadata["filename"],
            "file_sha256": metadata["source_sha256"],
            "uploaded_at": metadata["uploaded_at"],
        })
        source_line = f"Attached file: {metadata['filename']} (SHA256: {metadata['source_sha256']})"
        review_hint = "Inspect the frozen original file before promoting this extracted text."
    else:
        frontmatter.update({
            "source_url": metadata["source_url"],
            "fetched_at": metadata["fetched_at"],
        })
        source_line = f"Source: {metadata['source_url']}"
        review_hint = "Inspect the original page and frozen snapshot before promoting this note."
    content = (
        "---\n" + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
        + "---\n\n# " + doc["title"] + "\n\n"
        + source_line + "\n\n"
        + "> Extracted text is an unreviewed source, not a verified finding. "
        + review_hint + "\n\n"
        + "## Captured text\n\n" + excerpt + "\n"
        + ("\n> Vault copy is truncated; inspect the frozen snapshot for the rest.\n"
           if truncated else "")
    )
    staged = stage_drafts(vault, f"source-{source_id}", {f"Sources/{source_id}.md": content})
    return {
        "source_id": source_id,
        "title": doc["title"],
        "source_kind": metadata["source_kind"],
        "source_url": None if is_file else metadata["source_url"],
        "filename": metadata.get("filename"),
        "download_url": (f"/api/lab/threads/{thread_id}/sources/{source_id}/raw"
                         if is_file else None),
        "coverage": metadata["coverage"],
        "fetched_at": metadata.get("fetched_at") or metadata.get("uploaded_at"),
        "source_snapshot_hash": manifest["corpus_hash"],
        "passages": _passages(_source_dir(root, thread_id, source_id), doc, question),
        "staged_path": str(staged.files[f"Sources/{source_id}.md"].relative_to(vault)),
        "promotion_status": "draft_in_inbox",
    }


def _sources(vault: Path, root: Path, thread_id: str, question: str) -> list[dict]:
    folder = root / "sources" / thread_id
    if not folder.exists():
        return []
    result = []
    for snapshot in sorted(folder.iterdir()):
        if not snapshot.is_dir() or _SOURCE_ID.fullmatch(snapshot.name) is None:
            continue
        try:
            manifest, doc = _source(root, thread_id, snapshot.name)
        except (ValueError, OSError, KeyError, TypeError):
            result.append({
                "source_id": snapshot.name,
                "title": "Frozen source unavailable or changed",
                "source_kind": "unavailable",
                "source_url": None,
                "filename": None,
                "download_url": None,
                "coverage": "unavailable",
                "fetched_at": None,
                "source_snapshot_hash": None,
                "passages": [],
                "staged_path": None,
                "promotion_status": "source_unavailable_or_changed",
            })
            continue
        draft_path = f"_inbox/source-{snapshot.name}/Sources/{snapshot.name}.md"
        result.append({
            "source_id": snapshot.name,
            "title": doc["title"],
            "source_kind": doc["metadata"]["source_kind"],
            "source_url": (doc["metadata"]["source_url"]
                           if doc["metadata"]["source_kind"] == "web_page" else None),
            "filename": doc["metadata"].get("filename"),
            "download_url": (
                f"/api/lab/threads/{thread_id}/sources/{snapshot.name}/raw"
                if doc["metadata"]["source_kind"] == "uploaded_file" else None
            ),
            "coverage": doc["metadata"]["coverage"],
            "fetched_at": (doc["metadata"].get("fetched_at")
                           or doc["metadata"].get("uploaded_at")),
            "source_snapshot_hash": manifest["corpus_hash"],
            "passages": _passages(snapshot, doc, question),
            "staged_path": draft_path,
            "promotion_status": ("draft_in_inbox" if (vault / draft_path).is_file()
                                 else "draft_moved_or_missing"),
        })
    return result


def _checked_evidence(root: Path, thread_id: str,
                      items: list[EvidenceInput]) -> tuple[list[str], list[dict]]:
    refs: list[str] = []
    receipts: list[dict] = []
    for item in items:
        manifest, doc = _source(root, thread_id, item.source_id)
        metadata = doc["metadata"]
        is_file = metadata["source_kind"] == "uploaded_file"
        start, end = item.start, item.end
        if not 0 <= start < end <= len(doc["text"]) or end - start > _MAX_QUOTE:
            raise ValueError("evidence span is outside the frozen source or too long")
        quote = doc["text"][start:end]
        if not quote.strip():
            raise ValueError("evidence quote is empty")
        quote_hash = hashlib.sha256(quote.encode("utf-8")).hexdigest()
        ref_kind = "file" if is_file else "web"
        ref = f"{ref_kind}:{item.source_id}:{start}:{end}:{quote_hash[:16]}"
        refs.append(ref)
        receipt = {
            "kind": "attached_file" if is_file else "web_source",
            "reference": ref,
            "title": doc["title"],
            "quote": quote,
            "source_snapshot_hash": manifest["corpus_hash"],
            "source_doc_id": doc["doc_id"],
            "start": start,
            "end": end,
            "relation": item.relation,
            "pages": _span_pages(doc, start, end),
        }
        if is_file:
            receipt.update({"filename": metadata["filename"],
                            "file_sha256": metadata["source_sha256"]})
        else:
            receipt["source_url"] = metadata["source_url"]
        receipts.append(receipt)
    if len(set(refs)) != len(refs):
        raise ValueError("duplicate evidence spans")
    return refs, receipts


def _checked_observations(snapshot: Path,
                          items: list[ObservationInput]) -> tuple[list[str], list[dict]]:
    manifest, docs = load_snapshot(snapshot)
    refs: list[str] = []
    receipts: list[dict] = []
    for item in items:
        if item.corpus_hash != manifest["corpus_hash"]:
            raise ValueError(
                "vault observation selection is stale; search the refreshed vault again"
            )
        doc = docs.get(item.doc_id)
        if doc is None:
            raise ValueError("observation document is not in the selected vault snapshot")
        start, end = item.start, item.end
        frontmatter = _FRONTMATTER.match(doc["text"])
        body_start = frontmatter.end() if frontmatter else 0
        if not body_start <= start < end <= len(doc["text"]) or end - start > _MAX_QUOTE:
            raise ValueError("observation span is outside the frozen note or too long")
        quote = doc["text"][start:end]
        if not quote.strip():
            raise ValueError("observation quote is empty")
        quote_hash = hashlib.sha256(quote.encode("utf-8")).hexdigest()
        ref = (f"vault:{manifest['corpus_hash'][:16]}:{item.doc_id}:"
               f"{start}:{end}:{quote_hash[:16]}")
        record = _record_context(doc)
        kind = ("vault_observation" if record.get("kind") in {
            "attempt", "result", "correction", "observation", "experiment",
        } else "vault_note")
        refs.append(ref)
        receipts.append({
            "kind": kind,
            "reference": ref,
            "title": doc["title"],
            "quote": quote,
            "source_path": doc["metadata"].get("source_path"),
            "source_snapshot_hash": manifest["corpus_hash"],
            "source_doc_id": item.doc_id,
            "source_kind": doc["metadata"].get("source_kind", "personal_note"),
            "record": record,
            "start": start,
            "end": end,
            "relation": item.relation,
        })
    if len(set(refs)) != len(refs):
        raise ValueError("duplicate observation spans")
    return refs, receipts


def _receipt_markdown(receipts: list[dict]) -> str:
    lines = [
        "", "## Citation key", "",
        "- E1: the user-stated current view shown above; this is a belief, not a test result.",
        *[f"- E{index}: `{receipt['reference']}`"
          for index, receipt in enumerate(receipts, 2)],
        "", "## Frozen source receipts", "",
    ]
    for index, receipt in enumerate(receipts, 2):
        lines.extend([
            f"### E{index}: {receipt['title']}", "",
            f"Kind: {receipt['kind']}", "",
            f"Reference: `{receipt['reference']}`", "",
            (f"Original URL: {receipt['source_url']}" if receipt["kind"] == "web_source"
             else (f"Attached file: {receipt['filename']} "
                   f"(SHA256: {receipt['file_sha256']})"
                   if receipt["kind"] == "attached_file"
                   else f"Vault note: {receipt['source_path']}")), "",
            f"Snapshot: `{receipt['source_snapshot_hash']}`", "",
            f"Offsets: {receipt['start']}–{receipt['end']}", "",
            *([f"PDF pages: {', '.join(map(str, receipt['pages']))}", ""]
              if receipt.get("pages") else []),
            f"Your assessment: {receipt['relation']} (interpretation, not verified support)", "",
            *["> " + line for line in receipt["quote"].splitlines()], "",
        ])
    lines.append("> Exact spans establish provenance. A person must judge support and meaning.")
    return "\n".join(lines) + "\n"


def _resolve_reference(vault: Path, state_dir: Path, root: Path,
                       thread_id: str, reference: str) -> dict:
    """Resolve a dated timeline pointer against its immutable source bytes."""
    source_ref = _WEB_REF.fullmatch(reference) or _FILE_REF.fullmatch(reference)
    try:
        if source_ref is not None:
            source_id, start_text, end_text, expected = source_ref.groups()
            manifest, doc = _source(root, thread_id, source_id)
            metadata = doc["metadata"]
            is_file = metadata["source_kind"] == "uploaded_file"
            if (reference.startswith("file:") != is_file):
                raise ValueError("reference source kind changed")
            start, end = int(start_text), int(end_text)
            if not 0 <= start < end <= len(doc["text"]):
                raise ValueError("source reference offsets are invalid")
            quote = doc["text"][start:end]
            if hashlib.sha256(quote.encode()).hexdigest()[:16] != expected:
                raise ValueError("source reference quote changed")
            resolved = {
                "reference": reference,
                "kind": "attached_file" if is_file else "web_source",
                "title": doc["title"], "quote": quote,
                "source_snapshot_hash": manifest["corpus_hash"],
                "start": start, "end": end, "status": "exact_span_verified",
                "pages": _span_pages(doc, start, end),
            }
            if is_file:
                resolved.update({"filename": metadata["filename"],
                                 "file_sha256": metadata["source_sha256"]})
            else:
                resolved["source_url"] = metadata["source_url"]
            return resolved
        note = _VAULT_REF.fullmatch(reference)
        if note is not None:
            prefix, doc_id, start_text, end_text, expected = note.groups()
            for snapshot in sorted(state_dir.glob("snapshot-*")):
                if not snapshot.is_dir():
                    continue
                try:
                    manifest, docs = load_snapshot(snapshot)
                except (OSError, ValueError, KeyError):
                    continue
                if (manifest.get("vault_root") != str(vault)
                        or not manifest["corpus_hash"].startswith(prefix)):
                    continue
                doc = docs.get(doc_id)
                if doc is None:
                    continue
                start, end = int(start_text), int(end_text)
                if not 0 <= start < end <= len(doc["text"]):
                    raise ValueError("vault reference offsets are invalid")
                quote = doc["text"][start:end]
                if hashlib.sha256(quote.encode()).hexdigest()[:16] != expected:
                    raise ValueError("vault reference quote changed")
                record = _record_context(doc)
                return {
                    "reference": reference,
                    "kind": ("vault_observation" if record.get("kind") in {
                        "attempt", "result", "correction", "observation", "experiment",
                    } else "vault_note"),
                    "title": doc["title"], "quote": quote,
                    "source_path": doc["metadata"].get("source_path"),
                    "source_snapshot_hash": manifest["corpus_hash"],
                    "start": start, "end": end, "status": "exact_span_verified",
                }
    except (ValueError, OSError, KeyError):
        pass
    return {"reference": reference, "status": "source_unavailable_or_changed"}


def _with_receipts(vault: Path, state_dir: Path, root: Path,
                   thread_id: str, state: dict) -> dict:
    for item in [*state["timeline"], *state["pending_proposals"]]:
        assessments = {row["reference"]: row["relation"]
                       for row in item.get("assessments", [])}
        for field, target in (("evidence_refs", "evidence_receipts"),
                              ("observation_refs", "observation_receipts")):
            item[target] = [
                {**_resolve_reference(vault, state_dir, root, thread_id, ref),
                 "relation": assessments.get(ref, "unclear")}
                for ref in item[field]
            ]
    return state


def _checked_map(root: Path, thread_id: str, body: MapInput) -> tuple[list[dict], list[dict]]:
    """Validate graph structure and freeze exact source pointers on its arrows."""
    nodes = [item.model_dump() for item in body.nodes]
    edges = [item.model_dump() for item in body.edges]
    node_ids = {item["id"] for item in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError("map contains duplicate node IDs")
    if len({item["id"] for item in edges}) != len(edges):
        raise ValueError("map contains duplicate edge IDs")
    evidence = []
    for input_edge, edge in zip(body.edges, edges, strict=True):
        if (edge["source"] not in node_ids or edge["target"] not in node_ids
                or edge["source"] == edge["target"]):
            raise ValueError("map edge must connect two different existing nodes")
        if edge["basis"] in {"documented", "measured"} and not edge["evidence"]:
            raise ValueError("documented or measured edges need at least one source passage")
        evidence.extend(input_edge.evidence)
    if len(evidence) > 50:
        raise ValueError("a map can cite at most 50 source passages")
    refs, _ = _checked_evidence(root, thread_id, evidence) if evidence else ([], [])
    ref_index = iter(refs)
    for input_edge, edge in zip(body.edges, edges, strict=True):
        for input_item, item in zip(input_edge.evidence, edge["evidence"], strict=True):
            reference = next(ref_index)
            if input_item.reference is not None and input_item.reference != reference:
                raise ValueError("map evidence reference does not match the frozen span")
            item["reference"] = reference
    return nodes, edges


def _map_with_receipts(vault: Path, state_dir: Path, root: Path,
                       thread_id: str, state: dict) -> dict:
    for edge in state["revision"]["edges"]:
        edge["receipts"] = [
            {**_resolve_reference(vault, state_dir, root, thread_id, item["reference"]),
             "relation": item["relation"]}
            for item in edge["evidence"]
        ]
    state["semantic_support"] = "not_verified"
    return state


def create_learning_lab_router(
    vault: Path,
    state_dir: Path,
    *,
    template: Path,
    refresh_vault: Callable[[], dict] | None = None,
    get_vault_snapshot: Callable[[], Path] | None = None,
    coordinator=None,
    investigator=None,
    record_model_call: Callable[[int], None] | None = None,
) -> APIRouter:
    """Return a local-only router mounted by the existing loopback web app."""
    root = state_dir / "learning_lab"
    threads = root / "threads"
    root.mkdir(parents=True, exist_ok=True)
    binding = root / "vault_binding.json"
    if binding.exists():
        bound = json.loads(binding.read_text(encoding="utf-8"))
        if bound.get("vault") != str(vault):
            raise ValueError("learning lab state belongs to a different vault")
    else:
        binding.write_text(json.dumps({"vault": str(vault)}) + "\n", encoding="utf-8")
    router = APIRouter()

    @router.get("/lab", response_class=HTMLResponse)
    def lab_page() -> str:
        return template.read_text(encoding="utf-8")

    @router.get("/lab/map", response_class=HTMLResponse)
    def map_page() -> str:
        return template.with_name("system_map.html").read_text(encoding="utf-8")

    @router.get("/api/lab/threads")
    def all_threads() -> dict:
        return {"threads": list_threads(threads)}

    @router.post("/api/lab/threads")
    def new_thread(body: ThreadInput) -> dict:
        try:
            return create_thread(
                threads,
                question=body.question,
                current_view=body.current_view,
                effective_date=body.effective_date,
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/lab/threads/{thread_id}")
    def thread(thread_id: str) -> dict:
        try:
            state = get_thread(threads, thread_id)
            state = _with_receipts(vault, state_dir, root, thread_id, state)
            return {**state, "sources": _sources(vault, root, thread_id, state["question"])}
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/lab/threads/{thread_id}/map")
    def current_map(thread_id: str) -> dict:
        try:
            state = get_thread(threads, thread_id)
            result = get_map(root / "maps", thread_id, state["question"])
            return _map_with_receipts(vault, state_dir, root, thread_id, result)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/lab/threads/{thread_id}/map/proposal")
    def map_proposal(thread_id: str) -> dict:
        """Show a locally curated suggestion against the current map revision."""
        try:
            state = get_thread(threads, thread_id)
            path = root / "map_proposals" / f"{thread_id}.json"
            if not path.is_file() or path.is_symlink():
                raise ValueError("no curated map proposal for this thread")
            proposal = CuratedMapProposal.model_validate_json(path.read_text(encoding="utf-8"))
            if proposal.thread_id != thread_id:
                raise ValueError("curated map proposal belongs to another thread")
            if any(not text.strip() for text in (
                    proposal.title, proposal.rationale, proposal.summary)):
                raise ValueError("curated map proposal text must contain content")
            current = get_map(root / "maps", thread_id, state["question"])
            if proposal.base_revision_id != current["revision"]["id"]:
                raise ValueError("curated map proposal is stale")
            checked = MapInput(
                nodes=proposal.proposed_map.nodes,
                edges=proposal.proposed_map.edges,
                summary=proposal.summary,
            )
            nodes, edges = _checked_map(root, thread_id, checked)
            hydrated = _map_with_receipts(
                vault, state_dir, root, thread_id,
                {"revision": {"edges": edges}},
            )
            return {
                "authorship": proposal.authorship,
                "title": proposal.title,
                "rationale": proposal.rationale,
                "summary": proposal.summary,
                "base_revision_id": proposal.base_revision_id,
                "proposed_map": {"nodes": nodes, "edges": hydrated["revision"]["edges"]},
            }
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.put("/api/lab/threads/{thread_id}/map")
    def update_map(thread_id: str, body: MapInput) -> dict:
        try:
            state = get_thread(threads, thread_id)
            if not body.summary.strip():
                raise ValueError("map change note must contain text")
            nodes, edges = _checked_map(root, thread_id, body)
            result = save_map(
                root / "maps", thread_id, state["question"],
                nodes=nodes, edges=edges, summary=body.summary.strip(),
            )
            return _map_with_receipts(vault, state_dir, root, thread_id, result)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/lab/threads/{thread_id}/map/revisions/{revision_id}")
    def historical_map(thread_id: str, revision_id: str) -> dict:
        try:
            state = get_thread(threads, thread_id)
            result = get_map(root / "maps", thread_id, state["question"], revision_id)
            return _map_with_receipts(vault, state_dir, root, thread_id, result)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/lab/threads/{thread_id}/sources")
    def import_source(thread_id: str, body: SourceInput) -> dict:
        source_id = uuid4().hex
        snapshot = None
        try:
            snapshot = _source_dir(root, thread_id, source_id)
            state = get_thread(threads, thread_id)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            build_web_snapshot(body.url, snapshot)
            return _source_card(vault, root, thread_id, source_id, state["question"])
        except (ValueError, OSError, TypeError, KeyError) as exc:
            if (snapshot is not None and snapshot.exists()
                    and not (vault / "_inbox" / f"source-{source_id}").exists()):
                shutil.rmtree(snapshot)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/lab/threads/{thread_id}/files")
    async def import_file(thread_id: str, request: Request, filename: str) -> dict:
        """Freeze one bounded local attachment; never write binary bytes to the vault."""
        source_id = uuid4().hex
        snapshot = None
        try:
            snapshot = _source_dir(root, thread_id, source_id)
            state = get_thread(threads, thread_id)
            if not filename or len(filename) > 255:
                raise ValueError("filename must be 1–255 characters")
            raw = bytearray()
            async for chunk in request.stream():
                if len(raw) + len(chunk) > MAX_FILE_BYTES:
                    raise HTTPException(status_code=413, detail="attachment exceeds 5 MB")
                raw.extend(chunk)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            build_file_snapshot(filename, bytes(raw), snapshot)
            return _source_card(vault, root, thread_id, source_id, state["question"])
        except HTTPException:
            raise
        except (ValueError, OSError, TypeError, KeyError) as exc:
            if (snapshot is not None and snapshot.exists()
                    and not (vault / "_inbox" / f"source-{source_id}").exists()):
                shutil.rmtree(snapshot)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/lab/threads/{thread_id}/sources/{source_id}/raw")
    def download_file(thread_id: str, source_id: str) -> FileResponse:
        try:
            get_thread(threads, thread_id)
            _, doc = _source(root, thread_id, source_id)
            metadata = doc["metadata"]
            if metadata["source_kind"] != "uploaded_file":
                raise ValueError("this source has no uploaded file")
            filename = metadata["filename"]
            raw_path = _source_dir(root, thread_id, source_id) / "raw"
            raw_path /= f"{doc['doc_id']}{Path(filename).suffix.lower()}"
            if not raw_path.is_file() or raw_path.is_symlink():
                raise ValueError("frozen attachment is missing")
            return FileResponse(
                raw_path, filename=filename, media_type="application/octet-stream",
                content_disposition_type="attachment",
            )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/lab/threads/{thread_id}/observations")
    def observations(thread_id: str, query: str) -> dict:
        if get_vault_snapshot is None:
            raise HTTPException(status_code=503, detail="vault search is unavailable")
        try:
            get_thread(threads, thread_id)
            review = inspect_evidence(get_vault_snapshot(), query, top_k=5)
            return {"corpus_hash": review["corpus_hash"],
                    "evidence": review["evidence"], "warning": review["warning"]}
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/lab/threads/{thread_id}/investigate-vault")
    def investigate_vault(thread_id: str) -> dict:
        """Offer checked Qwen note spans for review; never change the thread."""
        if investigator is None or get_vault_snapshot is None:
            raise HTTPException(status_code=503, detail="Qwen investigation is not configured")
        try:
            state = get_thread(threads, thread_id)
            snapshot = get_vault_snapshot()
            packet = investigator.investigate(state["question"], snapshot)
            if not isinstance(packet, dict):
                raise ValueError("Qwen returned no investigation record")
            attempts = packet.get("model_requests_attempted", 0)
            if type(attempts) is not int or not 0 <= attempts <= 100:
                raise ValueError("Qwen returned an invalid request count")
            if record_model_call is not None:
                record_model_call(attempts)
            status = packet.get("status")
            evidence: list[dict] = []
            corpus_hash = None
            if status in {"evidence_found", "no_evidence"}:
                review = _verified_review(snapshot, packet, state["question"])
                corpus_hash = review["corpus_hash"]
                evidence = [{**item, "corpus_hash": corpus_hash}
                            for item in review["evidence"]]
            elif status not in {"unavailable", "error", "incomplete"}:
                raise ValueError("Qwen returned an unknown investigation status")
            run_id = uuid4().hex
            runs = root / "runs"
            runs.mkdir(parents=True, exist_ok=True)
            (runs / f"{run_id}.json").write_text(json.dumps({
                "run_id": run_id,
                "thread_id": thread_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "vault_snapshot": str(snapshot),
                "packet": packet,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return {
                "status": status,
                "error": packet.get("error"),
                "warning": packet.get("warning", "Qwen output needs human review."),
                "model_identity": packet.get("model_identity"),
                "model_requests_attempted": attempts,
                "evidence": evidence,
                "corpus_hash": corpus_hash,
                "run_id": run_id,
                "trajectory_path": str((runs / f"{run_id}.json").relative_to(state_dir)),
            }
        except (ValueError, OSError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/lab/threads/{thread_id}/proposals")
    def new_proposal(thread_id: str, body: ProposalInput) -> dict:
        try:
            get_thread(threads, thread_id)
            if body.observation_refs:
                raise ValueError("submit selected vault observations, not unchecked reference IDs")
            web_refs, receipts = _checked_evidence(root, thread_id, body.evidence)
            observation_refs, observation_receipts = (
                _checked_observations(get_vault_snapshot(), body.observations)
                if body.observations and get_vault_snapshot is not None else ([], [])
            )
            if body.observations and get_vault_snapshot is None:
                raise ValueError("vault search is unavailable")
            if len(receipts) + len(observation_receipts) > _MAX_EVIDENCE:
                raise ValueError("at most five total evidence passages are allowed")
            proposal = propose_revision(
                threads, thread_id,
                revised_view=body.revised_view,
                reason=body.reason,
                effective_date=body.effective_date,
                evidence_refs=web_refs,
                observation_refs=observation_refs,
                assessments=[
                    {"reference": receipt["reference"], "relation": receipt["relation"]}
                    for receipt in [*receipts, *observation_receipts]
                ],
            )
            try:
                markdown = render_proposal_markdown(threads, thread_id, proposal["proposal_id"])
                all_receipts = receipts + observation_receipts
                markdown += _receipt_markdown(all_receipts)
                name = f"Learning/{thread_id}-{proposal['proposal_id']}.md"
                staged = stage_drafts(vault, proposal["proposal_id"], {name: markdown})
                proposal = record_staged_draft(
                    threads, thread_id, proposal["proposal_id"], staged.hashes[name],
                )
            except Exception:
                discard_pending_proposal(threads, thread_id, proposal["proposal_id"])
                raise
            return {
                "proposal": proposal,
                "staged_path": str(staged.files[name].relative_to(vault)),
                "source_receipts": all_receipts,
                "promotion_status": "draft_in_inbox",
            }
        except (ValueError, OSError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/lab/threads/{thread_id}/suggest")
    def suggest(thread_id: str, body: SuggestionInput) -> dict:
        """Draft wording from checked spans; this never changes the thread."""
        if coordinator is None:
            raise HTTPException(status_code=400, detail="Nemotron is not configured")
        try:
            state = get_thread(threads, thread_id)
            _, receipts = _checked_evidence(root, thread_id, body.evidence)
            _, observation_receipts = (
                _checked_observations(get_vault_snapshot(), body.observations)
                if body.observations and get_vault_snapshot is not None else ([], [])
            )
            if body.observations and get_vault_snapshot is None:
                raise ValueError("vault search is unavailable")
            receipts += observation_receipts
            if len(receipts) >= _MAX_EVIDENCE:
                raise ValueError("Nemotron can review at most four selected passages")
            prior = state["current_approved_view"]
            packet = {
                "schema_version": "learning-lab-suggestion-v1",
                "retriever_protocol": "frozen_web_source_spans",
                "warning": "Exact spans prove provenance, not semantic support.",
                "evidence": [{
                    "evidence_id": "E1",
                    "quote": prior["view"][:_MAX_QUOTE],
                    "title": "User-stated current view",
                    "source_kind": "user_statement",
                    "record": {"effective_date": prior["effective_date"],
                               "status": "working_view"},
                }] + [{
                    "evidence_id": f"E{index}",
                    "quote": item["quote"],
                    "title": item["title"],
                    "source_kind": ("web_page" if item["kind"] == "web_source"
                                    else ("uploaded_file" if item["kind"] == "attached_file"
                                          else item.get("source_kind", "personal_note"))),
                    "source_url": item.get("source_url"),
                    "filename": item.get("filename"),
                    "source_path": item.get("source_path"),
                    "doc_id": item["source_doc_id"],
                    "record": item.get("record"),
                } for index, item in enumerate(receipts, 2)],
            }
            question = (
                f"For my saved question, {state['question']} Draft a cautious possible "
                "update to my current view based only on these passages. "
                "Separate my stated belief from source claims; do not invent an "
                "experiment or claim the source settles the question. Include one "
                "specific remaining uncertainty. This is wording for my review, "
                "not an approved conclusion."
            )
            if record_model_call is not None:
                record_model_call(1)
            answer = coordinator.answer(question, packet, history=[])
            if not isinstance(answer, dict) or not isinstance(answer.get("answer"), str):
                raise ValueError("Nemotron returned no usable draft")
            return {
                "suggested_view": answer["answer"],
                "reason": "Review the cited passages and edit this suggestion before staging.",
                "limitations": answer.get("limitations", []),
                "model": answer.get("model"),
                "model_requests_attempted": 1,
                "semantic_support": "not_reviewed",
                "citation_key": {
                    "E1": "user-stated current view",
                    **{f"E{index}": item["reference"]
                       for index, item in enumerate(receipts, 2)},
                },
            }
        except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/lab/threads/{thread_id}/proposals/{proposal_id}/accept")
    def accept(thread_id: str, proposal_id: str, body: AcceptInput) -> dict:
        if body.confirmed is not True:
            raise HTTPException(status_code=400, detail="explicit confirmation is required")
        try:
            state = get_thread(threads, thread_id)
            matches = [item for item in state["pending_proposals"]
                       if item["proposal_id"] == proposal_id]
            if len(matches) != 1 or not matches[0].get("staged_sha256"):
                raise ValueError("proposal is missing, stale, or was not staged")
            expected_hash = matches[0]["staged_sha256"]
            staged = vault / "_inbox" / proposal_id
            name = f"Learning/{thread_id}-{proposal_id}.md"
            draft = staged / name
            manifest_path = staged / "manifest.json"
            if (not draft.is_file() or draft.is_symlink()
                    or not manifest_path.is_file() or manifest_path.is_symlink()):
                raise ValueError("the exact staged proposal draft is missing")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            draft_bytes = draft.read_bytes()
            actual_hash = hashlib.sha256(draft_bytes).hexdigest()
            if (manifest.get("schema_version") != "staged-drafts-v1"
                    or manifest.get("run_id") != proposal_id
                    or manifest.get("files") != {name: {
                        "sha256": expected_hash, "bytes": len(draft_bytes),
                    }}
                    or actual_hash != expected_hash):
                raise ValueError("staged draft changed; create a fresh proposal before acceptance")
            revision = approve_revision(
                threads, thread_id, proposal_id,
                approved_by="local_user", confirmed=True,
            )
            return {
                "revision": revision,
                "thread": get_thread(threads, thread_id),
                "vault_promotion_status": "pending_manual_promotion",
            }
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if refresh_vault is not None:
        @router.post("/api/lab/refresh-vault")
        def refresh() -> dict:
            try:
                return refresh_vault()
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
