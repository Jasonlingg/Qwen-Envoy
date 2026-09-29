"""Build a provenance-checked handoff from a frozen research snapshot.

This module does not decide whether a passage supports a proposed finding. The
caller supplies trusted run metadata and document IDs from actual tool returns;
model-proposed spans and findings are checked before entering the bundle.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Literal, TypedDict

from src.research.agent import load_snapshot

SCHEMA_VERSION = "research-evidence-bundle-v1"
MAX_PASSAGES = 5
MAX_QUOTE_CHARS = 1_200
MAX_FINDINGS = 20
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class VerifiedPassage(TypedDict):
    passage_id: str
    doc_id: str
    source_revision: str
    source_version_hash: str
    start: int
    end: int
    quote: str
    quote_sha256: str
    coverage: str


class CandidateFinding(TypedDict):
    text: str
    passage_ids: list[str]


class EvidenceBundle(TypedDict):
    schema_version: str
    run_id: str
    question: str
    corpus_hash: str
    index_version: str
    tool_protocol_version: str
    qwen_checkpoint: str
    qwen_status: Literal["evidence_found"]
    provenance_status: Literal["valid"]
    semantic_support_status: Literal["not_reviewed"]
    passages: list[VerifiedPassage]
    candidate_findings: list[CandidateFinding]
    unresolved_gaps: list[str]


def _text(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    if len(value) > limit:
        raise ValueError(f"{label} exceeds {limit} characters")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _source_revision(document: dict, manifest: dict) -> str:
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("source document has no versioned metadata")
    revision = (
        metadata.get("arxiv_id")
        or metadata.get("source_revision")
        or metadata.get("source_version")
    )
    if not isinstance(revision, str) or not revision:
        raise ValueError("source document has no pinned source revision")
    papers = manifest.get("papers")
    if isinstance(papers, list):
        matches = [
            paper
            for paper in papers
            if isinstance(paper, dict) and paper.get("doc_id") == document["doc_id"]
        ]
        if len(matches) != 1:
            raise ValueError(f"manifest does not identify {document['doc_id']} once")
        manifest_revision = (
            matches[0].get("arxiv_id")
            or matches[0].get("source_revision")
            or matches[0].get("source_version")
        )
        if manifest_revision is not None and manifest_revision != revision:
            raise ValueError(f"source revision differs from manifest for {document['doc_id']}")
    return revision


def _passage_id(doc_id: str, source_hash: str, start: int, end: int, quote_hash: str) -> str:
    identity = json.dumps(
        [doc_id, source_hash, start, end, quote_hash], ensure_ascii=False, separators=(",", ":")
    )
    return "p_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_evidence_bundle(
    *,
    snapshot: Path,
    run_id: str,
    question: str,
    corpus_hash: str,
    index_version: str,
    tool_protocol_version: str,
    qwen_checkpoint: str,
    candidate_spans: Sequence[Mapping[str, object]],
    returned_doc_ids: Collection[str],
    candidate_findings: Sequence[Mapping[str, object]] = (),
    unresolved_gaps: Sequence[str] = (),
) -> EvidenceBundle:
    """Verify exact source spans and return a JSON-serializable evidence bundle.

    ``returned_doc_ids`` must come from the host's tool-return ledger, not from
    Qwen's claimed trajectory. Each candidate span supplies a document ID,
    source revision, SHA-256 of canonical document text, character offsets, and
    exact quote. Findings refer to zero-based candidate ``span_indices``; the
    verifier replaces those with stable passage IDs. All model text remains
    semantically unreviewed even when provenance checks pass.
    """
    run_id = _text(run_id, "run_id", 200)
    question = _text(question, "question", 4_000)
    corpus_hash = _sha256(corpus_hash, "corpus_hash")
    index_version = _text(index_version, "index_version", 200)
    tool_protocol_version = _text(tool_protocol_version, "tool_protocol_version", 200)
    qwen_checkpoint = _text(qwen_checkpoint, "qwen_checkpoint", 500)
    if isinstance(returned_doc_ids, (str, bytes)) or not all(
        isinstance(doc_id, str) for doc_id in returned_doc_ids
    ):
        raise ValueError("returned_doc_ids must be a collection of document IDs")
    discovered = set(returned_doc_ids)
    if not isinstance(candidate_spans, Sequence) or isinstance(candidate_spans, (str, bytes)):
        raise ValueError("candidate_spans must be a sequence")
    if not 1 <= len(candidate_spans) <= MAX_PASSAGES:
        raise ValueError(f"candidate_spans must contain 1..{MAX_PASSAGES} spans")

    # load_snapshot checks the manifest against the current bytes on disk.
    manifest, documents = load_snapshot(Path(snapshot))
    if manifest.get("corpus_hash") != corpus_hash:
        raise ValueError("requested corpus hash differs from frozen snapshot")

    passages: list[VerifiedPassage] = []
    seen: set[tuple[str, int, int]] = set()
    for index, candidate in enumerate(candidate_spans):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidate span {index} must be an object")
        doc_id = _text(candidate.get("doc_id"), f"candidate span {index} doc_id", 200)
        if doc_id not in discovered:
            raise ValueError(f"candidate span {index} cites an undiscovered document: {doc_id}")
        document = documents.get(doc_id)
        if not isinstance(document, dict):
            raise ValueError(f"candidate span {index} document is absent: {doc_id}")
        revision = _source_revision(document, manifest)
        if candidate.get("source_revision") != revision:
            raise ValueError(f"candidate span {index} source revision mismatch: {doc_id}")
        source_text = document.get("text")
        if not isinstance(source_text, str):
            raise ValueError(f"source document has no text: {doc_id}")
        source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        if candidate.get("source_version_hash") != source_hash:
            raise ValueError(f"candidate span {index} source text hash mismatch: {doc_id}")
        start, end = candidate.get("start"), candidate.get("end")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(source_text)
        ):
            raise ValueError(f"candidate span {index} has invalid character offsets: {doc_id}")
        if (doc_id, start, end) in seen:
            raise ValueError(f"candidate span {index} duplicates an earlier span")
        seen.add((doc_id, start, end))
        quote = _text(candidate.get("quote"), f"candidate span {index} quote", MAX_QUOTE_CHARS)
        if source_text[start:end] != quote:
            raise ValueError(f"candidate span {index} quote does not match source: {doc_id}")
        quote_hash = hashlib.sha256(quote.encode("utf-8")).hexdigest()
        if "quote_sha256" in candidate and candidate["quote_sha256"] != quote_hash:
            raise ValueError(f"candidate span {index} quote hash mismatch: {doc_id}")
        metadata = document["metadata"]
        coverage = metadata.get("coverage")
        if not isinstance(coverage, str) or not coverage:
            raise ValueError(f"source document has no coverage metadata: {doc_id}")
        passages.append(
            {
                "passage_id": _passage_id(doc_id, source_hash, start, end, quote_hash),
                "doc_id": doc_id,
                "source_revision": revision,
                "source_version_hash": source_hash,
                "start": start,
                "end": end,
                "quote": quote,
                "quote_sha256": quote_hash,
                "coverage": coverage,
            }
        )

    if not isinstance(candidate_findings, Sequence) or isinstance(candidate_findings, (str, bytes)):
        raise ValueError("candidate_findings must be a sequence")
    if len(candidate_findings) > MAX_FINDINGS:
        raise ValueError(f"candidate_findings exceeds {MAX_FINDINGS} items")
    findings: list[CandidateFinding] = []
    for index, finding in enumerate(candidate_findings):
        if not isinstance(finding, Mapping):
            raise ValueError(f"candidate finding {index} must be an object")
        proposal = _text(finding.get("text"), f"candidate finding {index} text", 2_500)
        span_indices = finding.get("span_indices")
        if not isinstance(span_indices, list) or not span_indices:
            raise ValueError(f"candidate finding {index} needs span_indices")
        if any(type(i) is not int or not 0 <= i < len(passages) for i in span_indices):
            raise ValueError(f"candidate finding {index} references an unknown span")
        if len(set(span_indices)) != len(span_indices):
            raise ValueError(f"candidate finding {index} repeats a span")
        findings.append(
            {"text": proposal, "passage_ids": [passages[i]["passage_id"] for i in span_indices]}
        )

    if not isinstance(unresolved_gaps, Sequence) or isinstance(unresolved_gaps, (str, bytes)):
        raise ValueError("unresolved_gaps must be a sequence")
    if len(unresolved_gaps) > MAX_FINDINGS:
        raise ValueError(f"unresolved_gaps exceeds {MAX_FINDINGS} items")
    gaps = [_text(gap, "unresolved gap", 1_000) for gap in unresolved_gaps]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "question": question,
        "corpus_hash": corpus_hash,
        "index_version": index_version,
        "tool_protocol_version": tool_protocol_version,
        "qwen_checkpoint": qwen_checkpoint,
        "qwen_status": "evidence_found",
        "provenance_status": "valid",
        "semantic_support_status": "not_reviewed",
        "passages": passages,
        "candidate_findings": findings,
        "unresolved_gaps": gaps,
    }
