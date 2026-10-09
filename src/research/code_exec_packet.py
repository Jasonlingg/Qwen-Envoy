"""Bounded evidence handoff from an active code-execution evaluation row."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.eval.artifacts import content_hash
from src.research.explainer import PACKET_VERSION

MAX_EVIDENCE_SPANS = 5
MAX_QUOTE_CHARS = 1_200
MAX_QUESTION_CHARS = 4_000
MAX_ANSWER_CHARS = 2_500
_DOC_ID = re.compile(r"[A-Za-z0-9_.-]+\Z")


def _bounded_text(value: Any, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    if len(value) > limit:
        raise ValueError(f"{label} exceeds {limit} characters")
    return value


def _metadata_text(metadata: dict, key: str, limit: int = 500) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and len(value) <= limit else None


def _load_document(corpus_dir: Path, doc_id: str) -> dict:
    if not _DOC_ID.fullmatch(doc_id) or doc_id in {".", ".."}:
        raise ValueError(f"unsafe document ID: {doc_id!r}")
    path = corpus_dir / f"{doc_id}.json"
    if not path.is_file():
        raise ValueError(f"evidence document is absent from the frozen corpus: {doc_id}")
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get("doc_id") != doc_id:
        raise ValueError(f"corpus document ID mismatch: {doc_id}")
    if not isinstance(document.get("text"), str):
        raise ValueError(f"corpus document has no text: {doc_id}")
    return document


def build_code_exec_packet(row: dict, corpus_dir: Path) -> dict:
    """Materialize Qwen-cited offsets, without copying evaluator or gold fields.

    A real quote proves only that the bytes came from the snapshot. It does not
    establish that the quote supports Qwen's proposed answer.
    """
    if not isinstance(row, dict) or row.get("status") != "completed":
        raise ValueError("a completed code-execution transcript row is required")
    if not corpus_dir.is_dir():
        raise ValueError(f"frozen corpus directory does not exist: {corpus_dir}")

    question_id = _bounded_text(row.get("question_id"), "question_id", 200)
    question = _bounded_text(row.get("question"), "question", MAX_QUESTION_CHARS)
    proposal = _bounded_text(row.get("predicted_answer"), "predicted_answer", MAX_ANSWER_CHARS)
    predicted = row.get("predicted_evidence")
    if not isinstance(predicted, list) or not predicted:
        raise ValueError("completed row has no predicted evidence spans")

    evidence: list[dict] = []
    documents: dict[str, dict] = {}
    seen_spans: set[tuple[str, int, int]] = set()
    for index, span in enumerate(predicted[:MAX_EVIDENCE_SPANS], start=1):
        if not isinstance(span, dict):
            raise ValueError(f"evidence span {index} must be an object")
        doc_id, start, end = span.get("doc_id"), span.get("start"), span.get("end")
        if not isinstance(doc_id, str):
            raise ValueError(f"evidence span {index} has no document ID")
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"evidence span {index} needs integer character offsets")
        span_key = (doc_id, start, end)
        if span_key in seen_spans:
            raise ValueError(f"evidence span {index} duplicates an earlier span")
        seen_spans.add(span_key)
        if doc_id not in documents:
            documents[doc_id] = _load_document(corpus_dir, doc_id)
        document = documents[doc_id]
        text = document["text"]
        if not 0 <= start < end <= len(text):
            raise ValueError(f"evidence span {index} is outside {doc_id}")
        full_quote = text[start:end]
        if "quote" in span and span["quote"] != full_quote:
            raise ValueError(f"evidence span {index} quote does not match {doc_id}")

        bounded_end = min(end, start + MAX_QUOTE_CHARS)
        metadata = document.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"corpus document has invalid metadata: {doc_id}")
        evidence.append({
            "evidence_id": f"E{index}",
            "doc_id": doc_id,
            "start": start,
            "end": bounded_end,
            "quote": text[start:bounded_end],
            "quote_truncated": bounded_end != end,
            "source_url": _metadata_text(metadata, "source_url"),
            "coverage": _metadata_text(metadata, "coverage", 100),
            "source_kind": _metadata_text(metadata, "source_kind", 100)
            or ("paper" if metadata.get("arxiv_id") else "unknown"),
            "title": _metadata_text(document, "title", 300),
        })

    evidence_ids = [item["evidence_id"] for item in evidence]
    return {
        "schema_version": PACKET_VERSION,
        "retriever_protocol": "code-execution",
        "retriever_run_id": row.get("run_id") if isinstance(row.get("run_id"), str) else None,
        "question_id": question_id,
        "question": question,
        "corpus_hash": content_hash(corpus_dir),
        "candidate_claims": [{"text": proposal, "evidence_ids": evidence_ids}],
        "evidence": evidence,
        "omitted_predicted_spans": max(0, len(predicted) - MAX_EVIDENCE_SPANS),
        "warning": (
            "The Qwen answer is an untrusted proposal. Exact source spans prove provenance, "
            "not semantic support. At most five spans are included, and long quotes are truncated."
        ),
    }
