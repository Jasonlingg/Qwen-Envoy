"""Pinned public-paper snapshot validation for the active harness task."""

from __future__ import annotations

import re
from pathlib import Path

from src.eval.artifacts import content_hash
from src.research.agent import load_snapshot


def load_public_paper_snapshot(snapshot: Path) -> tuple[dict, dict[str, dict]]:
    """Reject incomplete or vault-note snapshots before a paper run starts."""
    manifest, docs = load_snapshot(snapshot)
    papers = manifest.get("papers")
    if (
        manifest.get("schema_version") != "research-snapshot-v1"
        or manifest.get("status") != "complete"
        or not isinstance(papers, list)
        or not papers
        or len(papers) != len(docs)
        or any(not isinstance(item, dict) for item in papers)
    ):
        raise ValueError("expected a complete frozen public-paper snapshot")
    if any(not isinstance(item.get("doc_id"), str) for item in papers):
        raise ValueError("paper manifest has an invalid document ID")
    rows = {item["doc_id"]: item for item in papers}
    if len(rows) != len(papers) or set(rows) != set(docs):
        raise ValueError("paper manifest and corpus document IDs differ")
    for doc_id, doc in docs.items():
        row = rows[doc_id]
        metadata = doc.get("metadata")
        arxiv_id = row.get("arxiv_id")
        if (
            not isinstance(arxiv_id, str)
            or not re.fullmatch(r"\d{4}\.\d{4,5}v\d+", arxiv_id)
            or doc_id != "arxiv_" + arxiv_id.replace(".", "_")
            or not isinstance(metadata, dict)
            or metadata.get("arxiv_id") != arxiv_id
            or metadata.get("source_url") != f"https://arxiv.org/abs/{arxiv_id}"
            or row.get("source_url", metadata["source_url"]) != metadata["source_url"]
            or not isinstance(metadata.get("coverage"), str)
            or not metadata["coverage"]
            or row.get("coverage", metadata["coverage"]) != metadata["coverage"]
        ):
            raise ValueError(f"invalid pinned public-paper identity: {doc_id}")
        source_file = snapshot / "corpus" / f"{doc_id}.json"
        if not source_file.is_file() or (
            row.get("sha256") is not None and row["sha256"] != content_hash(source_file)
        ):
            raise ValueError(f"paper source hash mismatch: {doc_id}")
    return manifest, docs
