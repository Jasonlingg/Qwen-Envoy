"""The active pipeline accepts public papers, not staged or personal notes."""

from __future__ import annotations

import json

import pytest

from src.eval.artifacts import content_hash
from src.harness.source import load_public_paper_snapshot


def test_public_paper_snapshot_requires_pinned_source_identity(tmp_path):
    snapshot = tmp_path / "snapshot"
    corpus = snapshot / "corpus"
    corpus.mkdir(parents=True)
    doc_id = "arxiv_2609_12345v1"
    paper_id = "2609.12345v1"
    doc = {
        "doc_id": doc_id,
        "title": "Example paper",
        "text": "A source passage.",
        "metadata": {
            "arxiv_id": paper_id,
            "source_url": f"https://arxiv.org/abs/{paper_id}",
            "coverage": "abstract_only",
        },
    }
    source = corpus / f"{doc_id}.json"
    source.write_text(json.dumps(doc), encoding="utf-8")
    manifest = {
        "schema_version": "research-snapshot-v1",
        "status": "complete",
        "corpus_hash": content_hash(corpus),
        "papers": [{"doc_id": doc_id, "arxiv_id": paper_id}],
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _, loaded = load_public_paper_snapshot(snapshot)
    assert loaded[doc_id]["text"] == "A source passage."

    doc["metadata"]["source_url"] = "https://example.com/swapped"
    source.write_text(json.dumps(doc), encoding="utf-8")
    manifest["corpus_hash"] = content_hash(corpus)
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        load_public_paper_snapshot(snapshot)


def test_vault_snapshot_is_not_a_public_paper(tmp_path):
    snapshot = tmp_path / "snapshot"
    corpus = snapshot / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "vault_note.json").write_text(
        json.dumps({"doc_id": "vault_note", "text": "A personal note.", "metadata": {}})
    )
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "research-snapshot-v1",
                "status": "complete",
                "corpus_hash": content_hash(corpus),
                "papers": [{"doc_id": "vault_note"}],
            }
        )
    )
    with pytest.raises(ValueError, match="identity"):
        load_public_paper_snapshot(snapshot)
