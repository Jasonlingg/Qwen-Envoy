"""Integrity checks for the CPU-only paper evidence handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.artifacts import content_hash
from src.harness.evidence import build_evidence_bundle


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def frozen_papers(tmp_path: Path) -> tuple[Path, dict[str, dict]]:
    snapshot = tmp_path / "snapshot"
    corpus = snapshot / "corpus"
    corpus.mkdir(parents=True)
    documents = {
        "arxiv_2609_11111v1": {
            "doc_id": "arxiv_2609_11111v1",
            "title": "Finding evidence",
            "text": "A retrieval worker searches papers. A verifier checks exact quotes.",
            "metadata": {"arxiv_id": "2609.11111v1", "coverage": "html_paragraphs"},
        },
        "arxiv_2609_22222v2": {
            "doc_id": "arxiv_2609_22222v2",
            "title": "Separate result",
            "text": "Different source text.",
            "metadata": {"arxiv_id": "2609.22222v2", "coverage": "abstract_only"},
        },
    }
    for document in documents.values():
        _write_json(corpus / f"{document['doc_id']}.json", document)
    manifest = {
        "schema_version": "research-snapshot-v1",
        "corpus_hash": content_hash(corpus),
        "papers": [
            {"doc_id": document["doc_id"], "arxiv_id": document["metadata"]["arxiv_id"]}
            for document in documents.values()
        ],
    }
    _write_json(snapshot / "manifest.json", manifest)
    return snapshot, documents


def _inputs(snapshot: Path, documents: dict[str, dict]) -> dict:
    document = documents["arxiv_2609_11111v1"]
    quote = "verifier checks exact quotes"
    start = document["text"].index(quote)
    return {
        "snapshot": snapshot,
        "run_id": "paper-run-1",
        "question": "How are quotes checked?",
        "corpus_hash": json.loads((snapshot / "manifest.json").read_text())["corpus_hash"],
        "index_version": "paper-index-v1",
        "tool_protocol_version": "research-tools-v4",
        "qwen_checkpoint": "Qwen3-8B/base@revision",
        "candidate_spans": [
            {
                "doc_id": document["doc_id"],
                "source_revision": document["metadata"]["arxiv_id"],
                "source_version_hash": hashlib.sha256(document["text"].encode()).hexdigest(),
                "start": start,
                "end": start + len(quote),
                "quote": quote,
                "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
            }
        ],
        "returned_doc_ids": {document["doc_id"]},
        "candidate_findings": [{"text": "The verifier checks exact quotes.", "span_indices": [0]}],
        "unresolved_gaps": ["This does not establish semantic support."],
    }


def test_verified_bundle_is_json_ready_and_does_not_claim_semantic_support(frozen_papers):
    snapshot, documents = frozen_papers
    inputs = _inputs(snapshot, documents)
    bundle = build_evidence_bundle(**inputs)
    assert json.loads(json.dumps(bundle)) == bundle
    assert bundle["schema_version"] == "research-evidence-bundle-v1"
    assert bundle["qwen_status"] == "evidence_found"
    assert bundle["provenance_status"] == "valid"
    assert bundle["semantic_support_status"] == "not_reviewed"
    passage = bundle["passages"][0]
    assert (
        passage["quote"] == documents[passage["doc_id"]]["text"][passage["start"] : passage["end"]]
    )
    assert bundle["candidate_findings"][0]["passage_ids"] == [passage["passage_id"]]

    inputs["run_id"] = "paper-run-2"
    assert build_evidence_bundle(**inputs)["passages"][0]["passage_id"] == passage["passage_id"]


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("doc_id", "arxiv_2609_22222v2", "undiscovered"),
        ("source_revision", "2609.11111v9", "source revision mismatch"),
        ("source_version_hash", "0" * 64, "source text hash mismatch"),
        ("start", -1, "invalid character offsets"),
        ("end", True, "invalid character offsets"),
        ("quote", "verifier checks invented quotes", "quote does not match"),
        ("quote_sha256", "0" * 64, "quote hash mismatch"),
    ],
)
def test_candidate_span_must_match_discovered_pinned_source(
    frozen_papers,
    field,
    bad_value,
    message,
):
    snapshot, documents = frozen_papers
    inputs = _inputs(snapshot, documents)
    inputs["candidate_spans"][0][field] = bad_value
    with pytest.raises(ValueError, match=message):
        build_evidence_bundle(**inputs)


def test_corpus_hash_and_manifest_revision_must_match(frozen_papers):
    snapshot, documents = frozen_papers
    inputs = _inputs(snapshot, documents)
    inputs["corpus_hash"] = "0" * 64
    with pytest.raises(ValueError, match="requested corpus hash differs"):
        build_evidence_bundle(**inputs)

    inputs = _inputs(snapshot, documents)
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["papers"][0]["arxiv_id"] = "2609.11111v9"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="source revision differs from manifest"):
        build_evidence_bundle(**inputs)


def test_changed_corpus_bytes_are_rejected_before_quote_verification(frozen_papers):
    snapshot, documents = frozen_papers
    inputs = _inputs(snapshot, documents)
    document_path = snapshot / "corpus" / "arxiv_2609_11111v1.json"
    document_path.write_text(document_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Snapshot corpus hash mismatch"):
        build_evidence_bundle(**inputs)


def test_passage_id_depends_on_source_span_not_other_corpus_documents(frozen_papers):
    snapshot, documents = frozen_papers
    inputs = _inputs(snapshot, documents)
    passage_id = build_evidence_bundle(**inputs)["passages"][0]["passage_id"]
    other = documents["arxiv_2609_22222v2"]
    other["text"] += " Added text in another source."
    _write_json(snapshot / "corpus" / f"{other['doc_id']}.json", other)
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["corpus_hash"] = content_hash(snapshot / "corpus")
    _write_json(manifest_path, manifest)
    inputs["corpus_hash"] = manifest["corpus_hash"]
    assert build_evidence_bundle(**inputs)["passages"][0]["passage_id"] == passage_id


def test_finding_indices_and_duplicate_spans_fail_closed(frozen_papers):
    snapshot, documents = frozen_papers
    inputs = _inputs(snapshot, documents)
    inputs["candidate_findings"][0]["span_indices"] = [1]
    with pytest.raises(ValueError, match="unknown span"):
        build_evidence_bundle(**inputs)

    inputs = _inputs(snapshot, documents)
    inputs["candidate_spans"].append(dict(inputs["candidate_spans"][0]))
    with pytest.raises(ValueError, match="duplicates an earlier span"):
        build_evidence_bundle(**inputs)
