"""Keep shared provenance boundaries covered after retiring the chat product."""

import copy

import pytest

from src.product.chat import _verified_review
from src.product.memory import freeze_vault
from src.research.agent import load_snapshot


@pytest.fixture
def source_packet(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Finding.md").write_text(
        "---\nrecord_id: original-record\nkind: result\n---\n"
        "# Original finding\n\n" + "Measured source evidence. " * 80
    )
    snapshot = tmp_path / "snapshot"
    manifest = freeze_vault(vault, snapshot)
    _, docs = load_snapshot(snapshot)
    doc_id, doc = next(iter(docs.items()))
    start = doc["text"].index("Measured")
    packet = {
        "status": "evidence_found", "corpus_hash": manifest["corpus_hash"],
        "evidence": [{"evidence_id": "E1", "doc_id": doc_id, "start": start,
                      "end": len(doc["text"]), "quote": doc["text"][start:],
                      "title": "invented", "record": {"record_id": "invented"}}],
    }
    return snapshot, packet


def test_review_replaces_supplied_metadata_and_bounds_the_exact_source(source_packet):
    snapshot, packet = source_packet
    result = _verified_review(snapshot, packet, "What was measured?")
    item = result["evidence"][0]
    assert item["title"] != "invented"
    assert item["source_path"] == "Finding.md"
    assert item["record"]["record_id"] == "original-record"
    assert item["quote"] == packet["evidence"][0]["quote"][:1200]
    assert item["end"] == item["start"] + 1200
    assert item["truncated_for_model"] is True
    assert "not semantic support" in result["warning"]


@pytest.mark.parametrize("mutation", ["corpus", "quote", "offset", "duplicate", "missing",
                                     "contradictory_status"])
def test_review_rejects_unbound_or_forged_evidence(source_packet, mutation):
    snapshot, original = source_packet
    packet = copy.deepcopy(original)
    if mutation == "corpus":
        packet["corpus_hash"] = "0" * 64
    elif mutation == "quote":
        packet["evidence"][0]["quote"] = "forged passage"
    elif mutation == "offset":
        packet["evidence"][0]["start"] = True
    elif mutation == "duplicate":
        packet["evidence"] *= 2
    elif mutation == "missing":
        packet["evidence"] = []
    else:
        packet["status"] = "no_evidence"
    with pytest.raises(ValueError):
        _verified_review(snapshot, packet, "What was measured?")
