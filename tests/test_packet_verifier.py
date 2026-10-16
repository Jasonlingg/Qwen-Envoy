"""One verifier for v1 and v2 investigation packets."""

import hashlib

import pytest

from src.env.tools import SEARCH_PROTOCOL_VERSION
from src.env.tools_v2 import HARNESS_V2_SHA256
from src.product.packet_verifier import TOOL_PREAMBLE_SHA256, verify_investigation_packet

PROMPT = hashlib.sha256(b"prompt").hexdigest()
TEXT = "A verifier checks citations exactly."
DOCS = {"d1": {"doc_id": "d1", "title": "T", "text": TEXT,
               "metadata": {"arxiv_id": "2609.00001v1", "source_url": "u", "coverage": "c"}}}
MANIFEST = {"corpus_hash": "h"}


def _packet(**overrides):
    quote = "checks citations"
    start = TEXT.index(quote)
    packet = {
        "corpus_hash": "h", "source_domain": "public_papers", "system_prompt_sha256": PROMPT,
        "tool_search_version": SEARCH_PROTOCOL_VERSION,
        "tool_preamble_sha256": TOOL_PREAMBLE_SHA256, "status": "evidence_found",
        "evidence": [{"doc_id": "d1", "start": start, "end": start + len(quote),
                      "quote": quote}],
        "candidate_claims": [{"text": "It checks citations."}],
        "trajectory": [], "model_requests_attempted": 3,
    }
    packet.update(overrides)
    return packet


def _verify(packet):
    return verify_investigation_packet(packet, MANIFEST, DOCS,
                                       allowed_prompt_sha256s=frozenset({PROMPT}))


def test_v1_packet_without_harness_field_is_accepted():
    result = _verify(_packet())
    assert result["harness_version"] == "v1"
    assert result["evidence"][0]["quote"] == "checks citations"
    assert result["prompt_hash"] == PROMPT


def test_v2_packet_needs_the_current_layer_hash():
    assert _verify(_packet(harness_version="v2", harness_shim_sha256=HARNESS_V2_SHA256))
    with pytest.raises(ValueError, match="harness"):
        _verify(_packet(harness_version="v2", harness_shim_sha256="0" * 64))


def test_v2_allows_retry_requests_beyond_twenty():
    assert _verify(_packet(harness_version="v2", harness_shim_sha256=HARNESS_V2_SHA256,
                           model_requests_attempted=45))
    with pytest.raises(ValueError, match="request count"):
        _verify(_packet(model_requests_attempted=45))


def test_unlisted_prompt_and_tampered_quote_are_rejected():
    with pytest.raises(ValueError, match="prompt"):
        _verify(_packet(system_prompt_sha256="0" * 64))
    tampered = _packet()
    tampered["evidence"][0]["quote"] = "checks citationz"
    with pytest.raises(ValueError, match="exact bounded paper span"):
        _verify(tampered)
