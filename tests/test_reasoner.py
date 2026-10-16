import json

import pytest

from src.product.reasoner import Reasoner
from src.research.escalation import HostResolutionError

DOC = "arxiv_2609_12345v2"
QUOTE = "Eviction saves 40 percent of memory at equal accuracy."
TEXT = "x" * 100 + QUOTE + " Other text that is not in the evidence."
MERGED = {
    "question": "How much memory does eviction save?",
    "evidence": [{"evidence_id": "E1", "doc_id": DOC, "arxiv_id": "2609.12345v2",
                  "start": 100, "end": 100 + len(QUOTE), "quote": QUOTE}],
    "candidate_answers": [{"arxiv_id": "2609.12345v2", "text": "About 40%."}],
}


def _reasoner(reply):
    calls = []

    def transport(url, payload, headers, timeout):
        calls.append((url, payload, headers))
        content = json.dumps(reply) if isinstance(reply, dict) else reply
        return {"choices": [{"message": {"content": content}}]}

    return Reasoner("https://api.tokenfactory.nebius.com/v1", "nvidia/test", "key",
                    transport=transport), calls


def _answer(quote):
    return {"resolution": "answer", "answer": "About 40 percent.", "citations": [DOC],
            "evidence_quotes": [{"doc_id": DOC, "quote": quote}]}


def test_grounded_answer_gets_offsets_inside_the_packet():
    reasoner, calls = _reasoner(_answer("saves 40 percent of memory"))
    out = reasoner.synthesize(MERGED, {DOC: TEXT})
    item = out["evidence"][0]
    assert TEXT[item["start"]:item["end"]] == "saves 40 percent of memory"
    assert item["evidence_id"] == "E1" and out["model"] == "nvidia/test"
    url, payload, headers = calls[0]
    assert url.endswith("/chat/completions") and headers["Authorization"] == "Bearer key"
    assert "Other text" not in payload["messages"][1]["content"]


def test_quote_outside_verified_evidence_is_rejected():
    reasoner, _ = _reasoner(_answer("Other text that is not in the evidence."))
    with pytest.raises(HostResolutionError, match="outside the verified evidence"):
        reasoner.synthesize(MERGED, {DOC: TEXT})


def test_invented_quote_and_bad_json_are_rejected():
    with pytest.raises(HostResolutionError):
        _reasoner(_answer("Eviction saves 90 percent"))[0].synthesize(MERGED, {DOC: TEXT})
    with pytest.raises(HostResolutionError):
        _reasoner("not json")[0].synthesize(MERGED, {DOC: TEXT})


def test_no_evidence_means_no_model_call():
    reasoner, calls = _reasoner(_answer(QUOTE))
    with pytest.raises(HostResolutionError):
        reasoner.synthesize({**MERGED, "evidence": []}, {})
    assert calls == []


def test_duplicate_source_text_uses_the_verified_occurrence():
    text = QUOTE + 'x' * (100 - len(QUOTE)) + QUOTE
    out = _reasoner(_answer(QUOTE))[0].synthesize(MERGED, {DOC: text})
    assert out['evidence'][0]['start'] == 100
    assert out['semantic_support'] == 'not_reviewed'


def test_malformed_resolution_is_rejected():
    bad = {**_answer(QUOTE), 'resolution': []}
    with pytest.raises(HostResolutionError):
        _reasoner(bad)[0].synthesize(MERGED, {DOC: TEXT})
