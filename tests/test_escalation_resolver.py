from __future__ import annotations

import pytest

from src.research.escalation import (
    HostResolutionError,
    build_evidence_packet,
    validate_host_resolution,
)


def test_packet_keeps_document_observations_but_not_verifier_messages() -> None:
    row = {
        "question_id": "q1",
        "question": "What was reported?",
        "status": "escalated",
        "escalation": {
            "candidate_action": "SUBMIT: unsure CITATIONS: []",
            "state": {"inspected_doc_ids": ["paper"]},
        },
        "trajectory": [
            {
                "step": 1,
                "action": 'print(search_within("paper", "metric"))',
                "observation": "the metric was 91%",
            },
            {
                "step": 2,
                "action": "SUBMIT: unsure CITATIONS: []",
                "observation": "[Evidence verifier — recovery 1/2]",
            },
        ],
    }
    packet = build_evidence_packet(row)
    assert packet["allowed_document_ids"] == ["paper"]
    assert len(packet["evidence_steps"]) == 1
    assert packet["evidence_steps"][0]["observation"] == "the metric was 91%"


def test_host_resolution_requires_exact_quote_and_observed_citation() -> None:
    payload = {
        "resolution": "answer",
        "answer": "91%",
        "citations": ["paper"],
        "evidence_quotes": [{"doc_id": "paper", "quote": "metric was 91%"}],
    }
    resolved = validate_host_resolution(
        payload,
        allowed_doc_ids=["paper"],
        documents={"paper": "The metric was 91% on the test set."},
    )
    assert resolved["evidence"] == [{
        "doc_id": "paper", "start": 4, "end": 18, "quote": "metric was 91%",
    }]

    payload["evidence_quotes"][0]["quote"] = "approximately 91%"
    with pytest.raises(HostResolutionError, match="not exact corpus text"):
        validate_host_resolution(
            payload,
            allowed_doc_ids=["paper"],
            documents={"paper": "The metric was 91% on the test set."},
        )


def test_unanswerable_resolution_has_no_fake_provenance() -> None:
    resolved = validate_host_resolution(
        {
            "resolution": "unanswerable",
            "answer": "Unanswerable",
            "citations": [],
            "evidence_quotes": [],
        },
        allowed_doc_ids=["paper"],
        documents={"paper": "text"},
    )
    assert resolved == {
        "resolution": "unanswerable",
        "answer": "Unanswerable",
        "citations": [],
        "evidence": [],
    }
