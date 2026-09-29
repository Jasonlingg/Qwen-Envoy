"""The drafter may only put cited claims into an unreviewed note."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.harness.draft import DRAFT_VERSION, render_question_draft


@pytest.fixture
def bundle():
    return {
        "schema_version": "research-evidence-bundle-v1",
        "run_id": "run-1",
        "question": "What did the paper report?",
        "corpus_hash": "a" * 64,
        "provenance_status": "valid",
        "semantic_support_status": "not_reviewed",
        "passages": [
            {
                "passage_id": "p_abc123",
                "doc_id": "arxiv_2609_11111v1",
                "start": 4,
                "end": 24,
                "quote": "We report a result.",
            }
        ],
        "candidate_findings": [{"text": "Untrusted Qwen proposal"}],
    }


@pytest.fixture
def draft():
    return {
        "schema_version": DRAFT_VERSION,
        "claims": [{"text": "The paper reports a result.", "passage_ids": ["p_abc123"]}],
        "open_questions": [{"text": "Does it replicate?", "passage_ids": ["p_abc123"]}],
        "proposed_applications": [
            {
                "text": "Try the method on a local benchmark.",
                "passage_ids": ["p_abc123"],
            }
        ],
    }


def test_rendered_note_is_unreviewed_and_claims_link_to_exact_sources(bundle, draft):
    note = render_question_draft(bundle, draft)
    assert "review_status: agent_authored_draft" in note
    assert "semantic_support_status: not_reviewed" in note
    assert "The paper reports a result. [E1](#e1)" in note
    assert "### E1" in note
    assert "We report a result." in note
    assert "Untrusted Qwen proposal" not in note
    assert "Question: Does it replicate? [E1](#e1)" in note
    assert "Proposed application (not verified): Try the method" in note


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: item["claims"][0].update(passage_ids=[]),
        lambda item: item["claims"][0].update(passage_ids=["p_unknown"]),
        lambda item: item.update(unconstrained_summary="This is proven."),
        lambda item: item["claims"][0].update(unsupported=True),
        lambda item: item["open_questions"][0].update(passage_ids=[]),
        lambda item: item["proposed_applications"][0].update(passage_ids=["p_unknown"]),
    ],
)
def test_uncited_unknown_or_extra_model_prose_is_rejected(bundle, draft, mutate):
    broken = deepcopy(draft)
    mutate(broken)
    with pytest.raises(ValueError):
        render_question_draft(bundle, broken)


def test_claim_cannot_inject_heading_or_link(bundle, draft):
    draft["claims"][0]["text"] = "Result\n# Approved\n[link](https://example.com)"
    note = render_question_draft(bundle, draft)
    assert "\n# Approved" not in note
    assert "\\# Approved" in note
    assert "\\[link\\]" in note


def test_semantic_review_flag_from_model_cannot_promote_note(bundle, draft):
    bundle["semantic_support_status"] = "human_supported"
    with pytest.raises(ValueError, match="unreviewed semantics"):
        render_question_draft(bundle, draft)
