"""Audit counterexamples; no claims about model or judge quality."""

from pathlib import Path

import pytest

from scripts.audit_reader_f1 import audit


@pytest.fixture(scope="module")
def report():
    return audit(Path("data/rl/reader-facts-v1"))


def test_tiny_quote_exposes_overlap_mapping_without_gold_only_mapping(report):
    by_kind = report["by_kind"]
    tiny = by_kind["real_but_insufficient_quote"]
    assert tiny["overlap_mapped_paragraph_f1_not_official"] == 1
    assert tiny["literal_quotes_as_paragraphs_f1"] == 0
    assert tiny["character_coverage_f1_not_official"] < .01
    assert by_kind["unrelated_first_character"]["overlap_mapped_paragraph_f1_not_official"] == 0


def test_reference_and_wrong_answer_share_evidence_score(report):
    by_kind = report["by_kind"]
    assert by_kind["reference"]["literal_quotes_as_paragraphs_f1"] == 1
    assert by_kind["wrong_answer"]["literal_quotes_as_paragraphs_f1"] == 1
    assert by_kind["wrong_answer"]["upstream_answer_f1"] < 1


def test_scope_and_roles_are_disclosed_without_dropping_unavailable_annotations(report):
    assert len(report["rows"]) == 48
    assert {r["role"] for r in report["rows"]} == {"reward_validation"}
    assert any(a["fully_available_paragraphs"] < a["total_paragraphs"]
               for r in report["rows"] for a in r["reference_scope"])
    assert report["training_ready"] is False
    assert report["active_reward_changed"] is False
    assert report["independent_human_review"] is None
