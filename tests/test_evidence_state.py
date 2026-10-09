from __future__ import annotations

from src.env.evidence_state import (
    EvidenceState,
    format_feedback,
    is_research_abstention,
    parse_document_calls,
)

DOC_LENGTHS = {"paper": 1_000, "other": 500}


def test_parse_document_calls_does_not_execute_code() -> None:
    calls = parse_document_calls(
        'hits = search_within("paper", "reported metric", top_k=8)\nprint(hits)'
    )
    assert [(call.name, call.doc_id) for call in calls] == [("search_within", "paper")]


def test_duplicate_detection_allows_a_changed_query() -> None:
    state = EvidenceState(DOC_LENGTHS)
    first = 'print(search_within("paper", "evaluation results"))'
    state.observe(first, "[{'doc_id': 'paper', 'text': 'result'}]")

    assert state.duplicate_reason(first) is not None
    assert state.duplicate_reason(
        'print(search_within("paper", "reported metric score"))'
    ) is None


def test_premature_abstention_needs_two_successful_document_actions() -> None:
    state = EvidenceState(DOC_LENGTHS)
    state.observe(
        'print(search_within("paper", "evaluation"))',
        "[{'doc_id': 'paper', 'text': 'overview'}]",
    )
    reasons = state.submission_reasons(
        answer="Unanswerable", citations=[], evidence=[], require_evidence=True
    )
    assert any("premature" in reason for reason in reasons)

    state.observe(
        'print(search_within("paper", "metric score"))',
        "[{'doc_id': 'paper', 'text': 'still absent'}]",
    )
    assert state.submission_reasons(
        answer="Unanswerable", citations=[], evidence=[], require_evidence=True
    ) == []
    assert is_research_abstention("Unanswerable")
    assert is_research_abstention("The inspected passages do not mention this result.")
    assert is_research_abstention("I cannot confirm the exact number from these passages.")
    assert not is_research_abstention("No, they do not use active learning.")


def test_grounding_checks_use_observed_docs_and_valid_offsets() -> None:
    state = EvidenceState(DOC_LENGTHS)
    state.observe(
        'print(search_within("paper", "metric"))',
        "[{'doc_id': 'paper', 'text': 'accuracy was 91%'}]",
    )
    valid = [{"doc_id": "paper", "start": 10, "end": 30}]
    assert state.submission_reasons(
        answer="91%", citations=["paper"], evidence=valid, require_evidence=True
    ) == []

    reasons = state.submission_reasons(
        answer="91%",
        citations=["other"],
        evidence=[{"doc_id": "other", "start": 0, "end": 999}],
        require_evidence=True,
    )
    assert any("not observed" in reason for reason in reasons)
    assert any("invalid offsets" in reason for reason in reasons)


def test_recovery_feedback_gives_concrete_code_options() -> None:
    state = EvidenceState(DOC_LENGTHS)
    state.observe(
        'print(search_within("paper", "evaluation"))',
        "[{'doc_id': 'paper', 'text': 'overview'}]",
    )
    feedback = format_feedback(
        reasons=["The refusal is premature."],
        state=state,
        feedback_number=1,
        feedback_budget=2,
        kind="submission",
    )
    assert "one DIFFERENT Python investigation action" in feedback
    assert "search_within('paper', '<different keywords>', top_k=8)" in feedback
    assert "scan('paper'" in feedback
    assert "read('paper')" in feedback
