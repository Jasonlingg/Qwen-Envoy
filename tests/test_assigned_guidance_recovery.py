"""Deterministic controller checks, not a test of model answer quality."""
import copy

import pytest

from src.product.assigned_guidance import FOCUSED, action_recovery
from src.product.qwen_investigator import QwenInvestigator
from tests.test_assigned_reader import DOC_ID, INSPECT, SUBMIT, setup_reader, verify
from tests.test_reader_gaps import GAP

ABSTAIN = 'submit(answer="Unanswerable", citations=[], evidence=[])'
SECOND = f'print(read("{DOC_ID}"))'


def candidate(tmp_path, actions, steps=6):
    base, policy, repl, snapshot = setup_reader(tmp_path, actions, steps=steps)
    reader = QwenInvestigator(base.policy_factory, base.model_identity, max_steps=steps,
        repl_factory=base.repl_factory, source_domain="public_papers", harness_version="v2",
        reader_mode="assigned", reader_guidance=FOCUSED)
    return reader, policy, repl, snapshot


def test_rejected_assignment_did_not_execute_and_has_exact_safe_retry(tmp_path):
    bad = f'hits = search_within("{DOC_ID}", "memory")'
    reader, policy, repl, snapshot = candidate(tmp_path, [bad, INSPECT, SUBMIT], steps=3)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "evidence_found"
    assert [a for a, _ in repl.executed] == [INSPECT]
    feedback = policy.observations[1]
    assert "Nothing ran; no source text was inspected" in feedback
    assert f"print(search_within('{DOC_ID}', 'memory'))" in feedback
    assert "[2 steps left.]" in feedback
    assert packet["inspection_steps"] == packet["document_tool_steps"] == 1


@pytest.mark.parametrize("action", [
    'hits = search_within(__import__("os").getcwd(), "q")',
    'hits = read("id"); __import__("os").system("false")',
    'read = search_within("id", "q")',
])
def test_unsafe_code_is_not_echoed_as_a_suggested_retry(action):
    message = action_recovery(action, "invalid displayed call", 2)
    assert "__import__" not in message
    assert 'print(search_within("assigned_id", "question keywords"))' in message


def test_candidate_cannot_abstain_without_a_gap_and_can_recover(tmp_path):
    reader, policy, repl, snapshot = candidate(tmp_path,
        [INSPECT, SECOND, ABSTAIN, GAP, ABSTAIN], steps=5)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "no_evidence"
    assert "gap report first" in policy.observations[3]
    assert packet["model_requests_attempted"] == 5
    assert packet["document_tool_steps"] == packet["inspection_steps"] == 2
    assert [a for a, _ in repl.executed] == [INSPECT, SECOND]
    assert len(verify(reader, packet, snapshot)["research_gaps"]) == 1


def test_missing_gap_on_last_action_is_incomplete_not_success(tmp_path):
    reader, _, _, snapshot = candidate(tmp_path, [INSPECT, SECOND, ABSTAIN], steps=3)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "incomplete" and not packet["research_gaps"]
    assert "[0 steps left.]" in packet["trajectory"][-1]["observation"]
    assert verify(reader, packet, snapshot)["status"] == "incomplete"


def test_baseline_keeps_its_existing_abstention_behavior(tmp_path):
    reader, _, _, snapshot = setup_reader(tmp_path, [INSPECT, SECOND, ABSTAIN], steps=3)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "no_evidence" and packet["research_gaps"] == []
    assert verify(reader, packet, snapshot)["status"] == "no_evidence"


def test_gap_still_cannot_replace_second_source_action(tmp_path):
    reader, _, _, snapshot = candidate(tmp_path, [INSPECT, GAP, ABSTAIN], steps=3)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "incomplete" and packet["document_tool_steps"] == 1
    assert "two distinct successful" in packet["trajectory"][-1]["observation"]


def test_verifier_rejects_forged_candidate_abstention_with_no_valid_gap(tmp_path):
    reader, _, _, snapshot = candidate(tmp_path, [INSPECT, SECOND, GAP, ABSTAIN], steps=4)
    packet = reader.investigate("Memory?", snapshot)
    packet["research_gaps"] = []
    with pytest.raises(ValueError, match="requires a verified gap report"):
        verify(reader, packet, snapshot)


def test_bad_ancillary_gap_still_cannot_discard_independent_quotes(tmp_path):
    reader, _, _, snapshot = candidate(tmp_path, [INSPECT, SUBMIT], steps=2)
    packet = copy.deepcopy(reader.investigate("Memory?", snapshot))
    packet["research_gaps"] = None
    checked = verify(reader, packet, snapshot)
    assert checked["evidence"] and checked["gap_report_error"]


def test_inspection_feedback_requests_focused_context_not_an_answer(tmp_path):
    reader, policy, _, snapshot = candidate(tmp_path, [INSPECT, SUBMIT], steps=2)
    reader.investigate("Memory?", snapshot)
    assert "short passage around its returned offset" in policy.observations[1]
    assert "not semantic support" in policy.observations[1]
