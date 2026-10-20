"""Gap reports are bounded host-parsed control actions, never model Python execution."""
import copy

import pytest

from src.product.reader_gaps import parse_gap_call, record_gap
from tests.test_assigned_reader import DOC_ID, INSPECT, SUBMIT, setup_reader, verify

GAP = ('report_gap(missing="I have not found larger-batch measurements.", '
       'follow_up_query="speculative decoding batch size throughput")')


def test_literal_gap_call_and_normal_code_are_distinguished():
    assert parse_gap_call(GAP)["missing"] == "I have not found larger-batch measurements."
    assert parse_gap_call('report_gap("missing", "query")') == {
        "missing": "missing", "follow_up_query": "query"}
    assert parse_gap_call(INSPECT) is None
    assert parse_gap_call('print("report_gap is a control action")') is None


@pytest.mark.parametrize("action", [
    'report_gap(missing=value, follow_up_query="query")',
    'report_gap(missing=__import__("os").system("false"), follow_up_query="query")',
    'report_gap("a", "b"); print("side effect")',
    'print(report_gap("a", "b"))',
    'if True:\n    report_gap("a", "b")',
    'report_gap(missing="a")',
    'report_gap("a", "b", "c")',
    'report_gap("a", missing="duplicate", follow_up_query="b")',
    'report_gap(missing="a", follow_up_query="b", doc_id="invented")',
    'report_gap(**{"missing":"a", "follow_up_query":"b"})',
    'report_gap(missing=None, follow_up_query="b")',
    'report_gap(missing=" ", follow_up_query="b")',
    f'report_gap(missing={"a" * 601!r}, follow_up_query="b")',
    f'report_gap(missing="a", follow_up_query={"b" * 301!r})',
])
def test_nonliteral_or_oversized_gap_call_is_rejected(action):
    with pytest.raises(ValueError):
        parse_gap_call(action)


def test_gap_report_limit_and_duplicate_does_not_add_records():
    result = {"research_gaps": []}
    with pytest.raises(ValueError, match="inspect source"):
        record_gap(result, {"missing": "a", "follow_up_query": "b"}, 1, 0)
    for i in range(3):
        record_gap(result, {"missing": str(i), "follow_up_query": "q"}, i + 2, 1)
    with pytest.raises(ValueError, match="already reported"):
        record_gap(result, {"missing": "0", "follow_up_query": "q"}, 5, 1)
    with pytest.raises(ValueError, match="at most three"):
        record_gap(result, {"missing": "another", "follow_up_query": "q"}, 6, 1)
    assert len(result["research_gaps"]) == 3


def test_gap_and_verified_findings_both_survive(tmp_path):
    reader, _, repl, snapshot = setup_reader(tmp_path, [INSPECT, GAP, SUBMIT], steps=3)
    packet = reader.investigate("Memory?", snapshot)
    compact = verify(reader, packet, snapshot)
    assert compact["status"] == "evidence_found" and compact["evidence"]
    assert compact["research_gaps"][0]["reported_at_step"] == 2
    assert compact["research_gaps"][0]["review_status"] == "unreviewed_model_report"
    assert compact["research_gaps"][0]["absence_proven"] is False
    assert compact["gap_report_error"] is None
    assert packet["document_tool_steps"] == packet["inspection_steps"] == 1
    assert packet["model_requests_attempted"] == 3
    assert [action for action, _ in repl.executed] == [INSPECT]


def test_premature_gap_is_rejected_then_reader_can_recover(tmp_path):
    reader, _, repl, snapshot = setup_reader(tmp_path, [GAP, INSPECT, GAP, SUBMIT])
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "evidence_found" and len(packet["research_gaps"]) == 1
    assert "inspect source text" in packet["trajectory"][0]["observation"]
    assert packet["research_gaps"][0]["reported_at_step"] == 3
    assert packet["model_requests_attempted"] == 4 and len(repl.executed) == 1


def test_invalid_report_is_not_executed_and_cannot_discard_later_answer(tmp_path):
    bad = 'report_gap(missing=__import__("os").system("false"), follow_up_query="q")'
    reader, _, repl, snapshot = setup_reader(tmp_path, [INSPECT, bad, SUBMIT], steps=3)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "evidence_found" and packet["research_gaps"] == []
    assert "literal strings" in packet["trajectory"][1]["observation"]
    assert [action for action, _ in repl.executed] == [INSPECT]


def test_gap_does_not_satisfy_two_document_action_abstention_gate(tmp_path):
    reader, _, _, snapshot = setup_reader(tmp_path, [INSPECT, GAP,
        'submit(answer="Unanswerable", citations=[], evidence=[])'], steps=3)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "incomplete" and packet["evidence"] == []
    assert packet["document_tool_steps"] == 1 and packet["research_gaps"]
    assert "two distinct successful" in packet["trajectory"][-1]["observation"]
    assert verify(reader, packet, snapshot)["research_gaps"]


def test_gap_can_accompany_bounded_abstention(tmp_path):
    reader, _, _, snapshot = setup_reader(tmp_path, [INSPECT, f'print(read("{DOC_ID}"))',
        GAP, 'submit(answer="Unanswerable", citations=[], evidence=[])'])
    packet = reader.investigate("Memory?", snapshot)
    compact = verify(reader, packet, snapshot)
    assert compact["status"] == "no_evidence" and compact["research_gaps"]
    assert compact["evidence"] == [] and "does not prove" in packet["warning"]


def test_reader_failure_after_gap_preserves_report_with_error_status(tmp_path):
    reader, _, _, snapshot = setup_reader(tmp_path, [INSPECT, GAP], steps=4)
    packet = reader.investigate("Memory?", snapshot)  # exhausted fake policy
    assert packet["status"] == "error" and packet["inspection_steps"] == 1
    compact = verify(reader, packet, snapshot)
    assert compact["research_gaps"] and compact["evidence"] == []


@pytest.mark.parametrize("bad", [None, {}, [None], [{"missing": "extra"}], [0] * 4])
def test_malformed_gap_packet_does_not_discard_verified_evidence(tmp_path, bad):
    reader, _, _, snapshot = setup_reader(tmp_path, [INSPECT, GAP, SUBMIT], steps=3)
    packet = copy.deepcopy(reader.investigate("Memory?", snapshot))
    packet["research_gaps"] = bad
    compact = verify(reader, packet, snapshot)
    assert compact["evidence"] and compact["status"] == "evidence_found"
    assert compact["research_gaps"] == [] and compact["gap_report_error"]


def test_forged_gap_step_is_excluded_without_losing_quotes(tmp_path):
    reader, _, _, snapshot = setup_reader(tmp_path, [INSPECT, GAP, SUBMIT], steps=3)
    packet = reader.investigate("Memory?", snapshot)
    packet["research_gaps"][0]["reported_at_step"] = 1
    compact = verify(reader, packet, snapshot)
    assert compact["research_gaps"] == [] and compact["evidence"]


def test_inspection_later_in_trace_cannot_justify_earlier_gap(tmp_path):
    reader, _, _, snapshot = setup_reader(tmp_path, [INSPECT, GAP, SUBMIT], steps=3)
    packet = reader.investigate("Memory?", snapshot)
    packet["trajectory"][0].pop("source_inspection")
    packet["trajectory"][2]["source_inspection"] = True
    compact = verify(reader, packet, snapshot)
    assert compact["research_gaps"] == [] and compact["gap_report_error"]
    assert compact["evidence"]
