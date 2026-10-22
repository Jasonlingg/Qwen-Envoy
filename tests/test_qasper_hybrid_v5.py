"""The live hybrid test may change ranges, never the reader/source or old allowance."""
import copy
import hashlib
import json

import pytest

from scripts.try_qasper_hybrid_v5 import replace_scope_only, selected_windows
from src.product.passage_scope import PASSAGES_V2, make_scope


def fixture():
    document = {"text": "source " * 2000}
    case = {"case_id": "q1", "doc_id": "qasper_p1", "question": "Which facts?"}
    assignment = {"method": "hybrid", "split": "test_candidate", "question_id": "q1",
        "doc_id": "qasper_p1", "question": "Which facts?", "status": "selected",
        "source_sha256": hashlib.sha256(document["text"].encode()).hexdigest(),
        "counts_as_source_inspection": False,
        "windows": [{"start": 800, "end": 2400}, {"start": 3200, "end": 4800}]}
    return document, case, assignment


def test_assignment_uses_frozen_source_query_and_exact_two_window_budget():
    doc, case, assignment = fixture()
    original = copy.deepcopy(assignment)
    selected = selected_windows(case, assignment, doc)
    assert selected == assignment["windows"]
    selected[0]["start"] = 0
    assert assignment == original


@pytest.mark.parametrize("field,value", [
    ("method", "hybrid_rerank"), ("question_id", "another"), ("question", "gold hints"),
    ("doc_id", "invented"), ("source_sha256", "wrong"),
    ("counts_as_source_inspection", True), ("split", "dev"), ("status", "error"),
])
def test_mismatched_assignment_cannot_be_used_in_live_reader(field, value):
    doc, case, assignment = fixture()
    assignment[field] = value
    with pytest.raises(ValueError):
        selected_windows(case, assignment, doc)


@pytest.mark.parametrize("windows", [
    [{"start": 0, "end": 1600}],
    [{"start": 0, "end": 2000}, {"start": 2400, "end": 4000}],
    [{"start": 0, "end": 1600}, {"start": 800, "end": 2400}],
    [{"start": 0, "end": 1600}, {"start": 20000, "end": 21600}],
])
def test_changed_budget_overlap_or_out_of_bounds_ranges_are_rejected(windows):
    doc, case, assignment = fixture()
    assignment["windows"] = windows
    with pytest.raises(ValueError):
        selected_windows(case, assignment, doc)


def payloads():
    doc, case, assignment = fixture()
    old = make_scope({case["doc_id"]: doc},
        [{"start": 0, "end": 1600}, {"start": 4000, "end": 5600}], version=PASSAGES_V2)
    new = make_scope({case["doc_id"]: doc}, assignment["windows"], version=PASSAGES_V2)
    payload = {"model": "envoy", "temperature": 0, "max_tokens": 1024,
        "messages": [{"role": "system", "content": "Original pinned system prompt"},
        {"role": "user", "content": "Original observation " + json.dumps(old, ensure_ascii=False)}]}
    return old, new, payload


def test_first_request_changes_only_the_literal_scope_json():
    old, new, payload = payloads()
    original = copy.deepcopy(payload)
    expected = replace_scope_only(payload, old, new)
    assert payload == original
    assert expected["messages"][0] == payload["messages"][0]
    assert expected["messages"][-1]["content"] == "Original observation " + json.dumps(new)
    for key in ("model", "temperature", "max_tokens"):
        assert expected[key] == payload[key]


@pytest.mark.parametrize("field,value", [
    ("doc_id", "different"), ("source_text_sha256", "wrong"),
    ("version", "passages-v1"), ("counts_as_source_inspection", True)])
def test_source_prompt_protocol_or_inspection_changes_cannot_hide_in_scope(field, value):
    old, new, payload = payloads()
    new[field] = value
    with pytest.raises(ValueError):
        replace_scope_only(payload, old, new)


def test_missing_or_duplicated_scope_is_not_accepted_as_a_matched_input():
    old, new, payload = payloads()
    for content in ["missing", json.dumps(old) + json.dumps(old)]:
        payload["messages"][-1]["content"] = content
        with pytest.raises(ValueError):
            replace_scope_only(payload, old, new)
