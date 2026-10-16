"""Submission conversion parses literals and never executes model-authored code."""

import pytest

from src.env.reward import parse_submission_details
from src.product.submit_call import submit_call_to_line


def test_keyword_call_round_trips():
    line = submit_call_to_line(
        'submit(answer="Eight heads", citations=["doc_a"], '
        'evidence=[{"doc_id": "doc_a", "start": 3, "end": 9}])')
    assert parse_submission_details(line) == (
        "Eight heads", ["doc_a"], [{"doc_id": "doc_a", "start": 3, "end": 9}])


def test_positional_uppercase_and_whitespace_normalization():
    line = submit_call_to_line('SUBMIT(" Yes\\n indeed ", ["d"], [])')
    assert parse_submission_details(line) == ("Yes indeed", ["d"], [])


def test_abstention_has_exact_empty_suffix():
    line = submit_call_to_line('submit(answer="Unanswerable", citations=[], evidence=[])')
    assert line.endswith("CITATIONS: [] EVIDENCE: []")


@pytest.mark.parametrize("code", [
    'print("x")', 'x = 1\nsubmit(answer="a")', "not python (", "",
    'print(submit(answer="a"))', 'submit(answer="a"); 1', 'obj.submit(answer="a")',
])
def test_other_code_is_not_a_submit_call(code):
    assert submit_call_to_line(code) is None


@pytest.mark.parametrize("code, message", [
    ("submit(answer=ans)", "literal values"),
    ('submit(answer="a" + "b")', "literal values"),
    ('submit(*["a"])', "literal values"),
    ('submit(answer="")', "non-empty answer"),
    ('submit(answer="a CITATIONS: x")', "must not contain"),
    ('submit(answer="a evidence: x")', "must not contain"),
    ('submit(answer="a", citations="doc")', "list of document IDs"),
    ('submit(answer="a", evidence=["x"])', "list of objects"),
    ('submit(answer="a", extra=1)', "unexpected or repeated"),
    ('submit("a", answer="b")', "unexpected or repeated"),
    ('submit(answer="a", answer="b")', "unexpected or repeated"),
    ('submit(**{"answer": "a"})', "unexpected or repeated"),
    ('submit("a", [], [], 1)', "takes answer"),
    ('submit(answer="a", evidence=[{"x": {1, 2}}])', "JSON-compatible"),
    ('submit(answer="a", evidence=[{"x": 1e999}])', "JSON-compatible"),
    ('submit(answer="a", evidence=[{[1]: 2}])', "literal values"),
])
def test_malformed_submit_calls_raise(code, message):
    with pytest.raises(ValueError, match=message):
        submit_call_to_line(code)


def test_call_argument_is_not_executed(tmp_path):
    target = tmp_path / "should-not-exist"
    action = f'submit(answer=__import__("pathlib").Path({str(target)!r}).write_text("executed"))'
    with pytest.raises(ValueError, match="literal values"):
        submit_call_to_line(action)
    assert not target.exists()
