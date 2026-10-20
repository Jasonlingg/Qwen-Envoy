"""Assigned-paper source inspection requires an executed displayed tool result."""
import ast

import pytest

from src.product.assigned_actions import displayed_tool_call, source_text_visible
from tests.test_assigned_reader import DOC_ID, INSPECT, SUBMIT, setup_reader
from tests.test_qwen_investigator import FakeREPL


@pytest.mark.parametrize("action", [
    'if False:\n    read("id")\nprint("Metadata only.")',
    'def later():\n    read("id")\nprint("Metadata only.")',
    'print("metadata", read("id"))',
    'read = lambda _: "fake"',
    'read = search("query"); print(read)',
    'json = search("query"); print(json)',
    'os = search("query"); print(os)',
    'Path = search("query"); print(Path)',
    're = search("query"); print(re)',
    'Counter = search("query"); print(Counter)',
    '_v1_read = search("query"); print(_v1_read)',
    'print(read.__call__("id"))',
    'print(read("id")[0])',
    'print(read(*["id"]))',
    'print(read(**{"doc_id":"id"}))',
    'print(read(__import__("os").getcwd()))',
    'print(passage("id", True, 10))',
    'print(read(["id"]))',
    'print(read("id")); print("metadata")',
    'hits = read("id"); print("metadata")',
    'print("metadata")',
])
def test_unsupported_code_cannot_redefine_tools_or_spoof_inspection(action):
    with pytest.raises(ValueError, match="literal document-tool"):
        displayed_tool_call(action)


@pytest.mark.parametrize("action", [
    'read("id")', 'print(read("id"))',
    'hits = read("id"); print(hits)',
    'result = passage("id", start=0, length=100); print(result)',
    'print(search("query", "id", top_k="8"))',
])
def test_displayed_tool_forms_remain_python(action):
    assert isinstance(displayed_tool_call(action), ast.Call)


def test_observation_must_contain_exact_source_text():
    docs = {"id": {"text": "Actual source text."}}
    read = displayed_tool_call('print(read("id"))')
    assert source_text_visible(read, "Actual source text.", docs)
    assert not source_text_visible(read, "Metadata only.", docs)
    search = displayed_tool_call('print(search_within("id", "source"))')
    assert source_text_visible(search, "[{'text':'source text.', 'offset':7}]", docs)
    assert not source_text_visible(search, "[{'text':'invented', 'offset':7}]", docs)
    assert not source_text_visible(search, "[]", docs)
    assert not source_text_visible(search, "[{'text':'source text.', 'offset':True}]", docs)


def test_dead_code_regression_does_not_execute_or_unlock_submission(tmp_path):
    class MetadataOnly(FakeREPL):
        def execute(self, code, **kwargs):
            self.executed.append((code, kwargs))
            return "Metadata only."
    repl = MetadataOnly()
    reader, _, _, snapshot = setup_reader(tmp_path,
        [f'if False:\n    read("{DOC_ID}")\nprint("Metadata only.")', SUBMIT],
        repl=repl, steps=2)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "incomplete" and packet["evidence"] == []
    assert packet["inspection_steps"] == 0 and repl.executed == []


def test_valid_tool_syntax_with_metadata_output_still_does_not_unlock_submission(tmp_path):
    reader, _, _, snapshot = setup_reader(tmp_path, [INSPECT, SUBMIT],
                                            repl=FakeREPL(), steps=2)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "incomplete" and packet["inspection_steps"] == 0


def test_wrapper_and_whitespace_changes_cannot_make_a_duplicate_call_distinct(tmp_path):
    reader, _, repl, snapshot = setup_reader(tmp_path, [INSPECT,
        f'  search_within("{DOC_ID}", "memory")  ', SUBMIT], steps=3)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "evidence_found" and len(repl.executed) == 1
    assert packet["document_tool_steps"] == 1
