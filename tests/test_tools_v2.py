"""Trusted fixtures exercise the opt-in layer without running recorded model code."""

import hashlib
import json

import pytest

from src.env.repl import LocalREPL
from src.env.tools import TOOL_PREAMBLE
from src.env.tools_v2 import HARNESS_V2_PREAMBLE, HARNESS_V2_SHA256, V1_TOOL_PREAMBLE_SHA256

TEXT = "Intro. " * 600 + "The model uses eight attention heads. " + "Tail. " * 600


@pytest.fixture()
def repl(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index in range(10):
        doc_id = f"doc_{index}"
        (corpus / f"{doc_id}.json").write_text(json.dumps(
            {"doc_id": doc_id, "title": f"Paper {index}", "text": TEXT}))
    session = LocalREPL(corpus_path=str(corpus), extra_preamble=HARNESS_V2_PREAMBLE)
    session.start_session()
    yield session
    session.kill_session()


def test_v1_preamble_is_pinned():
    assert V1_TOOL_PREAMBLE_SHA256 == (
        "233adfef3cd72c8b8221e45dcaa58c76809f36bd717d1d342514a7c332c3dd71")
    assert hashlib.sha256(TOOL_PREAMBLE.encode()).hexdigest() == V1_TOOL_PREAMBLE_SHA256
    assert HARNESS_V2_SHA256 == hashlib.sha256(HARNESS_V2_PREAMBLE.encode()).hexdigest()


@pytest.mark.parametrize("args", [
    '"doc_0", "attention heads"', '"attention heads", "doc_0"',
    '"doc_0", "attention heads", "2"', '"attention heads", "doc_0", 2',
    '"doc_0", query="attention heads"', '"attention heads", doc_id="doc_0", k=2',
    'doc_id="doc_0", q="attention heads", limit="2"',
])
def test_paper_scoped_aliases_in_both_orders(repl, args):
    out = repl.execute(f'print(search({args}))')
    assert "'offset'" in out and "'text'" in out and "Traceback" not in out


@pytest.mark.parametrize("alias", ["top_k", "k", "n", "limit", "max_results", "num_results"])
def test_numeric_aliases(repl, alias):
    assert repl.execute(f'print(len(search("attention", {alias}="2")))').strip() == "2"


def test_default_limits_distinguish_discovery_and_scoped_search(repl):
    assert repl.execute('print(len(search("intro")))').strip() == "5"
    assert repl.execute('print(len(search_within("doc_0", "intro tail")))').strip() == "8"
    assert repl.execute('print(len(search("doc_0", "intro tail")))').strip() == "8"


def test_legacy_chunk_method_positional_form(repl):
    out = repl.execute('print(len(search("attention", "2", "chunk")))')
    assert out.strip() == "2"


def test_read_ranges_and_passage_end_and_cap(repl):
    assert repl.execute('r = read("doc_0", 0, 20); print(r["start"], r["end"])').strip() == "0 20"
    assert repl.execute('r = read("doc_0", offset="5", length=10); print(r["end"])').strip() == "15"
    assert repl.execute('r = passage("doc_0", 0, end="30"); print(r["end"])').strip() == "30"
    assert repl.execute('r = passage("doc_0", 0, 99999); print(r["end"])').strip() == "3000"


@pytest.mark.parametrize("action", [
    'search("attention", bogus=1)', 'search("attention", 1, top_k=2)',
    'search("attention", k=1, limit=2)', 'search("attention", True)',
    'search("attention", query="other")', 'search("attention", "unknown_doc")',
    'search_within("doc_0", "attention", query="other")',
    'search_within("doc_0", 123)', 'read("doc_0", 0, start=1)',
    'read("doc_0", 0, 10, end=20)', 'passage("doc_0", 0, 10, end=20)',
    'passage("doc_0", length=0)', 'read(["doc_0"])',
])
def test_bad_and_duplicate_arguments_have_usage(repl, action):
    assert "Usage: " in repl.execute(action)


def test_submit_inside_code_explains_only_statement_rule(repl):
    assert "only statement" in repl.execute('x = 1\nsubmit(answer="y")')


def test_cumulative_preamble_does_not_double_wrap(repl):
    repl.execute(TOOL_PREAMBLE + "\n" + HARNESS_V2_PREAMBLE)
    out = repl.execute('print(search("doc_0", query="attention heads"))')
    assert "'offset'" in out and "Traceback" not in out
