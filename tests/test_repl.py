"""Tests for the persistent REPL."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.env.repl import (
    DockerREPL,
    LocalREPL,
    PersistentREPL,
    _auto_print_trailing_expression,
)
from src.env.tools import SEARCH_PROTOCOL_VERSION


@pytest.fixture
def repl() -> LocalREPL:
    """Create a local REPL for testing."""
    r = LocalREPL(corpus_path="data/corpus")
    r.start_session()
    yield r
    r.kill_session()


def test_state_persistence(repl: LocalREPL) -> None:
    """Variables from step 1 should be available in step 2."""
    repl.execute("x = 42")
    output = repl.execute("print(x)")
    assert "42" in output


def test_function_persistence(repl: LocalREPL) -> None:
    """Functions defined in step 1 should be callable in step 2."""
    repl.execute("def double(n): return n * 2")
    output = repl.execute("print(double(21))")
    assert "42" in output


def test_successful_action_is_not_replayed_on_next_turn(
    repl: LocalREPL, tmp_path,
) -> None:
    """A persistent worker executes each action once instead of replaying history."""
    marker = tmp_path / "executions.txt"
    repl.execute(f'open({str(marker)!r}, "a").write("executed\\n")')
    repl.execute('print("next turn")')
    assert marker.read_text().splitlines() == ["executed"]


def test_same_worker_process_handles_successive_turns(repl: LocalREPL) -> None:
    first_pid = repl.execute("__import__('os').getpid()").strip()
    second_pid = repl.execute("__import__('os').getpid()").strip()
    assert first_pid == second_pid


def test_concurrent_sessions_keep_processes_and_state_isolated() -> None:
    def run_session(value: int) -> tuple[str, str]:
        session = LocalREPL(corpus_path="data/corpus")
        session.start_session()
        try:
            session.execute(f"private_value = {value}")
            pid = session.execute("__import__('os').getpid()").strip()
            observed = session.execute("private_value").strip()
            return pid, observed
        finally:
            session.kill_session()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run_session, 11)
        second = pool.submit(run_session, 29)
        first_pid, first_value = first.result()
        second_pid, second_value = second.result()

    assert first_pid != second_pid
    assert (first_value, second_value) == ("11", "29")


def test_tools_available(repl: LocalREPL) -> None:
    """Tool preamble should make search/read available."""
    output = repl.execute("print(type(search))")
    assert "function" in output


def test_search_tool(repl: LocalREPL) -> None:
    """search() should return results from the corpus."""
    output = repl.execute('results = search("revenue"); print(len(results))')
    # Should have at least 1 result
    assert any(c.isdigit() and int(c) > 0 for c in output.split() if c.isdigit())


def test_search_finds_a_match_in_a_single_document_corpus(tmp_path) -> None:
    """A personal vault may begin with one note; matching terms must stay searchable."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "only_note.json").write_text(json.dumps({
        "doc_id": "only_note",
        "title": "Only note",
        "text": "trajectory distillation improves a compact research agent",
    }))
    repl = LocalREPL(corpus_path=str(corpus))
    repl.start_session()
    try:
        output = repl.execute('search("trajectory")')
    finally:
        repl.kill_session()
    assert "only_note" in output


def test_search_ranks_relevant_paper_above_long_generic_text(tmp_path) -> None:
    """Repeated common question words must not bury a title-specific source."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for doc_id, title, content in (
        ("generic", "General Review", "how does the method compare to reported results " * 500),
        ("agent_flan", "Agent FLAN", "Agent FLAN trains Llama2-7B on agent tuning data."),
        ("other", "Other Agent", "Searches documents and reports results."),
    ):
        (corpus / f"{doc_id}.json").write_text(json.dumps({
            "doc_id": doc_id, "title": title, "text": content,
        }))
    repl = LocalREPL(corpus_path=str(corpus))
    repl.start_session()
    try:
        output = repl.execute(
            'search("How does Agent FLAN compare to reported results?", top_k=3)'
        )
        protocol_version = repl.execute("SEARCH_PROTOCOL_VERSION").strip()
    finally:
        repl.kill_session()
    assert output.index("'doc_id': 'agent_flan'") < output.index("'doc_id': 'generic'")
    assert f"'search_version': '{SEARCH_PROTOCOL_VERSION}'" in output
    assert protocol_version == SEARCH_PROTOCOL_VERSION


def test_search_within_default_depth_is_configurable(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    separated_matches = ("needle " + "x" * 600) * 10
    (corpus / "paper.json").write_text(json.dumps({
        "doc_id": "paper",
        "title": "Paper",
        "text": separated_matches,
    }))
    monkeypatch.setenv("ENVOY_SEARCH_WITHIN_TOP_K", "8")
    repl = LocalREPL(corpus_path=str(corpus))
    repl.start_session()
    try:
        output = repl.execute('print(len(search_within("paper", "needle")))')
    finally:
        repl.kill_session()
    assert output.strip() == "8"


def test_search_within_deduplicates_and_merges_overlapping_hits(
    tmp_path, monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = list("x" * 3000)
    for offset, value in [
        (100, "needle"), (300, "needle"), (500, "needle"),
        (2000, "needle"),
    ]:
        text[offset:offset + len(value)] = value
    document = "".join(text)
    (corpus / "paper.json").write_text(json.dumps({
        "doc_id": "paper", "title": "Paper", "text": document,
    }))
    monkeypatch.setenv("ENVOY_SEARCH_WITHIN_MODE", "dedupe_merge")
    repl = LocalREPL(corpus_path=str(corpus))
    repl.start_session()
    try:
        output = repl.execute(
            'print(json.dumps(search_within("paper", "needle", top_k=3)))'
        )
    finally:
        repl.kill_session()
    results = json.loads(output)
    intervals = [
        (item["offset"], item["offset"] + len(item["text"]))
        for item in results
    ]

    assert len(results) == 2
    assert any(start <= 2000 < end for start, end in intervals)
    assert all(end - start <= 900 for start, end in intervals)
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    assert all(item["text"] == document[start:end]
               for item, (start, end) in zip(results, intervals))


def test_search_within_raw_mode_preserves_ranked_windows(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    document = "needle " + "x" * 200 + "needle " + "x" * 800
    (corpus / "paper.json").write_text(json.dumps({
        "doc_id": "paper", "title": "Paper", "text": document,
    }))
    monkeypatch.setenv("ENVOY_SEARCH_WITHIN_MODE", "raw")
    repl = LocalREPL(corpus_path=str(corpus))
    repl.start_session()
    try:
        output = repl.execute(
            'print(json.dumps(search_within("paper", "needle", top_k=2)))'
        )
    finally:
        repl.kill_session()
    results = json.loads(output)

    assert len(results) == 2
    assert all(len(item["text"]) <= 500 for item in results)


def test_search_within_dedupe_does_not_expand_filled_results(
    tmp_path, monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = list("x" * 3000)
    # These produce three equally ranked regions. Once the third region fills
    # the final slot, its later overlapping windows must not expand it.
    for offset in [450, 1450, 2450]:
        text[offset:offset + len("needle needle")] = "needle needle"
    document = "".join(text)
    (corpus / "paper.json").write_text(json.dumps({
        "doc_id": "paper", "title": "Paper", "text": document,
    }))
    monkeypatch.setenv("ENVOY_SEARCH_WITHIN_MODE", "dedupe_merge")
    repl = LocalREPL(corpus_path=str(corpus))
    repl.start_session()
    try:
        output = repl.execute(
            'print(json.dumps(search_within("paper", "needle", top_k=3)))'
        )
    finally:
        repl.kill_session()
    results = json.loads(output)

    assert len(results) == 3
    assert [item["offset"] for item in results] == [0, 1000, 2000]
    assert [len(item["text"]) for item in results] == [900, 900, 500]


def test_search_within_ranked_diverse_preserves_raw_prefix(
    tmp_path, monkeypatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = list("x" * 5000)
    for offset in [450, 1450, 2450, 3450, 4450]:
        text[offset:offset + len("needle needle")] = "needle needle"
    document = "".join(text)
    (corpus / "paper.json").write_text(json.dumps({
        "doc_id": "paper", "title": "Paper", "text": document,
    }))

    monkeypatch.setenv("ENVOY_SEARCH_WITHIN_MODE", "raw")
    raw = LocalREPL(corpus_path=str(corpus))
    raw.start_session()
    try:
        raw_results = json.loads(raw.execute(
            'print(json.dumps(search_within("paper", "needle", top_k=3)))'
        ))
    finally:
        raw.kill_session()

    monkeypatch.setenv("ENVOY_SEARCH_WITHIN_MODE", "ranked_diverse")
    diverse = LocalREPL(corpus_path=str(corpus))
    diverse.start_session()
    try:
        diverse_results = json.loads(diverse.execute(
            'print(json.dumps(search_within("paper", "needle", top_k=6)))'
        ))
    finally:
        diverse.kill_session()

    assert diverse_results[:3] == raw_results
    assert len(diverse_results) == 6
    prefix_intervals = [
        (item["offset"], item["offset"] + len(item["text"]))
        for item in diverse_results[:3]
    ]
    for item in diverse_results[3:]:
        interval = (item["offset"], item["offset"] + len(item["text"]))
        assert all(interval[1] <= start or interval[0] >= end
                   for start, end in prefix_intervals)


def test_read_tool(repl: LocalREPL) -> None:
    """read() should return document text."""
    output = repl.execute('text = read("apex_corp_2024_financial"); print("Apex" in text)')
    assert "True" in output


def test_passage_returns_exact_text_and_offsets(repl: LocalREPL) -> None:
    output = repl.execute(
        'result = passage("apex_corp_2024_financial", 0, 20); '
        'print(result["start"], result["end"], len(result["text"]))'
    )
    assert output.strip() == "0 20 20"


def test_scan_returns_bounded_non_overlapping_contexts(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "prefix " + "x" * 300 + "Dataset Alpha and corpus Beta. " + "y" * 1000
    text += "Benchmark Gamma contains the answer."
    (corpus / "paper.json").write_text(json.dumps({
        "doc_id": "paper", "title": "Paper", "text": text,
    }))
    repl = LocalREPL(corpus_path=str(corpus))
    repl.start_session()
    try:
        output = repl.execute(
            'print(json.dumps(scan("paper", r"dataset|corpus|benchmark", '
            'max_hits=3, context_chars=300)))'
        )
    finally:
        repl.kill_session()
    results = json.loads(output)

    assert len(results) == 2
    assert [item["match"].lower() for item in results] == ["dataset", "benchmark"]
    assert all(item["text"] == text[item["offset"]:item["end"]] for item in results)
    assert all(len(item["text"]) <= 300 for item in results)
    assert results[0]["end"] <= results[1]["offset"]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        ('scan("missing", "answer")', "not found"),
        ('scan("paper", "[")', "Invalid regex"),
        (f'scan("paper", {"x" * 161!r})', "at most 160"),
        ('scan("paper", "answer", max_hits=0)', "max_hits"),
        ('scan("paper", "answer", context_chars=100)', "context_chars"),
    ],
)
def test_scan_reports_invalid_inputs(tmp_path, call, message) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "paper.json").write_text(json.dumps({
        "doc_id": "paper", "title": "Paper", "text": "answer",
    }))
    repl = LocalREPL(corpus_path=str(corpus))
    repl.start_session()
    try:
        output = repl.execute(call)
    finally:
        repl.kill_session()

    assert message in output


def test_bare_search_call_auto_prints_its_result(repl: LocalREPL) -> None:
    """A bare search(...) with no print() must still produce visible output —
    otherwise the model gets zero feedback and keeps retrying blind."""
    output = repl.execute('search("revenue")')
    assert output.strip()
    assert "doc_id" in output or "[" in output


def test_multiline_step_still_auto_prints_only_the_trailing_expression(
    repl: LocalREPL,
) -> None:
    output = repl.execute('x = 1\nsearch("revenue")')
    assert output.strip()


def test_print_already_present_is_not_double_wrapped() -> None:
    code = 'print(search("revenue"))'
    assert _auto_print_trailing_expression(code) == code


def test_trailing_assignment_is_left_untouched() -> None:
    code = 'results = search("revenue")'
    assert _auto_print_trailing_expression(code) == code


def test_trailing_for_loop_is_left_untouched() -> None:
    code = 'for r in search("revenue"):\n    pass'
    assert _auto_print_trailing_expression(code) == code


def test_invalid_syntax_is_returned_unchanged() -> None:
    code = "def broken("
    assert _auto_print_trailing_expression(code) == code


def test_semicolon_separated_statements_on_one_line_are_all_preserved() -> None:
    """A line-based rewrite would clobber the import when it shares a line
    with the trailing expression; the fix must not drop it."""
    rewritten = _auto_print_trailing_expression("import time; time.sleep(0)")
    assert "import time" in rewritten
    compile(rewritten, "<test>", "exec")


def test_timeout(repl: LocalREPL) -> None:
    """Long-running code should timeout."""
    output = repl.execute("import time; time.sleep(10)", timeout=2)
    assert "timeout" in output.lower() or "ERROR" in output


def test_session_cleanup() -> None:
    """kill_session should clean up without errors."""
    r = LocalREPL(corpus_path="data/corpus")
    r.start_session()
    r.execute("x = 1")
    r.kill_session()
    # Should not raise
    r.kill_session()


def test_auto_select_local() -> None:
    """PersistentREPL should fall back to local when Docker unavailable."""
    repl = PersistentREPL(use_docker=False, corpus_path="data/corpus")
    repl.start_session()
    output = repl.execute("print(1 + 1)")
    assert "2" in output
    repl.kill_session()


def test_large_previous_output_does_not_hide_current_step(repl: LocalREPL) -> None:
    repl.execute('print("x" * 9000)')
    assert repl.execute('print("CURRENT_STEP_SENTINEL")').strip() == "CURRENT_STEP_SENTINEL"


@pytest.mark.parametrize("code", [
    'raise ValueError("old failure")',
    '1 / 0',
    'if broken syntax',
    'import sys; sys.exit(3)',
])
def test_failed_step_rolls_back_and_next_action_runs(repl: LocalREPL, code: str) -> None:
    repl.execute("x = 42")
    repl.execute(code)
    assert repl.execute('print("RECOVERED", x)').strip() == "RECOVERED 42"


def test_timeout_does_not_poison_next_action(repl: LocalREPL) -> None:
    repl.execute("import time; time.sleep(10)", timeout=1)
    assert repl.execute('print("RECOVERED")', timeout=2).strip() == "RECOVERED"


def test_old_stderr_is_not_replayed_and_printed_errors_are_not_failures(repl: LocalREPL) -> None:
    repl.execute('import sys; print("old warning", file=sys.stderr); x = 7')
    assert repl.execute('print(x)').strip() == "7"
    repl.execute('print("SyntaxError"); x = 9')
    assert repl.execute('print(x)').strip() == "9"


def test_docker_transport_preserves_output_and_recovers(monkeypatch):
    """Exercise Docker's script transport/output logic with a local process.

    This is not a real Docker isolation or resource-limit test.
    """
    original_run = subprocess.run
    script = ""

    def docker_run(args, **kwargs):
        nonlocal script
        if args[1] == "create":
            return subprocess.CompletedProcess(args, 0, "test-container", "")
        if args[1] in {"start", "rm"}:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[-1] == "cat > /tmp/step.py":
            script = kwargs["input"]
            return subprocess.CompletedProcess(args, 0, "", "")
        assert "timeout" in args and "--signal=KILL" in args
        return original_run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", docker_run)
    repl = DockerREPL()
    repl.start_session()
    try:
        repl.execute('print("x" * 9000); x = 7')
        assert repl.execute('print(x)').strip() == "7"
        repl.execute('raise ValueError("bad")')
        assert repl.execute('print(x)').strip() == "7"
        # A literal heredoc delimiter is data, not a shell command boundary.
        assert "SCRIPT_EOF" in repl.execute('print("""\nSCRIPT_EOF\n""")')
    finally:
        repl.kill_session()
