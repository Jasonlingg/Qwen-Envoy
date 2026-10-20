"""Scripted inference only: budgets, source checks, shutdown and one-attempt behavior."""
import json

import pytest

from src.eval import assigned_smoke_lifecycle as lifecycle
from src.eval.assigned_smoke_plan import prepare
from src.eval.assigned_smoke_run import require_completed_cases, run_cases
from src.policies.runpod_queue import RunpodQueueTransport
from src.product.assigned_reader import build_assigned_reader
from tests.test_assigned_reader import SourceREPL
from tests.test_assigned_smoke_plan import IMAGE, Tokenizer, cached  # noqa: F401


@pytest.mark.parametrize("rejected_assignment", [False, True])
@pytest.mark.parametrize("guidance", ["baseline", "focused-v1"])
def test_two_real_reader_loops_use_fixed_payloads_separate_six_request_ledgers(
        cached, rejected_assignment, guidance):  # noqa: F811
    library, _, prepared = cached
    plan = prepare(library.root, prepared, sandbox_image=IMAGE, tokenizer=Tokenizer(),
                   reader_guidance=guidance)
    doc = library.document(plan["paper_version"])
    did = doc["doc_id"]
    start = doc["text"].index("We measure walltime")
    actions = [f'print(search_within("{did}", "batch size"))',
        f'submit(answer="1", citations=["{did}"], evidence=[{{"doc_id":"{did}",'
        f'"start":{start},"end":{start + 80}}}])',
        f'print(read("{did}"))', f'print(search_within("{did}", "A5000"))',
        'report_gap(missing="A5000 batch8 measurements", follow_up_query="A5000 batch8")',
        'submit(answer="Unanswerable", citations=[], evidence=[])']
    if rejected_assignment:
        # Observed live: an assignment without display is rejected before execution.
        actions.insert(0, f'hits = search_within("{did}", "batch size")')
    actions = iter(actions)
    posts = []
    def factory(endpoint, key, tokenizer, audit, **kwargs):
        assert kwargs == {"requests_max": 6}
        def http(method, path, payload=None):
            assert method == "POST" and path == "/run"
            posts.append(payload)
            return {"id": f"job-{len(posts)}", "status": "COMPLETED", "output": {
                "model": "envoy", "choices": [{"message": {"content": next(actions)}}]}}
        return RunpodQueueTransport(endpoint, key, tokenizer, audit, http=http, **kwargs)
    def reader(*args, **kwargs):
        result = build_assigned_reader(*args, **kwargs)
        assert result.max_steps == 6
        result.repl_factory = lambda _: SourceREPL(doc["text"])
        return result
    output = prepared.parent / "run"
    output.mkdir()
    summary = run_cases(prepared, output, tokenizer=Tokenizer(), key="fake",
                         transport_factory=factory, reader_factory=reader)
    assert [r["status"] for r in summary["cases"]] == ["evidence_found", "no_evidence"]
    assert summary["model_requests_reserved"] == len(posts) == 6 + rejected_assignment
    assert all(r["packet_verification"] == "passed" for r in summary["cases"])
    assert all(r["semantic_support"] == "not_reviewed" for r in summary["cases"])
    assert summary["cases"][0]["action_rejection_steps"] == ([1] if rejected_assignment else [])
    assert all(r["execution_error_steps"] == [] for r in summary["cases"])
    if guidance == "focused-v1":
        assert all(r["execution_path"] == "mcp_in_process" and
                   r["mcp_verification"] == "passed" for r in summary["cases"])
        replies = [json.loads(p.read_text()) for p in output.glob("*.mcp.json")]
        assert len(replies) == 2 and all(r["tool"] == "investigate" for r in replies)
        gap_reply = json.loads((output / "assigned-a5000-batch8-v1.mcp.json").read_text())
        assert gap_reply["result"]["research_gaps"][0]["absence_proven"] is False
    with pytest.raises(FileExistsError):
        run_cases(prepared, output, tokenizer=Tokenizer(), key="fake",
                  transport_factory=factory, reader_factory=reader)
    assert len(posts) == 6 + rejected_assignment  # No replacement budget.


@pytest.mark.parametrize("failure", ["status", "packet_verification", "audit", "unattempted"])
def test_partial_or_unverified_session_cannot_be_success(failure):
    row = {"status": "evidence_found", "packet_verification": "passed", "audit": {"failed": 0}}
    summary = {"cases": [row], "unattempted_cases": []}
    if failure == "unattempted":
        summary["unattempted_cases"] = ["case2"]
    else:
        row[failure] = {"status": "error", "packet_verification": "failed",
                        "audit": {"failed": 1}}[failure]
    with pytest.raises(RuntimeError, match="results retained"):
        require_completed_cases(summary)


@pytest.fixture
def control(monkeypatch):
    calls = []
    state = {"id": "test", "workersMin": 0, "workersMax": 0, "gpuCount": 1,
             "gpuIds": ["NVIDIA RTX A5000"], "idleTimeout": 60}
    health = {"jobs": {"inQueue": 0, "inProgress": 0}, "workers": {"ready": 0}}
    def invoke(command, endpoint, *args, raw=False):
        calls.append((command, args))
        if command == "get":
            return {**state, "workers": [{"id": "current"}] if state["workersMin"] else []}
        if command == "health":
            return health
        if command == "logs":
            return ""
        if command == "update":
            state.update(workersMin=int(args[1]), workersMax=int(args[3]))
            health["workers"]["ready"] = state["workersMin"]
            return dict(state)
        raise AssertionError(command)
    monkeypatch.setattr(lifecycle.WorkerCatalog, "list", lambda self: [
        {"id": "current", "status": "IDLE"}] if state["workersMin"] else [])
    return invoke, calls, state


def test_watchdog_covers_cold_start_and_parks_without_inference(tmp_path, control):
    invoke, calls, state = control
    def forbidden(*args, **kwargs):
        raise AssertionError("worker must not start without verified warm identity")
    with pytest.raises(TimeoutError):
        lifecycle.supervise("test", tmp_path, [], key="fake", control=invoke,
                            popen=forbidden, seconds=0.02)
    result = json.loads((tmp_path / "lifecycle.json").read_text())
    assert result["watchdog_fired"] and result["exit_code"] is None
    assert state["workersMin"] == state["workersMax"] == 0
    assert calls[0][0] == "get" and calls[-2][0] == "get"


def test_activation_failure_still_parks_and_reads_back(tmp_path, control):
    invoke, calls, state = control
    def broken(command, endpoint, *args, **kwargs):
        if command == "update" and args[1] == "1":
            raise RuntimeError("uncertain control submission")
        return invoke(command, endpoint, *args, **kwargs)
    with pytest.raises(RuntimeError):
        lifecycle.supervise("test", tmp_path, [], key="fake", control=broken, seconds=1)
    assert state["workersMin"] == state["workersMax"] == 0
    assert ("update", ("--workers-min", "0", "--workers-max", "0")) in calls
    after = json.loads((tmp_path / "lifecycle.json").read_text())["endpoint_after"]
    assert after["workersMax"] == 0


def test_busy_endpoint_is_never_changed(tmp_path, control):
    invoke, calls, state = control
    state["workersMax"] = 1
    with pytest.raises(ValueError, match="parked"):
        lifecycle.supervise("test", tmp_path, [], key="fake", control=invoke)
    assert not any(c[0] == "update" for c in calls)


def test_interrupted_worker_is_killed_reaped_and_endpoint_parked(tmp_path, control, monkeypatch):
    import subprocess
    import sys
    invoke, calls, state = control
    monkeypatch.setattr(lifecycle, "worker_identity", lambda *a, **k: {"scripted_identity": True})
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             start_new_session=True)
    wait = child.wait
    count = 0
    def interrupted():
        nonlocal count
        count += 1
        if count == 1:
            raise KeyboardInterrupt
        return wait(timeout=5)
    monkeypatch.setattr(child, "wait", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            lifecycle.supervise("test", tmp_path, [], key="fake", control=invoke,
                                popen=lambda *a, **k: child, seconds=1)
        assert child.poll() is not None and count == 2
        assert state["workersMin"] == state["workersMax"] == 0
    finally:
        lifecycle.kill_group(child.pid)
        wait(timeout=5)


def test_shutdown_rejects_lingering_workers_and_pending_jobs(control):
    from src.eval.assigned_smoke_cleanup import wait_parked
    invoke, _, state = control
    def lingering(command, *args, **kwargs):
        result = invoke(command, *args, **kwargs)
        if command == "get":
            result["workers"] = [{"id": "lingering"}]
        return result
    with pytest.raises(RuntimeError, match="shutdown not verified"):
        wait_parked(lingering, "test", worker_reader=lambda: [{"id": "lingering"}], seconds=0)


def test_failed_child_cannot_exit_as_success(tmp_path, control, monkeypatch):
    invoke, _, _ = control
    monkeypatch.setattr(lifecycle, "worker_identity", lambda *a, **k: {"scripted_identity": True})
    monkeypatch.setattr(lifecycle, "kill_group", lambda _: None)
    class Failed:
        pid = 123
        def wait(self):
            return 1
    with pytest.raises(RuntimeError, match="did not complete"):
        lifecycle.supervise("test", tmp_path, [], key="fake", control=invoke,
                            popen=lambda *a, **k: Failed(), seconds=1)
    assert json.loads((tmp_path / "lifecycle.json").read_text())["shutdown_verified"]
