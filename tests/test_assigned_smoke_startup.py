"""A warming worker is not a log-ready or verified inference worker."""
import json
import subprocess
import threading
from datetime import datetime, timezone

import pytest

from src.eval import assigned_smoke_lifecycle as lifecycle


def test_transient_log_read_returns_pending_to_the_session_watchdog(tmp_path):
    calls = []
    def control(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise lifecycle.ControlError("logs", "not_found", 404)
        return "current logs"
    value = lifecycle.identity_logs(control, "endpoint", {"worker"},
                                   datetime.now(timezone.utc), threading.Event(),
                                   audit=tmp_path / "reads.jsonl")
    assert value == "" and len(calls) == 1
    value = lifecycle.identity_logs(control, "endpoint", {"worker"},
                                   datetime.now(timezone.utc), threading.Event())
    assert value == "current logs" and len(calls) == 2
    assert json.loads((tmp_path / "reads.jsonl").read_text())["status"] == "pending"
    assert all(args[0] == "logs" for args in calls)  # No inference resubmission.


@pytest.mark.parametrize("code", ["unauthorized", "forbidden", "unknown"])
def test_permanent_log_failure_is_immediate(code):
    calls = []
    def broken(*args, **kwargs):
        calls.append(args)
        raise lifecycle.ControlError("logs", code, 404)
    with pytest.raises(lifecycle.ControlError) as failure:
        lifecycle.identity_logs(broken, "endpoint", {"worker"},
                                datetime.now(timezone.utc), threading.Event())
    assert len(calls) == 1 and failure.value.details["code"] == code


def test_partial_logs_survive_failed_read_without_trusting_an_incomplete_record(tmp_path):
    complete = json.dumps({"line": "complete private line"}) + "\n"
    partial = complete + '{"line":"unfinished'
    def broken(*args, **kwargs):
        raise lifecycle.ControlError("logs", "timeout", None, log_output=partial)
    audit = tmp_path / "reads.jsonl"
    logs = lifecycle.identity_logs(broken, "endpoint", {"worker"},
        datetime.now(timezone.utc), threading.Event(), audit=audit)
    assert logs == complete
    assert json.loads((tmp_path / "startup-log-output.jsonl").read_text())["output"] == partial
    assert "private" not in audit.read_text()
    assert (tmp_path / "startup-log-output.jsonl").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("process_timeout", [False, True])
def test_cli_preserves_partial_log_output_on_failure(monkeypatch, process_timeout):
    def failed(*args, **kwargs):
        if process_timeout:
            raise subprocess.TimeoutExpired(args, 30, output=b'complete\npartial')
        return subprocess.CompletedProcess(args, 1, 'complete\npartial',
            json.dumps({"code": "timeout", "error": "private stderr"}))
    monkeypatch.setattr(lifecycle.subprocess, "run", failed)
    with pytest.raises(lifecycle.ControlError) as failure:
        lifecycle.cli("logs", "endpoint", raw=True)
    assert failure.value.log_output == 'complete\npartial'
    assert failure.value.details == {"command": "logs", "code": "timeout", "status": None}
    assert "private" not in str(failure.value)


def test_cli_preserves_safe_error_code_without_printing_stderr(monkeypatch):
    def failed(*args, **kwargs):
        stderr = "--tail is ignored when --since is set\n" + json.dumps({
            "code": "not_found", "status": 404, "error": "sensitive raw message"})
        return subprocess.CompletedProcess(args, 1, "", stderr)
    monkeypatch.setattr(lifecycle.subprocess, "run", failed)
    with pytest.raises(lifecycle.ControlError) as failure:
        lifecycle.cli("logs", "endpoint")
    assert failure.value.details == {"command": "logs", "code": "not_found", "status": 404}
    assert "sensitive" not in str(failure.value)


def test_warmup_never_reads_logs_before_ready(tmp_path, monkeypatch):
    calls = []
    state = {"id": "test", "workersMin": 0, "workersMax": 0,
             "gpuIds": ["NVIDIA RTX A5000"], "gpuCount": 1}
    def control(command, endpoint, *args, **kwargs):
        calls.append(command)
        if command == "get":
            return {**state, "workers": [{"id": "warming"}] if state["workersMin"] else []}
        if command == "health":
            return {"jobs": {"inQueue": 0, "inProgress": 0},
                    "workers": {"ready": 0, "initializing": state["workersMin"]}}
        if command == "update":
            state.update(workersMin=int(args[1]), workersMax=int(args[3]))
            return state
        raise AssertionError("warming worker logs must not be fetched")
    monkeypatch.setattr(lifecycle.WorkerCatalog, "list", lambda self: [
        {"id": "warming", "status": "INITIALIZING"}] if state["workersMin"] else [])
    with pytest.raises(TimeoutError):
        lifecycle.supervise("test", tmp_path, [], key="fake", control=control, seconds=0.02)
    assert "logs" not in calls and state["workersMin"] == state["workersMax"] == 0
    assert json.loads((tmp_path / "lifecycle.json").read_text())["shutdown_verified"]
