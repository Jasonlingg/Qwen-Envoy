"""A warming worker is not a log-ready or verified inference worker."""
import json
import subprocess
from datetime import datetime, timezone

import pytest

from src.eval import assigned_smoke_lifecycle as lifecycle


def test_only_transient_log_reads_are_retried_within_same_session():
    calls = []
    def control(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise lifecycle.ControlError("logs", "not_found", 404)
        return "current logs"
    class Immediate:
        def wait(self, seconds):
            return False
    value = lifecycle.identity_logs(control, "endpoint", {"worker"},
                                   datetime.now(timezone.utc), Immediate())
    assert value == "current logs" and len(calls) == 2
    assert all(args[0] == "logs" for args in calls)  # No inference resubmission.


@pytest.mark.parametrize("code,expected", [("not_found", 3), ("unauthorized", 1), ("unknown", 1)])
def test_log_failure_has_finite_read_attempts_and_preserved_reason(code, expected):
    calls = []
    def broken(*args, **kwargs):
        calls.append(args)
        raise lifecycle.ControlError("logs", code, 404)
    class Immediate:
        def wait(self, seconds):
            return False
    with pytest.raises(lifecycle.ControlError) as failure:
        lifecycle.identity_logs(broken, "endpoint", {"worker"},
                                datetime.now(timezone.utc), Immediate())
    assert len(calls) == expected and failure.value.details["code"] == code


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
    with pytest.raises(TimeoutError):
        lifecycle.supervise("test", tmp_path, [], key="fake", control=control, seconds=0.02)
    assert "logs" not in calls and state["workersMin"] == state["workersMax"] == 0
    assert json.loads((tmp_path / "lifecycle.json").read_text())["shutdown_verified"]
