"""Readiness evidence cannot be borrowed from an old worker or activation."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.eval import assigned_smoke_lifecycle as lifecycle
from src.eval.assigned_smoke_lifecycle import logged_readiness, supervise


@pytest.mark.parametrize("worker,age,fitness,expected", [
    ("current", 1, True, True), ("old", 1, True, False),
    ("current", -1, True, False), ("current", 1, False, False)])
def test_both_current_readiness_events_required(worker, age, fitness, expected):
    since = datetime.now(timezone.utc)
    stamp = (since + timedelta(seconds=age)).isoformat()
    lines = ["vLLM is healthy"] + (["All fitness checks passed."] if fitness else [])
    logs = "\n".join(json.dumps({"workerId": worker, "ts": stamp, "line": line})
                     for line in lines)
    assert logged_readiness(logs, {"current"}, since) is expected


def test_elapsed_absolute_deadline_never_activates(tmp_path):
    calls = []
    def control(command, endpoint):
        calls.append(command)
        if command == "get":
            return {"workersMin": 0, "workersMax": 0,
                    "gpuIds": ["NVIDIA RTX A5000"], "gpuCount": 1}
        return {"jobs": {"inQueue": 0, "inProgress": 0}, "workers": {"ready": 0}}
    with pytest.raises(TimeoutError, match="original session deadline"):
        supervise("test", tmp_path, [], key="fake", control=control,
                  worker_reader=lambda: [], deadline_unix=0)
    assert calls == ["get", "health"]


@pytest.mark.parametrize("opt_in", [False, True])
def test_running_rollup_needs_explicit_opt_in_and_logged_readiness(tmp_path, monkeypatch, opt_in):
    state = {"workersMin": 0, "workersMax": 0, "gpuIds": ["NVIDIA RTX A5000"], "gpuCount": 1}
    started = []
    def control(command, endpoint, *args, **kwargs):
        if command == "get":
            return dict(state)
        if command == "update":
            state.update(workersMin=int(args[1]), workersMax=int(args[3]))
            return dict(state)
        if command == "health":
            return {"workers": {"ready": 0, "running": state["workersMax"]},
                    "jobs": {"inQueue": 0, "inProgress": 0}}
        assert command == "logs"
        stamp = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
        return "\n".join(json.dumps({"workerId": "current", "ts": stamp, "line": line})
                         for line in ["vLLM is healthy", "All fitness checks passed."])
    class Child:
        pid = 1
        def wait(self):
            return 0
    def popen(*args, **kwargs):
        started.append(True)
        return Child()
    monkeypatch.setattr(lifecycle, "worker_identity", lambda *a, **kw: {"fixture": True})
    monkeypatch.setattr(lifecycle, "kill_group", lambda pid: None)
    def workers():
        return [{"id": "current", "status": "RUNNING"}] if state["workersMax"] else []
    if opt_in:
        result = supervise("test", tmp_path, [], key="fixture", control=control,
            popen=popen, worker_reader=workers, seconds=1, logged_running_ready=True)
        assert result["exit_code"] == 0 and started == [True]
    else:
        with pytest.raises(TimeoutError):
            supervise("test", tmp_path, [], key="fixture", control=control,
                popen=popen, worker_reader=workers, seconds=0.02)
        assert started == []
    assert state["workersMin"] == state["workersMax"] == 0
