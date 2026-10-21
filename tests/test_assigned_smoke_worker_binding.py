"""Legacy endpoint worker records cannot supply or replace a v2 log identity."""
import json

import pytest

from src.eval import assigned_smoke_lifecycle as lifecycle
from tests.assigned_smoke_cli_fixture import identity


def setup(monkeypatch, *, stale=False):
    state = {"id": "fixture", "workersMin": 0, "workersMax": 0,
             "gpuCount": 1, "gpuIds": ["NVIDIA RTX A5000"]}
    calls = []
    def control(command, endpoint, *args, raw=False):
        calls.append((command, args))
        if command == "get":
            # This obsolete record is deliberately unrelated and carries a secret field.
            return {**state, "workers": [{"id": "legacy-stale", "env": "private-legacy"}]}
        if command == "health":
            return {"jobs": {"inQueue": 0, "inProgress": 0},
                    "workers": {"ready": state["workersMin"]}}
        if command == "update":
            state.update(workersMin=int(args[1]), workersMax=int(args[3]))
            return state
        if command == "logs":
            assert not stale
            assert args[args.index("--worker") + 1] == "current-v2"
            assert args[args.index("--max-wait") + 1] == "5s"
            assert "--since" not in args and args[args.index("--tail") + 1] == "5000"
            return json.dumps({"source": "container", "workerId": "current-v2",
                               "ts": "2099-01-01T00:00:00Z",
                               "line": "ENVOY_PINNED_IDENTITY " + json.dumps(identity())})
        raise AssertionError(command)
    def workers():
        return [{"id": "current-v2", "status": "IDLE", "isStale": stale}]
    def catalog():
        return workers() if state["workersMin"] else []
    monkeypatch.setattr(lifecycle, "kill_group", lambda _: None)
    return state, control, catalog, calls


def test_supervisor_and_shutdown_use_only_current_v2_records(tmp_path, monkeypatch):
    state, control, workers, calls = setup(monkeypatch)
    class Child:
        pid = 123
        def wait(self):
            return 0
    result = lifecycle.supervise("fixture", tmp_path, [], key="scripted", control=control,
        worker_reader=workers, popen=lambda *a, **k: Child(), seconds=1)
    assert result["shutdown_verified"] and result["startup_worker_ids"] == ["current-v2"]
    assert state["workersMin"] == state["workersMax"] == 0
    assert all("--include-workers" not in args for _, args in calls)
    assert "private-legacy" not in "".join(p.read_text() for p in tmp_path.iterdir())
    read = json.loads((tmp_path / "startup-log-reads.jsonl").read_text())
    assert read["worker_id"] == "current-v2" and read["status"] == "received"


def test_stale_v2_worker_does_not_unlock_a_model_request(tmp_path, monkeypatch):
    state, control, workers, calls = setup(monkeypatch, stale=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("stale identity must not dispatch inference")
    with pytest.raises(TimeoutError):
        lifecycle.supervise("fixture", tmp_path, [], key="scripted", control=control,
            worker_reader=workers, popen=forbidden, seconds=0.02)
    assert not any(name == "logs" for name, _ in calls)
    assert state["workersMin"] == state["workersMax"] == 0
