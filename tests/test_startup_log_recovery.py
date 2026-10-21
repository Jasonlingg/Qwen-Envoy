"""Recovery cannot relax identity, readiness, source, worker or deadline boundaries."""
import json
import threading
from datetime import datetime, timezone

import pytest

from src.eval import assigned_smoke_lifecycle as lifecycle
from tests.assigned_smoke_cli_fixture import identity


def record(worker, text, *, source="container"):
    return json.dumps({"source": source, "workerId": worker,
                       "ts": "2099-01-01T00:00:00Z", "line": text})


def test_conflicting_identity_in_later_read_cannot_be_hidden_by_cached_identity():
    good = record("current", "ENVOY_PINNED_IDENTITY " + json.dumps(identity()))
    bad = record("current", "ENVOY_PINNED_IDENTITY " +
                 json.dumps({**identity(), "adapter_revision": "wrong"}))
    with pytest.raises(ValueError, match="identity"):
        lifecycle.worker_identity(good + "\n" + bad, active_workers={"current"},
                                  since=datetime.now(timezone.utc))


def test_system_logs_cannot_supply_model_identity_or_readiness():
    logs = "\n".join(record("current", text, source="system") for text in [
        "ENVOY_PINNED_IDENTITY " + json.dumps(identity()),
        "vLLM is healthy", "All fitness checks passed."])
    since = datetime.now(timezone.utc)
    assert lifecycle.worker_identity(logs, active_workers={"current"}, since=since) is None
    assert not lifecycle.logged_readiness(logs, {"current"}, since)


@pytest.mark.parametrize("failure", ["quiet", "replacement", "missing_fitness", "same_id_stale"])
def test_incomplete_startup_parks_at_deadline_without_dispatch(tmp_path, monkeypatch, failure):
    state = {"workersMin": 0, "workersMax": 0, "gpuIds": ["NVIDIA RTX A5000"],
             "gpuCount": 1}
    reads = []
    active_reads = []
    original_announcement = []
    def control(command, endpoint, *args, **kwargs):
        if command == "get":
            return dict(state)
        if command == "update":
            state.update(workersMin=int(args[1]), workersMax=int(args[3]))
            return dict(state)
        if command == "health":
            # Even ready=1 cannot replace the explicitly required engine/fitness logs.
            return {"workers": {"ready": state["workersMin"]},
                    "jobs": {"inQueue": 0, "inProgress": 0}}
        assert command == "logs"
        reads.append(args)
        if failure == "quiet":
            raise lifecycle.ControlError("logs", "timeout", None)
        worker = args[args.index("--worker") + 1]
        if failure == "same_id_stale":
            if not original_announcement:
                original_announcement.append(json.dumps({"source": "container",
                    "workerId": worker, "ts": datetime.now(timezone.utc).isoformat(),
                    "line": "ENVOY_PINNED_IDENTITY " + json.dumps(identity())}))
                return original_announcement[0]
            return original_announcement[0] + "\n" + "\n".join(record(worker, t)
                for t in ["vLLM is healthy", "All fitness checks passed."])
        if len(reads) == 1 or failure == "missing_fitness":
            return record(worker, "ENVOY_PINNED_IDENTITY " + json.dumps(identity()))
        return "\n".join(record(worker, t) for t in
                         ["vLLM is healthy", "All fitness checks passed."])
    def workers():
        if state["workersMin"]:
            active_reads.append(True)
        worker = "replacement" if failure == "replacement" and reads else "original"
        stale = failure == "same_id_stale" and len(active_reads) == 2
        return ([{"id": worker, "status": "IDLE", "isStale": stale}]
                if state["workersMin"] else [])
    def forbidden(*args, **kwargs):
        raise AssertionError("unverified worker cannot dispatch inference")
    monkeypatch.setattr(lifecycle, "STARTUP_POLL_SECONDS", 0.001)
    with pytest.raises(TimeoutError):
        lifecycle.supervise("fixture", tmp_path, [], key="dummy", control=control,
            worker_reader=workers, popen=forbidden, seconds=0.1, logged_running_ready=True)
    result = json.loads((tmp_path / "lifecycle.json").read_text())
    assert result["shutdown_verified"] and result["watchdog_fired"]
    assert result["exit_code"] is None and len(reads) > 3
    assert state["workersMin"] == state["workersMax"] == 0


def test_expired_deadline_permits_no_further_log_read():
    expired = threading.Event()
    expired.set()
    def forbidden(*args, **kwargs):
        raise AssertionError("expired session must not read logs")
    with pytest.raises(TimeoutError):
        lifecycle.identity_logs(forbidden, "fixture", {"worker"},
                                datetime.now(timezone.utc), expired)


def test_private_diagnostics_have_a_finite_disk_budget(tmp_path):
    audit = tmp_path / "reads.jsonl"
    private = tmp_path / "startup-log-output.jsonl"
    with private.open("wb") as stream:
        stream.truncate(8_000_000)
    with pytest.raises(ValueError, match="8 MB"):
        lifecycle.identity_logs(lambda *a, **k: "new logs", "fixture", {"worker"},
            datetime.now(timezone.utc), threading.Event(), audit=audit)
    assert private.stat().st_size == 8_000_000
