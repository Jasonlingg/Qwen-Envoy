"""Automatic startup -> real child -> JSON queue -> hook -> cleanup on loopback."""
import json
import os
import subprocess
import sys
from copy import deepcopy

import pytest

from scripts import profile_two_readers as profile
from src.eval.assigned_smoke_lifecycle import supervise
from src.eval.worker_log_reader import WorkerLogReader
from src.policies.runpod_queue import RunpodQueueTransport
from tests.profile_wire_fixture import ProfileWireService


@pytest.mark.parametrize("corrupt", [False, True])
def test_automatic_startup_and_full_request_flow_across_numeric_json_boundary(
        tmp_path, monkeypatch, corrupt):
    for key, value in profile.CONFIG.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("src.eval.assigned_smoke_lifecycle.STARTUP_POLL_SECONDS", 0.001)
    with ProfileWireService(corrupt=corrupt) as fixture:
        config = tmp_path / "fixture-config.json"
        config.write_text(json.dumps({"plan": fixture.plan, "payloads": fixture.payloads}))
        command = [sys.executable, "-c", "from tests.profile_wire_fixture import run_child; "
            "import sys; run_child(*sys.argv[1:])", fixture.base, str(config), str(tmp_path)]
        state = {"id": "fixture", "workersMin": 0, "workersMax": 0,
                 "gpuIds": ["NVIDIA RTX A5000"], "gpuCount": 1}
        def control(operation, endpoint, *args, **kwargs):
            if operation == "get":
                return dict(state)
            if operation == "update":
                state.update(workersMin=int(args[1]), workersMax=int(args[3]))
                return dict(state)
            if operation == "health":
                return {"jobs": {"inQueue": 0, "inProgress": 0},
                        "workers": {"ready": 0, "running": state["workersMin"]}}
            raise AssertionError("The automatic profile must never request CLI logs")
        def workers():
            return [{"id": "worker", "status": "RUNNING"}] if state["workersMin"] else []
        def cleanup_request(client, method, path, body=None):
            assert client.base == "https://api.runpod.ai/v2/fixture"
            conn = fixture.connection()
            try:
                conn.request(method, "/v2/fixture" + path,
                    body=json.dumps(body) if body is not None else None,
                    headers={"Authorization": "Bearer fixture-only"})
                response = conn.getresponse()
                assert response.status == 200
                return json.loads(response.read())
            finally:
                conn.close()
        monkeypatch.setattr(RunpodQueueTransport, "_request", cleanup_request)
        def launch(args, **kwargs):
            return subprocess.Popen(args, **kwargs, env={"PATH": os.environ["PATH"],
                "RUNPOD_API_KEY": "fixture-only", "PYTHONNOUSERSITE": "1"})
        reader = WorkerLogReader("fixture", "fixture-only", connection=fixture.connection,
                                 seconds=0.05, quiet_seconds=0.05)
        args = dict(key="fixture-only", control=control, worker_reader=workers,
            log_reader=reader, popen=launch, seconds=20, logged_running_ready=True)
        if corrupt:
            with pytest.raises(RuntimeError, match="did not complete"):
                supervise("fixture", tmp_path, command, **args)
        else:
            supervise("fixture", tmp_path, command, **args)
        lifecycle = json.loads((tmp_path / "lifecycle.json").read_text())
        result = json.loads((tmp_path / "component-result.json").read_text())
        reads = [json.loads(x) for x in (tmp_path / "startup-log-reads.jsonl").read_text(
            ).splitlines()]
        assert [r["status"] for r in reads] == ["pending", "pending", "received"]
        assert all(r["reader"] == "https_sse" for r in reads)
        assert lifecycle["shutdown_verified"] and not lifecycle["watchdog_fired"]
        assert state["workersMin"] == state["workersMax"] == 0
        posts = [path for method, path, _ in fixture.events if method == "POST"]
        assert all(auth == "Bearer fixture-only" for _, _, auth in fixture.events)
        if corrupt:
            assert lifecycle["exit_code"] == 1 and len(fixture.forwarded) == 0
            assert result["model_requests_reserved"] == 1 and len(posts) == 3  # Incl. cancel.
            rows = [json.loads(x) for x in (tmp_path / "component.requests.jsonl").read_text(
                ).splitlines()]
            failed = next(r["job"] for r in rows if r["event"] == "terminal_failure")
            assert "received=" in failed["error"]["message"]
        else:
            assert lifecycle["exit_code"] == 0 and result["extra_request_blocked"]
            assert result["model_requests_reserved"] == len(fixture.forwarded) == 10
            assert len(result["warmups"]) == 2 and len(result["batches"]) == 4
            assert len(posts) == len(lifecycle["job_cleanup"]) == 19
            assert all(type(p["temperature"]) is int and type(p["top_p"]) is int
                       for p in fixture.forwarded)
            assert all("1.0 λ" in p["messages"][0]["content"] for p in fixture.forwarded)


def test_profile_guard_selects_direct_log_reader(tmp_path, monkeypatch):
    original = ["python3", "-u", "-c",
                (profile.ROOT / "scripts/runpod_qwen_bootstrap.py").read_text()]
    state = {"entrypoint": original, "workers": {"min": 0, "max": 0},
        "env": profile.CONFIG, "image":
        "registry.runpod.net/runpod-workers-worker-vllm-main-dockerfile:b7212acc6"}
    def api(client, method, path, body=None):
        if method == "PATCH":
            state.update(deepcopy(body))
        return deepcopy(state)
    monkeypatch.setattr(RunpodQueueTransport, "_request", api)
    monkeypatch.setattr(profile, "MARKER", tmp_path / "attempt")
    (tmp_path / "plan.json").write_text("{}")
    (tmp_path / "entrypoint.py").write_text("# operator fixture")
    calls = []
    def supervisor(*args, **kwargs):
        reader = kwargs["log_reader"]
        assert isinstance(reader, WorkerLogReader) and reader.endpoint == profile.ENDPOINT
        calls.append(True)
    profile.guarded_run(tmp_path, {}, {}, key="fixture-only", supervisor=supervisor,
                        control=lambda *a: None)
    assert calls == [True] and state["entrypoint"] == original
