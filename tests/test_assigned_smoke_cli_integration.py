"""Actual CLI and HTTP/SSE parsing, entirely on loopback with scripted worker logs."""
import json
import shutil
import subprocess
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import pytest

from src.eval import assigned_smoke_control as control
from src.eval import assigned_smoke_lifecycle as lifecycle
from src.eval.assigned_smoke_lifecycle import worker_identity
from tests.assigned_smoke_cli_fixture import LocalLogs, identity

pytestmark = pytest.mark.skipif(shutil.which("runpodctl") is None, reason="runpodctl unavailable")


def test_actual_cli_selects_v2_worker_and_materializes_current_identity():
    with LocalLogs() as fixture:
        def fetch():
            request = Request(fixture.base + "/serverless/fixture/workers",
                              headers={"Authorization": "Bearer fixture-only-key"})
            with urlopen(request, timeout=3) as response:
                return json.load(response)
        catalog = control.WorkerCatalog("fixture", "fixture-only-key", fetch=fetch).list()
        result = subprocess.run([shutil.which("runpodctl"), "serverless", "logs", "fixture",
            "--source", "container", "--since", "2026-10-20T00:00:00Z", "--max-wait", "5s"],
            env=fixture.environment(), capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        found = worker_identity(result.stdout, active_workers={w["id"] for w in catalog},
                                since=datetime(2026, 10, 20, tzinfo=timezone.utc))
        assert found["worker_id"] == "current-v2" and found["adapter_sha256"] == identity()[
            "adapter_sha256"]
        assert all(method == "GET" and key == "Bearer fixture-only-key"
                   for method, _, key in fixture.events)
        assert any(path.startswith("/v2/serverless/fixture/workers/current-v2/logs?")
                   for _, path, _ in fixture.events)


def test_real_cli_one_second_read_times_out_but_five_second_read_succeeds(monkeypatch):
    with LocalLogs() as fixture:
        fixture.delay = 1.3
        run = subprocess.run
        binary = shutil.which("runpodctl")
        def isolated(command, **kwargs):
            assert command[0] == "runpodctl"
            return run([binary, *command[1:]], env=fixture.environment(), **kwargs)
        monkeypatch.setattr(control.subprocess, "run", isolated)
        args = ("logs", "fixture", "--worker", "current-v2", "--source", "container",
                "--since", "2026-10-20T00:00:00Z", "--max-wait")
        with pytest.raises(control.ControlError) as error:
            control.cli(*args, "1s", raw=True)
        assert error.value.details["code"] == "timeout"
        logs = control.cli(*args, "5s", raw=True)
        assert json.loads(logs)["workerId"] == "current-v2"
        assert all(method == "GET" for method, _, _ in fixture.events)


def test_real_cli_quiet_startup_then_split_identity_and_readiness(tmp_path, monkeypatch):
    """Three no-header timeouts must not kill startup or lose earlier identity."""
    with LocalLogs() as fixture:
        fixture.silent_reads = 3
        def line(text):
            return {"source": "container", "ts": "2099-01-01T00:00:00Z", "line": text}
        fixture.payload_batches = [
            [line("ENVOY_PINNED_IDENTITY " + json.dumps(identity()))],
            [line("vLLM is healthy")], [line("All fitness checks passed.")]]
        run = subprocess.run
        binary = shutil.which("runpodctl")
        def isolated(command, **kwargs):
            assert command[0] == "runpodctl" and command[2] == "logs"
            command = [binary, *command[1:]]
            command[command.index("--max-wait") + 1] = "150ms"
            return run(command, env=fixture.environment(), **kwargs)
        monkeypatch.setattr(control.subprocess, "run", isolated)
        monkeypatch.setattr(lifecycle, "STARTUP_POLL_SECONDS", 0.001)
        monkeypatch.setattr(lifecycle, "kill_group", lambda _: None)
        state = {"id": "fixture", "workersMin": 0, "workersMax": 0,
                 "gpuCount": 1, "gpuIds": ["NVIDIA RTX A5000"]}
        def commands(command, endpoint, *args, **kwargs):
            if command == "get":
                return dict(state)
            if command == "update":
                state.update(workersMin=int(args[1]), workersMax=int(args[3]))
                return dict(state)
            if command == "health":
                return {"jobs": {"inQueue": 0, "inProgress": 0},
                        "workers": {"running": state["workersMin"], "ready": 0}}
            return control.cli(command, endpoint, *args, **kwargs)
        def workers():
            return ([{"id": "current-v2", "status": "RUNNING"}]
                    if state["workersMin"] else [])
        started = []
        class Child:
            pid = 123
            def wait(self):
                return 0
        def launch(*args, **kwargs):
            assert fixture.payload_batches == []  # All three conditions have arrived.
            started.append(True)
            return Child()
        outcome = lifecycle.supervise("fixture", tmp_path, [], key="dummy-only",
            control=commands, worker_reader=workers, popen=launch, seconds=5,
            logged_running_ready=True)
        reads = [json.loads(x) for x in
                 (tmp_path / "startup-log-reads.jsonl").read_text().splitlines()]
        assert [r["status"] for r in reads] == ["pending"] * 3 + ["received"] * 3
        assert all("source=" not in path for _, path, _ in fixture.events)
        assert started == [True] and outcome["shutdown_verified"]
        assert all(method == "GET" and key == "Bearer fixture-only-key"
                   for method, _, key in fixture.events)
