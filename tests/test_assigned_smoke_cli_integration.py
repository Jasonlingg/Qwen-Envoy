"""Actual CLI and HTTP/SSE parsing, entirely on loopback with scripted worker logs."""
import json
import shutil
import subprocess
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import pytest

from src.eval import assigned_smoke_control as control
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
