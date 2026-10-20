"""One-worker live-session guard; control CLI calls never submit inference."""
from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.eval.assigned_smoke_cleanup import cleanup_containers, cleanup_jobs, wait_parked
from src.eval.assigned_smoke_plan import MODEL_IDENTITY, ROOT, write_json
from src.eval.native_pilot_client import kill_group


def cli(*args, raw=False):
    result = subprocess.run(["runpodctl", "serverless", *args], capture_output=True,
                            text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Runpod control CLI failed; inspect endpoint before any retry")
    if len(result.stdout) > 1_000_000:
        raise ValueError("control CLI output too large")
    return result.stdout if raw else json.loads(result.stdout)


def configuration(endpoint: dict) -> dict:
    # Deliberately exclude environment/worker records, which may contain secrets.
    names = ("id", "workersMin", "workersMax", "idleTimeout", "gpuIds", "gpuCount",
             "minCudaVersion", "executionTimeoutMs", "templateId")
    return {k: endpoint.get(k) for k in names}


def worker_identity(logs: str, *, active_workers: set[str], since: datetime) -> dict | None:
    for line in logs.splitlines():
        item = json.loads(line)
        worker = item.get("workerId")
        stamp = item.get("ts")
        if (worker not in active_workers or not isinstance(stamp, str)
                or datetime.fromisoformat(stamp.replace("Z", "+00:00")) < since):
            continue
        text = item.get("line", "")
        if "ENVOY_PINNED_IDENTITY " in text:
            identity, _ = json.JSONDecoder().raw_decode(text.split("ENVOY_PINNED_IDENTITY ", 1)[1])
            expected = {"base_repo": MODEL_IDENTITY["base"],
                "base_revision": MODEL_IDENTITY["base_revision"],
                "adapter_repo": MODEL_IDENTITY["adapter"],
                "adapter_revision": MODEL_IDENTITY["adapter_revision"],
                "adapter_sha256": MODEL_IDENTITY["adapter_sha256"],
                "adapter_subfolder": MODEL_IDENTITY["subfolder"], "served_adapter": "envoy"}
            if (any(identity.get(k) != v for k, v in expected.items())
                    or identity.get("gpu") != "NVIDIA RTX A5000"):
                raise ValueError("live worker identity differs from the approved pins")
            return {**identity, "worker_id": worker, "identity_log_ts": stamp}
    return None


def supervise(endpoint: str, output: Path, command: list[str], *, key: str,
              control=cli, popen=subprocess.Popen, seconds=900) -> dict:
    before = configuration(control("get", endpoint))
    health = control("health", endpoint)
    if (before["workersMin"] != 0 or before["workersMax"] != 0
            or before["gpuIds"] != ["NVIDIA RTX A5000"] or before["gpuCount"] != 1
            or health["jobs"]["inQueue"] or health["jobs"]["inProgress"]
            or any(health["workers"].values())):
        raise ValueError("endpoint must be parked, empty and pinned to one A5000")
    write_json(output / "endpoint-before.json", {"configuration": before, "health": health})
    child = None
    expired = threading.Event()
    mutation_lock = threading.Lock()
    started = time.monotonic()
    since = datetime.now(timezone.utc)

    def park():
        with mutation_lock:
            control("update", endpoint, "--workers-min", "0", "--workers-max", "0")

    def stop():
        expired.set()
        try:
            with mutation_lock:
                if child is not None:
                    kill_group(child.pid)
                control("update", endpoint, "--workers-min", "0", "--workers-max", "0")
        except Exception:
            pass  # The main finally records/retries cleanup and its read-back.

    timer = threading.Timer(seconds, stop)
    timer.daemon = True
    timer.start()  # Armed BEFORE activation, including cold start.
    outcome = {"watchdog_seconds": seconds, "exit_code": None}
    try:
        with mutation_lock:
            if expired.is_set():
                raise TimeoutError("watchdog expired before activation")
            control("update", endpoint, "--workers-min", "1", "--workers-max", "1")
        after = configuration(control("get", endpoint))
        expected = {**before, "workersMin": 1, "workersMax": 1}
        if after != expected:
            raise ValueError("activation read-back changed unexpected settings")
        write_json(output / "endpoint-activated.json", after)
        identity = None
        while not expired.is_set():
            health = control("health", endpoint)
            write_json(output / "health-latest.json", health)
            workers = control("get", endpoint, "--include-workers").get("workers") or []
            active = {worker["id"] for worker in workers}
            if len(active) > 1:
                raise ValueError("more than one active worker; refusing inference")
            identity = None
            if active:
                logs = control("logs", endpoint, "--source", "container", "--tail", "2000",
                    "--worker", next(iter(active)), "--since", since.isoformat(),
                    "--max-wait", "1s", raw=True)
                identity = worker_identity(logs, active_workers=active, since=since)
            if identity and health["workers"].get("ready", 0):
                write_json(output / "worker-identity.json", identity)
                break
            expired.wait(5)
        if expired.is_set():
            raise TimeoutError("watchdog expired during worker warmup")
        with mutation_lock:
            if expired.is_set():
                raise TimeoutError("watchdog expired before model worker")
            with (output / "worker.stdout.log").open("w") as stdout:
                with (output / "worker.stderr.log").open("w") as stderr:
                    child = popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                  start_new_session=True)
        outcome["exit_code"] = child.wait()
    except BaseException as exc:
        outcome["error_type"] = type(exc).__name__
        raise
    finally:
        if child is not None:
            kill_group(child.pid)
            child.wait()
        timer.cancel()
        timer.join()
        outcome.update(watchdog_fired=expired.is_set(), seconds=time.monotonic() - started)
        errors = []
        for _ in range(3):
            try:
                park()
                break
            except Exception as exc:
                errors.append(type(exc).__name__)
        try:
            outcome["job_cleanup"] = cleanup_jobs(output, endpoint, key)
        except Exception as exc:
            errors.append("job_cleanup:" + type(exc).__name__)
        try:
            cleanup_containers(output)
            config, health = wait_parked(control, endpoint)
            outcome["endpoint_after"] = configuration(config)
            outcome["health_after"] = health
            outcome["shutdown_verified"] = True
        except Exception as exc:
            errors.append("read_back:" + type(exc).__name__)
        outcome["cleanup_errors"] = errors
        write_json(output / "lifecycle.json", outcome)
        if errors:
            raise RuntimeError("session cleanup failed; inspect lifecycle.json")
    if outcome["watchdog_fired"] or outcome["exit_code"] != 0:
        raise RuntimeError("live worker did not complete successfully")
    return outcome
