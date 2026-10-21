"""One-worker live-session guard; control CLI calls never submit inference."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.eval.assigned_smoke_cleanup import cleanup_containers, cleanup_jobs, wait_parked
from src.eval.assigned_smoke_control import ControlError, WorkerCatalog, cli
from src.eval.assigned_smoke_plan import MODEL_IDENTITY, ROOT, write_json
from src.eval.native_pilot_client import kill_group

STARTUP_POLL_SECONDS = 5


def configuration(endpoint: dict) -> dict:
    # Deliberately exclude environment/worker records, which may contain secrets.
    names = ("id", "workersMin", "workersMax", "idleTimeout", "gpuIds", "gpuCount",
             "minCudaVersion", "executionTimeoutMs", "templateId")
    return {k: endpoint.get(k) for k in names}


def worker_identity(logs: str, *, active_workers: set[str], since: datetime) -> dict | None:
    found = None
    for line in logs.splitlines():
        item = json.loads(line)
        worker = item.get("workerId")
        stamp = item.get("ts")
        if (item.get("source") != "container" or worker not in active_workers
                or not isinstance(stamp, str)
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
            found = {**identity, "worker_id": worker, "identity_log_ts": stamp}
    return found


def identity_logs(control, endpoint, active, since, expired, *, audit=None):
    """One bounded observation; the outer watchdog owns transient-read retries."""
    if expired.is_set():
        raise TimeoutError("watchdog expired before reading startup logs")
    event = {"worker_id": next(iter(active)), "since": since.isoformat(),
             "max_wait_seconds": 5, "source": "both"}
    logs = ""
    try:
        logs = control("logs", endpoint, "--source", "both",
            "--worker", next(iter(active)), "--since", since.isoformat(),
            "--max-wait", "5s", raw=True)
        event["status"] = "received"
        return logs
    except ControlError as exc:
        logs = exc.log_output
        transient = exc.details["code"] in {
            "not_found", "network_error", "server_error", "rate_limited", "timeout"}
        event.update(status="pending" if transient else "error", error=exc.details)
        if not transient:
            raise
        # A killed CLI may end mid-JSON-line. Preserve all bytes privately, but
        # only complete records may contribute to identity/readiness.
        return logs[:logs.rfind("\n") + 1]
    finally:
        if audit is not None:
            # Raw logs stay private under out/, separate from safe public read metadata.
            if logs:
                raw = (json.dumps({"worker_id": event["worker_id"], "output": logs})
                       + "\n").encode()
                private = audit.with_name("startup-log-output.jsonl")
                if (private.stat().st_size if private.exists() else 0) + len(raw) > 8_000_000:
                    raise ValueError("startup log diagnostics exceeded 8 MB")
                fd = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "ab") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                event.update(output_sha256=hashlib.sha256(logs.encode()).hexdigest(),
                             output_bytes=len(logs.encode()))
            with audit.open("a") as stream:
                stream.write(json.dumps(event) + "\n")


def logged_readiness(logs, active, since):
    """Bind both readiness events to the current worker and activation window."""
    text = []
    for raw in logs.splitlines():
        item = json.loads(raw)
        stamp = item.get("ts")
        if (item.get("source") == "container" and item.get("workerId") in active
                and isinstance(stamp, str)
                and datetime.fromisoformat(stamp.replace("Z", "+00:00")) >= since):
            text.append(item.get("line", ""))
    return (any("vLLM is healthy" in line for line in text)
            and any("All fitness checks passed." in line for line in text))


def supervise(endpoint: str, output: Path, command: list[str], *, key: str,
              control=cli, popen=subprocess.Popen, seconds=900, worker_reader=None,
              logged_running_ready=False, deadline_unix=None) -> dict:
    worker_reader = worker_reader or WorkerCatalog(endpoint, key).list
    before = configuration(control("get", endpoint))
    health = control("health", endpoint)
    if (before["workersMin"] != 0 or before["workersMax"] != 0
            or before["gpuIds"] != ["NVIDIA RTX A5000"] or before["gpuCount"] != 1
            or health["jobs"]["inQueue"] or health["jobs"]["inProgress"]
            or any(health["workers"].values()) or worker_reader()):
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

    if deadline_unix is not None:
        seconds = min(seconds, deadline_unix - time.time())
        if seconds <= 0:
            raise TimeoutError("original session deadline elapsed before activation")
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
        log_worker = None
        evidence_since = since
        startup_lines = {}
        while not expired.is_set():
            health = control("health", endpoint)
            write_json(output / "health-latest.json", health)
            workers = worker_reader()
            write_json(output / "workers-latest.json", workers)
            active = {worker["id"] for worker in workers}
            outcome["startup_worker_ids"] = sorted(active)
            if len(active) > 1:
                raise ValueError("more than one active worker; refusing inference")
            current = next(iter(active), None)
            if current != log_worker:
                startup_lines.clear()
                log_worker = current
                evidence_since = since
            identity = None
            # A newly scheduled worker can be listed before its log service exists.
            usable = all(w["status"] in {"IDLE", "RUNNING"} and not w.get("isStale")
                         for w in workers)
            if not usable:
                startup_lines.clear()
                evidence_since = datetime.now(timezone.utc)
            running = (logged_running_ready and len(workers) == 1
                       and workers[0]["status"] == "RUNNING"
                       and health["workers"].get("running", 0)
                       and not health["jobs"]["inQueue"] and not health["jobs"]["inProgress"])
            ready = health["workers"].get("ready", 0)
            if active and usable and (ready or running):
                logs = identity_logs(control, endpoint, active, evidence_since, expired,
                                     audit=output / "startup-log-reads.jsonl")
                for line in logs.splitlines():
                    startup_lines.setdefault(line, None)
                if sum(len(line.encode()) + 1 for line in startup_lines) > 4_000_000:
                    raise ValueError("startup evidence exceeded 4 MB")
                accumulated = "\n".join(startup_lines)
                identity = worker_identity(accumulated, active_workers=active, since=evidence_since)
                if logged_running_ready:
                    ready = (ready or running) and logged_readiness(
                        accumulated, active, evidence_since)
            if identity and ready:
                write_json(output / "worker-identity.json", identity)
                break
            expired.wait(STARTUP_POLL_SECONDS)
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
        if isinstance(exc, ControlError):
            outcome["control_error"] = exc.details
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
            config, health = wait_parked(control, endpoint, worker_reader=worker_reader)
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
