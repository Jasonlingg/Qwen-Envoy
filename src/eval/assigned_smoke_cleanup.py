"""Cancel only this session's jobs/containers; verify endpoint shutdown."""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

from src.policies.runpod_queue import RunpodQueueTransport


def cleanup_jobs(output: Path, endpoint: str, key: str):
    client = object.__new__(RunpodQueueTransport)
    client.base, client._key = f"https://api.runpod.ai/v2/{endpoint}", key
    results = []
    for path in output.glob("*.requests.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for job in {r["job_id"] for r in rows if r.get("job_id")}:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", job):
                raise ValueError("invalid audit job id")
            state = client._request("GET", "/status/" + job)
            if state.get("status") in {"IN_QUEUE", "IN_PROGRESS"}:
                client._request("POST", "/cancel/" + job, {})
                state = client._request("GET", "/status/" + job)
            results.append({"job_id": job, "status": state.get("status")})
    return results


def cleanup_containers(output: Path):
    path = output / "containers.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            cid = json.loads(line)["id"]
            if not re.fullmatch(r"[a-f0-9]{64}", cid):
                raise ValueError("invalid owned Docker container id")
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=15)


def wait_parked(control, endpoint, *, seconds=90):
    deadline = time.monotonic() + seconds
    while True:
        config = control("get", endpoint, "--include-workers")
        health = control("health", endpoint)
        if (config.get("workersMin") == config.get("workersMax") == 0
                and not config.get("workers") and not any(health["workers"].values())
                and not health["jobs"]["inQueue"] and not health["jobs"]["inProgress"]):
            return config, health
        if time.monotonic() >= deadline:
            raise RuntimeError("endpoint shutdown not verified within 90 seconds")
        time.sleep(min(5, max(0, deadline - time.monotonic())))
