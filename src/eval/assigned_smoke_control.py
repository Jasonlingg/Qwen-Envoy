"""Bounded control reads; worker IDs come from the same v2 API as worker logs."""
from __future__ import annotations

import json
import re
import subprocess

from src.policies.runpod_queue import RunpodQueueTransport


class ControlError(RuntimeError):
    def __init__(self, command: str, code: str, status, *, log_output=""):
        self.details = {"command": command, "code": code, "status": status}
        # Private diagnostics, deliberately excluded from printable/public details.
        self.log_output = log_output
        super().__init__(f"Runpod {command} failed ({code}, HTTP {status})")


def cli(*args, raw=False):
    try:
        result = subprocess.run(["runpodctl", "serverless", *args], capture_output=True,
                                text=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        if len(output.encode()) > 1_000_000:
            raise ValueError("control CLI output too large") from None
        raise ControlError(args[0], "timeout", None,
            log_output=output if raw and args[0] == "logs" else "") from None
    if len(result.stdout.encode()) > 1_000_000:
        raise ValueError("control CLI output too large")
    if result.returncode:
        error = {}
        for line in reversed(result.stderr.splitlines()):
            try:
                candidate = json.loads(line)
            except ValueError:
                continue
            if isinstance(candidate, dict) and "code" in candidate:
                error = candidate
                break
        raise ControlError(args[0], error.get("code", "unknown"), error.get("status"),
            log_output=result.stdout if raw and args[0] == "logs" else "")
    return result.stdout if raw else json.loads(result.stdout)


class WorkerCatalog:
    """Read-only v2 worker lookup with no redirect, proxy, retry or inference path."""

    def __init__(self, endpoint: str, key: str, *, fetch=None):
        if not re.fullmatch(r"[a-z0-9]{1,64}", endpoint) or not key:
            raise ValueError("endpoint id and credential required")
        self.base, self._key = f"https://api.runpod.io/v2/serverless/{endpoint}", key
        # Reuse only the existing bounded fixed-host HTTP routine, not queue dispatch.
        self.fetch = fetch or (lambda: RunpodQueueTransport._request(self, "GET", "/workers"))

    def list(self) -> list[dict]:
        raw = self.fetch()
        if not isinstance(raw, dict) or not isinstance(raw.get("workers"), list):
            raise ValueError("v2 worker list must be an object with worker records")
        records, summary = raw["workers"], raw.get("summary", {})
        if (not isinstance(summary, dict) or type(summary.get("total")) is not int
                or summary["total"] != len(records) or len(records) > 64):
            raise ValueError("v2 worker count disagrees with its records")
        selected, seen = [], set()
        for row in records:
            wid = row.get("id") if isinstance(row, dict) else None
            if (not isinstance(wid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", wid)
                    or wid in seen or row.get("status") not in
                    {"INITIALIZING", "IDLE", "RUNNING", "THROTTLED", "UNHEALTHY"}
                    or type(row.get("isStale", False)) is not bool):
                raise ValueError("invalid or duplicate active v2 worker")
            seen.add(wid)
            selected.append({k: row[k] for k in ("id", "status", "startedAt", "image",
                "version", "gpuCount", "gpuTypeId", "dataCenterId", "isStale") if k in row})
        return selected
