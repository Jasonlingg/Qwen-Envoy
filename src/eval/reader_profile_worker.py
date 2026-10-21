"""Opt-in telemetry around the pinned queue worker; never execute model output."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
from urllib import request
from uuid import uuid4

GAUGES = {"vllm:num_requests_running", "vllm:num_requests_waiting",
          "vllm:kv_cache_usage_perc"}
PHASES = {"vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds",
          "vllm:request_decode_time_seconds", "vllm:time_to_first_token_seconds"}
COUNTERS = {"vllm:num_preemptions_total", "vllm:num_preemptions"}


def canonical_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def transport_hash(payload):
    """Treat safe integral JSON numbers equally across language serializers."""
    def normalize(value):
        if type(value) is float and value.is_integer() and abs(value) <= 2 ** 53:
            return int(value)
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value
    return canonical_hash(normalize(payload))


def parse_metrics(body):
    values = {}
    wanted = GAUGES | COUNTERS | {p + suffix for p in PHASES for suffix in ("_sum", "_count")}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r'([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([^ ]+)(?:\s+\d+)?', line)
        if match and match[1] in wanted:
            value = float(match[2])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid profile metric")
            values[match[1]] = values.get(match[1], 0) + value
    if GAUGES - values.keys() or any(p + suffix not in values for p in PHASES
                                   for suffix in ("_sum", "_count")):
        raise ValueError("required engine telemetry missing")
    return values


def read_metrics():
    port = int(os.environ.get("VLLM_PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("invalid metrics port")
    # Fixed loopback path, no credentials, proxies, redirects or transport retry.
    class NoRedirect(request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    with opener.open(f"http://127.0.0.1:{port}/metrics", timeout=1) as response:
        body = response.read(1_000_001)
    if len(body) > 1_000_000:
        raise ValueError("metrics response too large")
    return parse_metrics(body.decode())


def gpu_memory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("profile needs exactly one visible GPU")
    used, total = [float(x.strip()) for x in lines[0].split(",")]
    if not math.isfinite(used) or not math.isfinite(total) or not 0 <= used <= total:
        raise ValueError("invalid GPU memory sample")
    return {"used_mib": used, "total_mib": total}


class ProfileHook:
    def __init__(self, session, payload_hashes, *, metrics=read_metrics, memory=gpu_memory):
        if (not isinstance(session, str) or not re.fullmatch(r"[a-f0-9]{32}", session)
                or len(payload_hashes) != 2 or any(not re.fullmatch(r"[a-f0-9]{64}", h)
                                                  for h in payload_hashes.values())):
            raise ValueError("profile needs one session and two pinned payloads")
        self.session, self.hashes = session, dict(payload_hashes)
        self.metrics, self.memory = metrics, memory
        self.boot_id = uuid4().hex
        self.lock = threading.Lock()
        self.reserved, self.admin_calls, self.active = 0, 0, 0
        self.window, self.failed = None, False

    def sample(self):
        return {"seconds": time.monotonic(), "metrics": self.metrics(), "memory": self.memory()}

    def control(self, operation):
        with self.lock:
            self.admin_calls += 1
            if self.admin_calls > 9 or self.failed or self.active:
                raise ValueError("profile control budget, state or active-request conflict")
            if operation == "preflight":
                if self.admin_calls != 1:
                    raise ValueError("preflight must be first and unique")
                return {"preflight": self.sample(), "memory_sampling": "nvidia-smi per snapshot",
                        "declared_config": {k: os.environ.get(k) for k in (
                            "MAX_NUM_SEQS", "MAX_CONCURRENCY", "MAX_MODEL_LEN",
                            "ENABLE_PREFIX_CACHING")}}
            if operation == "begin":
                if self.window is not None:
                    raise ValueError("telemetry window already active")
                self.window = {"before": self.sample(), "samples": [], "errors": []}
                stop = threading.Event()
                window = self.window
                def collect():
                    # Metrics only: avoid nvidia-smi subprocesses during timed inference.
                    while not stop.wait(0.1):
                        if len(window["samples"]) >= 1200:
                            window["errors"].append("telemetry sample budget exhausted")
                            return
                        try:
                            window["samples"].append({"seconds": time.monotonic(),
                                                       "metrics": self.metrics()})
                        except Exception as exc:
                            window["errors"].append(type(exc).__name__)
                            return
                thread = threading.Thread(target=collect, daemon=True)
                self.window.update(stop=stop, thread=thread)
                thread.start()
                return {"window_started": True, "before": window["before"]}
            if operation == "end" and self.window is not None:
                window, self.window = self.window, None
                window.pop("stop").set()
                thread = window.pop("thread")
                thread.join(timeout=2)
                if thread.is_alive():
                    raise ValueError("telemetry sampler did not stop")
                window["after"] = self.sample()
                return window
            raise ValueError("invalid profile operation or window state")

    def wrap(self, original):
        async def handler(job):
            number = None
            try:
                data = job["input"]
                profile = data["envoy_profile"]
                if (not isinstance(profile, dict) or profile.get("session") != self.session
                        or set(profile) - {"session", "operation", "sample_id"}):
                    raise ValueError("invalid profile session")
                if profile.get("operation") != "infer":
                    if set(data) != {"envoy_profile"}:
                        raise ValueError("control calls cannot dispatch a model")
                    result = await asyncio.to_thread(self.control, profile["operation"])
                    yield {"profile": {"boot_id": self.boot_id, **result}}
                    return
                if set(data) != {"envoy_profile", "openai_route", "openai_input"}:
                    raise ValueError("invalid inference envelope")
                payload = data["openai_input"]
                if data["openai_route"] != "/v1/chat/completions":
                    raise ValueError("profile route differs from the pinned route")
                received = transport_hash(payload)
                if received != self.hashes.get(profile.get("sample_id")):
                    raise ValueError("profile input differs from pinned payload; received="
                                     + received)
                with self.lock:
                    if self.failed or self.admin_calls == 0 or self.reserved >= 10:
                        raise ValueError("profile request budget or state invalid")
                    self.reserved += 1
                    self.active += 1
                    number = self.reserved
                response = []
                forwarded = {**job, "input": {k: v for k, v in data.items()
                                             if k != "envoy_profile"}}
                async for item in original(forwarded):
                    response.append(item)
                if (len(response) != 1 or not isinstance(response[0], dict)
                        or "error" in response[0]):
                    raise ValueError("unexpected upstream profile response")
                yield {"profile": {"boot_id": self.boot_id, "request_number": number},
                       "response": response[0]}
            except Exception as exc:
                with self.lock:
                    self.failed = True
                yield {"error": {"type": type(exc).__name__, "message": str(exc)[:300]},
                       "profile": {"boot_id": self.boot_id, "request_number": number}}
            finally:
                if number is not None:
                    with self.lock:
                        self.active -= 1
        return handler
