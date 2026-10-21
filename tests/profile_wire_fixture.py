"""Loopback provider and scripted model for the real profile protocol; no GPU."""
from __future__ import annotations

import asyncio
import http.client
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from src.eval.reader_profile import ProfileQueueClient, assess_profile, run_profile
from src.eval.reader_profile_worker import (
    GAUGES,
    PHASES,
    ProfileHook,
    canonical_hash,
    transport_hash,
)
from tests.assigned_smoke_cli_fixture import identity


class ProfileWireService:
    def __init__(self, *, payloads=None, prompt_tokens=None, corrupt=False):
        self.payloads = payloads or {key: {"model": "envoy", "temperature": 0.0,
            "top_p": 1.0, "seed": 42, "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": key + " text 1.0 λ"}]}
            for key in ("sample_a", "sample_b")}
        self.plan = {"session": "a" * 32,
            "payload_hashes": {k: canonical_hash(v) for k, v in self.payloads.items()},
            "prompt_tokens": prompt_tokens or {k: 10 for k in self.payloads}}
        self.corrupt = corrupt
        self.jobs, self.events, self.forwarded = {}, [], []
        self.completed, self.log_reads = 0, 0
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=2)
        def metrics():
            with self.lock:
                count = self.completed
            return {**{k: 0.0 for k in GAUGES},
                    "vllm:num_requests_running": float(self.hook.active),
                    **{p + "_count": float(count) for p in PHASES},
                    **{p + "_sum": float(count) * 0.55 for p in PHASES}}
        self.hook = ProfileHook(self.plan["session"],
            {k: transport_hash(v) for k, v in self.payloads.items()}, metrics=metrics,
            memory=lambda: {"used_mib": 18000, "total_mib": 24564})
        async def scripted_model(job):
            # Deliberately long enough for sampled overlap and the polling margin.
            # These delays are fixture behavior, never GPU performance evidence.
            with self.lock:
                self.forwarded.append(job["input"]["openai_input"])
            await asyncio.sleep(0.55)
            payload = job["input"]["openai_input"]
            sample = next(k for k, v in self.payloads.items()
                          if v["messages"] == payload["messages"])
            with self.lock:
                self.completed += 1
            yield {"model": "envoy", "choices": [{"finish_reason": "stop",
                   "message": {"content": "scripted response; do not execute"}}],
                   "usage": {"prompt_tokens": self.plan["prompt_tokens"][sample],
                             "completion_tokens": 2}}
        self.handler = self.hook.wrap(scripted_model)
        fixture = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def record(self):
                fixture.events.append((self.command, self.path,
                                       self.headers.get("Authorization")))
                return self.headers.get("Authorization") == "Bearer fixture-only"
            def reply(self, value, status=200):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_GET(self):
                if not self.record():
                    return self.reply({}, 401)
                if self.path == "/v2/serverless/fixture/workers/worker/logs?tail=5000":
                    fixture.log_reads += 1
                    if fixture.log_reads <= 2:
                        fixture.stop.wait(0.5)  # Delay headers to exercise startup recovery.
                        return
                    stamp = datetime.now(timezone.utc).isoformat()
                    lines = [{"source": "container", "ts": "2000-01-01T00:00:00Z",
                              "line": 'ENVOY_PINNED_IDENTITY {"adapter_revision":"old"}'},
                             {"source": "system", "ts": stamp,
                              "line": 'ENVOY_PINNED_IDENTITY {"adapter_revision":"wrong"}'}]
                    lines += [{"source": "container", "ts": stamp, "line": text} for text in [
                        "ENVOY_PINNED_IDENTITY " + json.dumps(identity()),
                        "vLLM is healthy", "All fitness checks passed."]]
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for line in lines:
                        self.wfile.write(("data: " + json.dumps(line) + "\n\n").encode())
                    self.wfile.flush()
                    return
                if self.path.startswith("/v2/fixture/status/"):
                    with fixture.lock:
                        job = dict(fixture.jobs[self.path.rsplit("/", 1)[-1]])
                    return self.reply(job)
                self.reply({}, 404)
            def do_POST(self):
                if not self.record():
                    return self.reply({}, 401)
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                # Independent JSON intermediary: 0.0/1.0 become integral JSON numbers.
                def number(text):
                    value = Decimal(text)
                    return int(value) if value == value.to_integral_value() else float(value)
                body = json.loads(raw, parse_float=number)
                if self.path == "/v2/fixture/run":
                    with fixture.lock:
                        jid = "job-" + str(len(fixture.jobs) + 1)
                        fixture.jobs[jid] = {"id": jid, "status": "IN_PROGRESS"}
                    fixture.executor.submit(fixture.execute, jid, body["input"])
                    return self.reply({"id": jid, "status": "IN_PROGRESS"})
                if self.path.startswith("/v2/fixture/cancel/"):
                    return self.reply({"status": "CANCELLED"})
                self.reply({}, 404)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.base = f"http://127.0.0.1:{self.server.server_port}/v2/fixture"

    def connection(self):
        return http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=1)

    def execute(self, jid, data):
        if self.corrupt and data["envoy_profile"]["operation"] == "infer":
            data["openai_input"]["messages"][0]["content"] += " altered"
        async def collect():
            return [row async for row in self.handler({"id": jid, "input": data})]
        rows = asyncio.run(collect())
        with self.lock:
            self.jobs[jid] = {"id": jid, "status": "FAILED" if "error" in rows[0]
                             else "COMPLETED", "output": rows, "delayTime": 0,
                             "executionTime": 1}
            if "error" in rows[0]:
                self.jobs[jid]["error"] = rows[0]["error"]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.executor.shutdown(wait=True)
        if self.hook.window:
            self.hook.window["stop"].set()
            self.hook.window["thread"].join(timeout=2)


def run_child(base, config_path, output):
    """Actual child process using the production client/hook protocol over HTTP."""
    url = urlsplit(base)
    if url.scheme != "http" or url.hostname != "127.0.0.1" or url.path != "/v2/fixture":
        raise ValueError("fixture must stay on loopback")
    config, output = json.loads(Path(config_path).read_text()), Path(output)
    def request(method, path, body=None):
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=3)
        try:
            conn.request(method, url.path + path,
                body=json.dumps(body) if body is not None else None,
                headers={"Authorization": "Bearer fixture-only",
                         "Content-Type": "application/json"})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError("fixture HTTP failure")
            return json.loads(response.read(1_000_001))
        finally:
            conn.close()
    def save(value):
        (output / "component-result.json").write_text(json.dumps(value, indent=2) + "\n")
    client = ProfileQueueClient("fixture", "fixture-only", config["plan"], config["payloads"],
                                output / "component.requests.jsonl", http=request)
    try:
        result = run_profile(client, list(config["payloads"]), save)
        result["assessment"] = assess_profile(result)
        result["scripted_local_component_check"] = True
        save(result)
        if result["status"] != "completed_pending_comparability_check":
            raise RuntimeError("local component request flow failed")
        if not result["assessment"]["comparable"]:
            raise RuntimeError("scripted fixture did not exercise matched overlapping batches")
        try:
            client.call("infer", next(iter(config["payloads"])))
        except ValueError:
            result["extra_request_blocked"] = True
        save(result)
    finally:
        client.audit.close()
