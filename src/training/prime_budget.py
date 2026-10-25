"""Single-run persistent token reservations in front of the local native engine."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import httpx

from src.training.prime_run import write

ENGINE = "http://127.0.0.1:8000"


def loopback_engine(url):
    """An explicit local engine or SSH local forward, never a public endpoint."""
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment or not parsed.port):
        raise ValueError("native engine must be an explicit HTTP 127.0.0.1 port")
    return url.rstrip("/")


class Budget:
    def __init__(self, output, limits, deadline, policy):
        self.path, self.limits = output / "dispatches.json", limits
        self.deadline, self.policy, self.lock = deadline, policy, threading.Lock()
        self.state = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"requests": [], "reserved_tokens": 0, "closed": False}
        )
        if any(r["status"] != "returned" for r in self.state["requests"]):
            self.state["closed"] = True

    def reserve(self, payload):
        params, ids = payload["sampling_params"], payload["token_ids"]
        if (
            payload["model"] != "envoy"
            or params["max_tokens"] != 1024
            or params.get("temperature") != 1
            or params.get("top_p") != 1
            or params.get("top_k") != -1
            or params.get("min_p") != 0
            or params.get("n", 1) != 1
            or payload.get("stream", False) is not False
            or not isinstance(ids, list)
            or not ids
            or any(type(t) is not int or t < 0 for t in ids)
            or len(ids) > 7168
        ):
            raise ValueError("unexpected native sampling contract")
        tokens = len(ids) + 1024
        with self.lock:
            if (
                self.state["closed"]
                or time.time() >= self.deadline
                or len(self.state["requests"]) >= self.limits["requests"]
                or self.state["reserved_tokens"] + tokens > self.limits["reserved_tokens"]
            ):
                raise RuntimeError("whole-run request/token/time budget exhausted")
            ordinal = len(self.state["requests"]) + 1
            self.state["requests"].append(
                {
                    "ordinal": ordinal,
                    "tokens": tokens,
                    "policy": self.policy,
                    "status": "reserved_no_refund",
                }
            )
            self.state["reserved_tokens"] += tokens
            write(self.path, self.state)
            return ordinal

    def finish(self, ordinal, healthy):
        with self.lock:
            self.state["requests"][ordinal - 1]["status"] = (
                "returned" if healthy else "failed_or_uncertain"
            )
            self.state["closed"] |= not healthy
            write(self.path, self.state)


@contextmanager
def proxy(output, plan, policy, *, engine=None):
    """Only two native routes; no credentials, proxy environment or redirects."""
    deadline = json.loads((output / "started.json").read_text())["deadline"]
    upstream = loopback_engine(ENGINE if engine is None else engine)
    budget = Budget(output, plan["limits"], deadline, policy)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def forward(self, payload=None):
            ordinal = None
            try:
                if payload is not None:
                    ordinal = budget.reserve(payload)
                    write(output / f"engine-request-{ordinal:04}.json", payload)
                timeout = min(25, deadline - time.time())
                if timeout <= 0:
                    raise TimeoutError("run expired")
                with httpx.Client(
                    trust_env=False, follow_redirects=False, timeout=timeout
                ) as client:
                    with client.stream(
                        self.command,
                        upstream + self.path,
                        **({"json": payload} if payload is not None else {}),
                    ) as r:
                        raw = bytearray()
                        for chunk in r.iter_bytes():
                            raw.extend(chunk)
                            if len(raw) > 2 * 1024 * 1024 or time.time() >= deadline:
                                raise ValueError("engine response size/time bound exceeded")
                        r.raise_for_status()
                        json.loads(raw)
                if ordinal is not None:
                    (output / f"engine-response-{ordinal:04}.json").write_bytes(raw)
                    budget.finish(ordinal, True)
                self.reply(200, raw)
            except Exception as error:
                if ordinal is not None:
                    budget.finish(ordinal, False)
                self.reply(502, json.dumps({"error": type(error).__name__}).encode())

        def do_GET(self):
            if self.path == "/v1/models":
                self.forward()
            else:
                self.reply(404, b"{}")

        def do_POST(self):
            if self.path != "/inference/v1/generate":
                return self.reply(404, b"{}")
            try:
                self.connection.settimeout(5)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("request size/encoding")
                payload = json.loads(self.rfile.read(size))
            except Exception:
                return self.reply(400, b"{}")
            self.forward(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
