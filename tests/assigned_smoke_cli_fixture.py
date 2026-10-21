"""Local-only v2 worker/log service for the installed Runpod CLI; fake credentials."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.eval.assigned_smoke_plan import MODEL_IDENTITY


def identity():
    return {"base_repo": MODEL_IDENTITY["base"], "base_revision": MODEL_IDENTITY["base_revision"],
        "adapter_repo": MODEL_IDENTITY["adapter"],
        "adapter_revision": MODEL_IDENTITY["adapter_revision"],
        "adapter_sha256": MODEL_IDENTITY["adapter_sha256"],
        "adapter_subfolder": MODEL_IDENTITY["subfolder"], "served_adapter": "envoy",
        "gpu": "NVIDIA RTX A5000"}


class LocalLogs:
    def __init__(self):
        self.events, self.delay = [], 0
        self.silent_reads, self.payload_batches = 0, None
        self.release = threading.Event()
        fixture = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                fixture.events.append((self.command, self.path, self.headers.get("Authorization")))
                if self.path == "/v2/serverless/fixture/workers":
                    body = json.dumps({"summary": {"total": 1}, "workers": [
                        {"id": "current-v2", "status": "IDLE"}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/v2/serverless/fixture/workers/current-v2/logs?"):
                    if fixture.silent_reads:
                        fixture.silent_reads -= 1
                        fixture.release.wait(5)  # No headers: same observable CLI timeout as prod.
                        return
                    time.sleep(fixture.delay)
                    payload = {"source": "container", "ts": "2026-10-20T12:00:00Z",
                               "line": "ENVOY_PINNED_IDENTITY " + json.dumps(identity())}
                    payloads = (fixture.payload_batches.pop(0)
                                if fixture.payload_batches is not None else [payload])
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        for payload in payloads:
                            self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # Expected when the short-read reproduction times out.
                else:
                    self.send_response(404)
                    self.end_headers()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.base = f"http://127.0.0.1:{self.server.server_port}/v2"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def environment(self):
        # All control/invoke URLs stay local, with a dummy key; no real credential is inherited.
        inherited = {k: os.environ[k] for k in ("HOME", "USERPROFILE") if k in os.environ}
        return {**inherited, "RUNPOD_API_KEY": "fixture-only-key", "RUNPOD_REST_V2_URL": self.base,
                "RUNPOD_API_URL": self.base, "RUNPOD_INVOKE_URL": self.base,
                "RUNPOD_GRAPHQL_URL": self.base}
