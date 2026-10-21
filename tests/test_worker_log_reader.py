"""Real loopback HTTP/SSE tests for the profile's fixed-host startup reader."""
import http.client
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.eval.assigned_smoke_control import ControlError
from src.eval.worker_log_reader import MAX_BYTES, WorkerLogReader


def entry(**changes):
    return {"source": "container", "line": "ready", "ts": "2026-10-21T00:00:00Z", **changes}


def frame(row):
    return ("data: " + json.dumps(row) + "\n\n").encode()


@contextmanager
def log_service(body, *, status=200, delay=0, content_type="text/event-stream", drip=False):
    events = []
    stop = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            events.append((self.command, self.path, self.headers.get("Authorization")))
            if stop.wait(delay):
                return
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                if status == 302:
                    self.send_header("Location", "/must-not-follow")
                self.end_headers()
                if drip:
                    while not stop.wait(0.01):
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                else:
                    self.wfile.write(body)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def factory():
            return http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=1)
        yield factory, events
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_sse_snapshot_handles_comments_crlf_multiline_and_worker_binding():
    body = (b": comment\r\n\r\nid: 1\r\ndata: {" +
        b'"source":"container",\r\ndata: "line":"ready",' +
        b'"ts":"2026-10-21T00:00:00Z"}\r\n\r\n')
    with log_service(body) as (factory, events):
        text = WorkerLogReader("fixture", "dummy-only", connection=factory).read("worker")
    assert json.loads(text) == {**entry(), "workerId": "worker"}
    assert events == [("GET", "/v2/serverless/fixture/workers/worker/logs?tail=5000",
                       "Bearer dummy-only")]


def test_default_connection_is_fixed_https_and_does_not_consult_proxy_env(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "https://untrusted.invalid")
    conn = WorkerLogReader("fixture", "dummy").connection()
    assert isinstance(conn, http.client.HTTPSConnection) and conn.host == "api.runpod.io"
    conn.close()  # No connection was opened.


@pytest.mark.parametrize("status,code", [(302, "bad_response"), (401, "unauthorized"),
    (403, "forbidden"), (404, "not_found"), (429, "rate_limited"), (503, "server_error")])
def test_http_failures_are_classified_and_never_followed_or_retried(status, code):
    with log_service(b"private error", status=status) as (factory, events):
        with pytest.raises(ControlError) as exc:
            WorkerLogReader("fixture", "dummy", connection=factory).read("worker")
        assert exc.value.details == {"command": "logs", "code": code, "status": status}
        assert len(events) == 1 and "private" not in str(exc.value)


@pytest.mark.parametrize("body,content_type", [
    (frame(entry(workerId="foreign")), "text/event-stream"),
    (frame(entry(truncated=True)), "text/event-stream"),
    (frame(entry(ts="not a timestamp")), "text/event-stream"),
    (frame(entry(ts="2026-10-21T00:00:00")), "text/event-stream"),
    (b"data: invalid json\n\n", "text/event-stream"),
    (b"data: []\n\n", "text/event-stream"),
    (b"<html>proxy</html>", "text/html"),
    (b":" + b"x" * MAX_BYTES, "text/event-stream")])
def test_foreign_malformed_or_oversized_stream_fails_closed(body, content_type):
    with log_service(body, content_type=content_type) as (factory, _):
        with pytest.raises(ControlError) as exc:
            WorkerLogReader("fixture", "dummy", connection=factory).read("worker")
        assert exc.value.details["code"] == "invalid_stream"


def test_unterminated_sse_frame_cannot_establish_readiness():
    with log_service(frame(entry()).rstrip(b"\n")) as (factory, _):
        assert WorkerLogReader("fixture", "dummy", connection=factory).read("worker") == ""


def test_quiet_header_timeout_and_continuous_keepalive_are_bounded():
    with log_service(b"", delay=1) as (factory, events):
        with pytest.raises(ControlError) as exc:
            WorkerLogReader("fixture", "dummy", connection=factory, seconds=0.05).read("worker")
        assert exc.value.details["code"] == "timeout" and len(events) == 1
    with log_service(b"", drip=True) as (factory, _):
        started = time.monotonic()
        assert WorkerLogReader("fixture", "dummy", connection=factory, seconds=0.1).read(
            "worker") == ""
        assert time.monotonic() - started < 1.5
