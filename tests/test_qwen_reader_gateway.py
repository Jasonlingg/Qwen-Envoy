"""Real local HTTP contract tests; no model, external network, or cloud calls."""

from __future__ import annotations

import http.client
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts import qwen_reader_gateway as gateway


@contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture
def local(monkeypatch):
    calls = []
    behavior = {"status": 200, "body": b'{"ok":true}'}

    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def dispatch(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append((self.command, self.path, self.headers.get("Authorization"), body))
            self.send_response(behavior["status"])
            self.send_header("Location", "/secret-route")
            self.send_header("Content-Length", str(len(behavior["body"])))
            self.end_headers()
            self.wfile.write(behavior["body"])

        do_GET = do_POST = dispatch  # noqa: N815

    monkeypatch.setenv("TEST_GATEWAY_TOKEN", "public-secret")
    monkeypatch.setenv("TEST_BACKEND_TOKEN", "backend-secret")
    with serving(ThreadingHTTPServer(("127.0.0.1", 0), Backend)) as backend:
        server = gateway.make_server(host="127.0.0.1", port=0, backend_port=backend.server_port,
                                     token_env="TEST_GATEWAY_TOKEN",
                                     backend_token_env="TEST_BACKEND_TOKEN", max_requests=2)
        with serving(server):
            yield server, calls, behavior


def call(server, method="GET", path="/health", *, body=None, token="public-secret", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        request_headers = {"Authorization": "Bearer " + token}
        request_headers.update(headers or {})
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def chat(**changes):
    return json.dumps({"model": "envoy", "messages": [{"role": "user", "content": "Hi"}],
                       "max_tokens": 1024, **changes}).encode()


def test_auth_health_and_path_filtering_never_call_backend(local, capsys):
    server, calls, _ = local
    assert call(server, token="wrong")[0] == 401
    status, health = call(server)
    assert status == 200 and health["gateway"] == "qwen-reader-gateway-v1"
    assert health["reserved_requests"] == 0 and health["model_readiness"] == "not_checked"
    for method, path in [("GET", "/metrics"), ("POST", "/tokenize"),
                         ("POST", "/health"), ("GET", "/v1/models?x=1"),
                         ("PUT", "/v1/chat/completions")]:
        assert call(server, method, path)[0] == 404
    assert calls == []
    assert capsys.readouterr() == ("", "")


def test_models_and_chat_forward_only_backend_token(local):
    server, calls, _ = local
    assert call(server, path="/v1/models") == (200, {"ok": True})
    assert call(server, "POST", "/v1/chat/completions", body=chat())[0] == 200
    assert [row[:3] for row in calls] == [
        ("GET", "/v1/models", "Bearer backend-secret"),
        ("POST", "/v1/chat/completions", "Bearer backend-secret")]
    payload = json.loads(calls[1][3])
    assert payload["n"] == 1 and payload["stream"] is False
    assert call(server)[1]["reserved_requests"] == 1


@pytest.mark.parametrize("body", [b"[]", b"broken", chat(max_tokens=1025), chat(max_tokens=True),
                                  chat(n=2), chat(n=True), chat(stream=True),
                                  chat(max_completion_tokens=99999), chat(best_of=2)])
def test_invalid_chat_is_rejected_before_dispatch_or_reservation(local, body):
    server, calls, _ = local
    assert call(server, "POST", "/v1/chat/completions", body=body)[0] == 400
    assert calls == [] and call(server)[1]["reserved_requests"] == 0


def test_request_and_response_size_bounds_and_invalid_framing(local):
    server, calls, behavior = local
    assert call(server, "POST", "/v1/chat/completions", body=b"",
                headers={"Content-Length": str(gateway.MAX_BYTES + 1)})[0] == 413
    assert call(server, "POST", "/v1/chat/completions", body=chat(),
                headers={"Transfer-Encoding": "chunked"})[0] == 400
    assert calls == []
    behavior["body"] = b"x" * (gateway.MAX_BYTES + 1)
    assert call(server, path="/v1/models") == (502, {"error": "response_too_large"})
    behavior["body"] = json.dumps({"ok": "\u00e9" * 200_000}, ensure_ascii=False).encode()
    assert len(behavior["body"]) < gateway.MAX_BYTES
    assert call(server, path="/v1/models") == (502, {"error": "response_too_large"})


@pytest.mark.parametrize("status,code", [(302, "upstream_redirect_refused"),
                                        (500, "upstream_http_error")])
def test_upstream_errors_are_sanitized_without_redirect_or_retry(local, status, code):
    server, calls, behavior = local
    behavior.update(status=status, body=b'private backend-secret trace')
    assert call(server, "POST", "/v1/chat/completions", body=chat()) == (502, {"error": code})
    assert len(calls) == 1 and call(server)[1]["reserved_requests"] == 1


def test_request_budget_is_atomic_and_failed_attempts_count(local):
    server, calls, behavior = local
    behavior.update(status=500)
    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: call(server, "POST", "/v1/chat/completions",
                                               body=chat())[0], range(8)))
    assert sorted(statuses) == [429] * 6 + [502] * 2
    assert len(calls) == 2 and call(server)[1]["reserved_requests"] == 2


def test_connection_failure_consumes_budget_without_retry(monkeypatch):
    monkeypatch.setenv("TEST_GATEWAY_TOKEN", "public-secret")
    unused = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = unused.server_port
    server = gateway.make_server(host="127.0.0.1", port=0, backend_port=port,
                                 token_env="TEST_GATEWAY_TOKEN", max_requests=1)
    unused.server_close()
    with serving(server):
        assert call(server, "POST", "/v1/chat/completions", body=chat()) == (
            503, {"error": "model_not_ready"})
        assert call(server)[1]["reserved_requests"] == 1
        assert call(server, "POST", "/v1/chat/completions", body=chat())[0] == 429


@pytest.mark.parametrize("value", ["", " ", "line\nbreak", "nonascii\u00e9"])
def test_missing_or_unsafe_token_fails_before_binding(monkeypatch, value):
    monkeypatch.setenv("TEST_GATEWAY_TOKEN", value)
    with pytest.raises(ValueError, match="nonempty ASCII token"):
        gateway.make_server(token_env="TEST_GATEWAY_TOKEN")


def test_non_loopback_backend_is_rejected():
    with pytest.raises(ValueError, match="loopback"):
        gateway.make_server(backend_host="192.0.2.1")
