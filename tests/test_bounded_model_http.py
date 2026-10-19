import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.eval.research_pilot_runtime import bounded_transport as pilot_transport
from src.policies.bounded_http import MAX_MODEL_RESPONSE_BYTES, bounded_transport
from src.policies.openai_compatible import MODEL_USER_AGENT, OpenAICompatiblePolicy, _post_json
from src.product.arxiv_mcp_config import build_reader, build_reasoner


@contextmanager
def origin(*, status=200, body=b"{}", headers=None):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def dispatch(self):
            calls.append({"method": self.command,
                          "authorization": self.headers.get("Authorization"),
                          "user_agent": self.headers.get("User-Agent")})
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(status)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = dispatch  # noqa: N815 - stdlib handler protocol
        do_POST = dispatch  # noqa: N815

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize("status", [302, 307])
def test_cross_origin_redirect_never_receives_a_request_or_authorization(status):
    with origin() as (second, second_calls):
        with origin(status=status, headers={"Location": second}) as (first, first_calls):
            with pytest.raises(ValueError, match="redirects are forbidden"):
                bounded_transport(first, {"model": "test"}, {"Authorization": "Bearer private"}, 1)
    assert len(first_calls) == 1
    assert first_calls[0]["authorization"] == "Bearer private"
    assert second_calls == []


@pytest.mark.parametrize("declare_size", [False, True])
def test_model_response_size_is_bounded_with_or_without_length(declare_size):
    body = b"x" * (MAX_MODEL_RESPONSE_BYTES + 1)
    headers = {"Content-Length": str(len(body))} if declare_size else {}
    with origin(body=body, headers=headers) as (url, calls):
        with pytest.raises(ValueError, match="byte limit"):
            bounded_transport(url, {}, {}, 1)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [401, 403, 500])
def test_http_errors_never_echo_secret_response_bodies_and_never_retry(status):
    secret = "sensitive-api-key-or-prompt"
    with origin(status=status, body=secret.encode()) as (url, calls):
        with pytest.raises(RuntimeError) as exc:
            bounded_transport(url, {}, {"Authorization": "Bearer " + secret}, 1)
    assert str(exc.value) == f"Model endpoint HTTP {status}; no retry"
    assert secret not in str(exc.value)
    assert len(calls) == 1


@pytest.mark.parametrize("body", [b"invalid secret JSON", b"\xff",
                                  b"[" * 1500 + b"0" + b"]" * 1500])
def test_malformed_or_deep_json_becomes_a_safe_validation_error(body):
    with origin(body=body) as (url, calls):
        with pytest.raises(ValueError, match="^Model response is invalid JSON$") as exc:
            bounded_transport(url, {}, {}, 1)
    assert "secret" not in str(exc.value)
    assert len(calls) == 1


def test_transport_ignores_environment_proxies_and_uses_truthful_identity(monkeypatch):
    with origin() as (proxy_url, proxy_calls):
        with origin(body=b'{"ok":true}') as (url, calls):
            for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
                monkeypatch.setenv(name, proxy_url)
            monkeypatch.setenv("NO_PROXY", "")
            monkeypatch.setenv("no_proxy", "")
            assert bounded_transport(url, {}, {}, 1) == {"ok": True}
    assert proxy_calls == []
    assert len(calls) == 1 and calls[0]["user_agent"] == MODEL_USER_AGENT


@pytest.mark.parametrize("kind", ["qwen", "nemotron"])
def test_mcp_reader_uses_shared_bounded_transport_and_keeps_injection(monkeypatch, kind):
    monkeypatch.setenv("NEBIUS_API_KEY", "selected")
    kwargs = {"qwen_endpoint": "http://127.0.0.1:8000/v1", "qwen_model": "envoy",
              "qwen_checkpoint": "pinned"}
    reader = build_reader(kind, **kwargs)
    assert reader.policy_factory()._transport is bounded_transport
    assert reader.policy_factory().system_prompt == reader.system_prompt
    custom = lambda *_args: {}  # noqa: E731
    injected = build_reader(kind, policy_transport=custom, **kwargs)
    assert injected.policy_factory()._transport is custom


def test_mcp_reasoner_uses_shared_bounded_transport_and_keeps_injection(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "selected")
    assert build_reasoner("nemotron").transport is bounded_transport
    custom = lambda *_args: {}  # noqa: E731
    assert build_reasoner("nemotron", policy_transport=custom).transport is custom
    assert build_reasoner("none") is None


def test_mcp_reader_default_makes_a_bounded_authenticated_request(monkeypatch):
    monkeypatch.setenv("ENVOY_MODEL_API_KEY", "selected")
    body = json.dumps({"choices": [{"message": {"content": "print('ok')"}}]}).encode()
    with origin(body=body) as (url, calls):
        reader = build_reader("qwen", qwen_endpoint=url, qwen_model="envoy",
                              qwen_checkpoint="pinned")
        assert reader.policy_factory().act("Inspect source") == "print('ok')"
    assert len(calls) == 1 and calls[0]["authorization"] == "Bearer selected"
    assert calls[0]["user_agent"] == MODEL_USER_AGENT


def test_pilot_reexports_shared_transport_and_generic_policy_default_is_preserved():
    assert pilot_transport is bounded_transport
    policy = OpenAICompatiblePolicy(endpoint="http://localhost/v1", model="local")
    assert policy._transport is _post_json
