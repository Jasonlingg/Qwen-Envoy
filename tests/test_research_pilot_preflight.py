import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

import pytest

from scripts.run_research_workflow_pilot import main
from src.eval.research_pilot_preflight import preflight, probe_models
from src.eval.research_pilot_runtime import MAX_MODEL_RESPONSE_BYTES, bounded_transport
from src.policies.openai_compatible import MODEL_USER_AGENT, OpenAICompatiblePolicy


@contextmanager
def endpoint(*, status=200, body=None, headers=None):
    seen = []
    payload = body if body is not None else json.dumps({"data": [{"id": "envoy"}]}).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def dispatch(self):
            seen.append({"method": self.command, "path": self.path,
                         "user_agent": self.headers.get("User-Agent"),
                         "authorization": self.headers.get("Authorization")})
            code, result = status, payload
            if self.headers.get("User-Agent", "").startswith("Python-urllib"):
                code, result = 403, b"error code: 1010\n"
            if self.command == "POST":
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                result = json.dumps({"choices": [{"message": {"content": "print('ok')"}}]})
                result = result.encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(result)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = dispatch  # noqa: N815 - stdlib request handler protocol
        do_POST = dispatch  # noqa: N815

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)


def settings(url, *, api_key_env="PILOT_SELECTED_KEY"):
    return {"endpoint": url, "model": "envoy", "api_key_env": api_key_env}


def test_real_http_policy_and_readiness_identify_the_same_project_client(monkeypatch):
    monkeypatch.setenv("PILOT_SELECTED_KEY", "selected-key")
    with endpoint() as (url, seen):
        # Demonstrate the failure being fixed, without a real cloud endpoint.
        with pytest.raises(error.HTTPError) as exc:
            request.urlopen(url + "/models", timeout=1)
        assert exc.value.code == 403
        result = probe_models(settings(url))
        policy = OpenAICompatiblePolicy(endpoint=url, model="envoy", api_key="selected-key")
        assert policy.act("Inspect") == "print('ok')"
    assert result["status"] == "ready"
    assert result["model_present"] is True
    assert [row["method"] for row in seen] == ["GET", "GET", "POST"]
    assert seen[1]["path"] == "/v1/models"
    assert seen[1]["authorization"] == seen[2]["authorization"] == "Bearer selected-key"
    assert seen[1]["user_agent"] == seen[2]["user_agent"] == MODEL_USER_AGENT


def test_bounded_runtime_uses_same_user_agent_without_retry():
    with endpoint() as (url, seen):
        response = bounded_transport(url + "/chat/completions", {}, {}, 1)
    assert response["choices"]
    assert len(seen) == 1 and seen[0]["user_agent"] == MODEL_USER_AGENT


def test_missing_selected_key_never_inherits_another_credential(monkeypatch):
    monkeypatch.setenv("ENVOY_MODEL_API_KEY", "wrong-provider-key")
    monkeypatch.delenv("PILOT_SELECTED_KEY", raising=False)
    with endpoint() as (url, seen):
        result = probe_models(settings(url))
    assert result["error"] == "missing_credentials"
    assert result["readiness_requests"] == 0
    assert seen == []
    assert "wrong-provider-key" not in json.dumps(result)


def test_explicit_anonymous_selection_does_not_inherit_generic_key(monkeypatch):
    monkeypatch.setenv("ENVOY_MODEL_API_KEY", "wrong-provider-key")
    with endpoint() as (url, seen):
        assert probe_models(settings(url, api_key_env=""))["status"] == "ready"
    assert seen[0]["authorization"] is None


def test_auth_failure_has_only_sanitized_diagnostics_and_no_body_or_key(monkeypatch):
    secret = "super-private-key"
    monkeypatch.setenv("PILOT_SELECTED_KEY", secret)
    with endpoint(status=403, body=f"error code: 1010\nBearer {secret}".encode(),
                  headers={"Server": "cloudflare " + secret}) as (url, seen):
        result = probe_models(settings(url))
    assert result["http_status"] == 403
    assert result["cloudflare_error_code"] == 1010
    assert result["error"] == "http_error"
    assert len(seen) == 1
    assert secret not in json.dumps(result)
    assert "Bearer" not in json.dumps(result)
    assert set(result) == {"status", "http_status", "server", "content_type",
                           "cloudflare_error_code", "model_present", "readiness_requests", "error"}


def test_redirect_is_not_followed_and_does_not_forward_credentials(monkeypatch):
    monkeypatch.setenv("PILOT_SELECTED_KEY", "selected-key")
    with endpoint() as (target, target_seen):
        with endpoint(status=302, headers={"Location": target + "/models"}) as (url, seen):
            result = probe_models(settings(url))
    assert result["error"] == "redirect_refused"
    assert result["http_status"] == 302
    assert len(seen) == 1 and target_seen == []


def test_incomplete_http_body_does_not_escape_as_raw_exception(monkeypatch):
    monkeypatch.setenv("PILOT_SELECTED_KEY", "selected-key")
    with endpoint(body=b"selected-key", headers={"Content-Length": "100"}) as (url, _seen):
        result = probe_models(settings(url))
    assert result["status"] == "not_ready"
    assert "selected-key" not in json.dumps(result)


@pytest.mark.parametrize("declared_size", [False, True])
def test_models_response_size_is_bounded(monkeypatch, declared_size):
    monkeypatch.setenv("PILOT_SELECTED_KEY", "selected-key")
    headers = {"Content-Length": str(MAX_MODEL_RESPONSE_BYTES + 1)} if declared_size else {}
    with endpoint(body=b"x" * (MAX_MODEL_RESPONSE_BYTES + 1), headers=headers) as (url, seen):
        result = probe_models(settings(url))
    assert result["error"] == "response_too_large"
    assert len(seen) == 1


@pytest.mark.parametrize(("body", "expected"), [
    (b'{"data": [{"id": "envoy-other"}]}', "model_missing"),
    (b'{"data": "envoy"}', "invalid_response"),
    (b'{"echo": "secret-but-not-json-model-list"}', "invalid_response"),
    (b"invalid JSON secret", "invalid_response"),
])
def test_exact_model_identity_and_response_shape_are_required(monkeypatch, body, expected):
    monkeypatch.setenv("PILOT_SELECTED_KEY", "selected-key")
    with endpoint(body=body) as (url, _seen):
        result = probe_models(settings(url))
    assert result["status"] == "not_ready" and result["error"] == expected
    assert "secret" not in json.dumps(result)


def test_preflight_cli_is_zero_inference_and_does_not_require_paid_run_ack(
        monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PILOT_SELECTED_KEY", "selected-key")
    with endpoint() as (url, seen):
        config = {"host": settings(url), "qwen": settings(url + "/chat/completions"),
                  "execution": {"billing_acknowledged": False}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        assert main(["preflight", "--config", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready" and result["model_requests"] == 0
    assert result["readiness_requests"] == 2
    assert all(row["method"] == "GET" and row["path"] == "/v1/models" for row in seen)
    assert "selected-key" not in json.dumps(result)


def test_preflight_cli_failure_returns_json_and_nonzero_exit(tmp_path, capsys):
    path = tmp_path / "invalid.json"
    path.write_text("secret raw invalid JSON")
    assert main(["preflight", "--config", str(path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "invalid_config" and result["model_requests"] == 0
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("bad", [None, {}, {"host": {}}, {"host": None, "qwen": {}}])
def test_invalid_config_fails_closed_without_requests(bad):
    result = preflight(bad)
    assert result["status"] == "not_ready"
    assert result["model_requests"] == result["readiness_requests"] == 0
