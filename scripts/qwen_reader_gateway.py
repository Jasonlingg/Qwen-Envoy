"""Bounded authenticated gateway; expose port 8000, keep vLLM on loopback:8001."""

from __future__ import annotations

import argparse
import hmac
import http.client
import ipaddress
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BYTES = 1_000_000
CHAT_FIELDS = {"model", "messages", "max_tokens", "temperature", "top_p", "seed",
               "chat_template_kwargs", "n", "stream"}


def _token(name):
    value = os.environ.get(name, "")
    if not value.strip() or not value.isascii() or "\r" in value or "\n" in value:
        raise ValueError(f"{name} must contain a nonempty ASCII token without newlines")
    return value


def _chat_body(body):
    payload = json.loads(body)
    if not isinstance(payload, dict) or set(payload) - CHAT_FIELDS:
        raise ValueError("unsupported request fields")
    count = payload.get("max_tokens")
    if type(count) is not int or not 1 <= count <= 1024:
        raise ValueError("max_tokens must be 1..1024")
    if type(payload.get("n", 1)) is not int or payload.get("n", 1) != 1:
        raise ValueError("n must be 1")
    if payload.get("stream", False) is not False:
        raise ValueError("stream must be false")
    payload.update(n=1, stream=False)
    return json.dumps(payload, allow_nan=False).encode()


def make_server(*, host="0.0.0.0", port=8000, backend_host="127.0.0.1", backend_port=8001,
                token_env="VLLM_API_KEY", backend_token_env=None, max_requests=96):
    """Create one gateway; its atomic request allowance lasts until process exit."""
    if not ipaddress.ip_address(backend_host).is_loopback:
        raise ValueError("backend_host must be a literal loopback address")
    if type(backend_port) is not int or not 1 <= backend_port <= 65535:
        raise ValueError("backend_port must be 1..65535")
    if type(max_requests) is not int or max_requests < 0:
        raise ValueError("max_requests must be a nonnegative integer")
    token, backend_token = _token(token_env), _token(backend_token_env or token_env)
    lock, reserved = threading.Lock(), 0

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_args):
            pass  # Never log authorization headers, request bodies, or backend details.

        def respond(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            if len(body) > MAX_BYTES:
                status, body = 502, b'{"error":"response_too_large"}'
            self.close_connection = True
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def fail(self, status, code):
            self.respond(status, {"error": code})

        def send_error(self, code, message=None, explain=None):
            self.fail(code, "unsupported_request")

        def dispatch(self):
            nonlocal reserved
            supplied = self.headers.get_all("Authorization", [])
            if len(supplied) != 1 or not hmac.compare_digest(
                    supplied[0].encode(), ("Bearer " + token).encode()):
                return self.fail(401, "unauthorized")
            if (self.command, self.path) not in {("GET", "/health"), ("GET", "/v1/models"),
                                                 ("POST", "/v1/chat/completions")}:
                return self.fail(404, "not_found")
            lengths = self.headers.get_all("Content-Length", [])
            try:
                if len(lengths) > 1 or self.headers.get("Transfer-Encoding"):
                    raise ValueError
                count = int(lengths[0]) if lengths else 0
                if not 0 <= count <= MAX_BYTES:
                    return self.fail(413, "request_too_large")
                if self.command == "GET" and count:
                    raise ValueError
            except ValueError:
                return self.fail(400, "invalid_framing")
            if self.path == "/health":
                with lock:
                    used = reserved
                return self.respond(200, {"gateway": "qwen-reader-gateway-v1", "status": "ok",
                                          "reserved_requests": used, "max_requests": max_requests,
                                          "model_readiness": "not_checked"})
            body = None
            if self.command == "POST":
                try:
                    raw = self.rfile.read(count)
                    if len(raw) != count:
                        raise ValueError
                    body = _chat_body(raw)
                    if len(body) > MAX_BYTES:
                        return self.fail(413, "request_too_large")
                except (ValueError, TypeError, UnicodeError, RecursionError):
                    return self.fail(400, "invalid_chat_request")
                except (OSError, TimeoutError):
                    return self.fail(408, "request_timeout")
                with lock:
                    exhausted = reserved >= max_requests
                    if not exhausted:
                        reserved += 1  # Reserve before dispatch; never refund failures.
                if exhausted:
                    return self.fail(429, "request_budget_exhausted")
            connection = http.client.HTTPConnection(backend_host, backend_port, timeout=90)
            try:
                connection.request(self.command, self.path, body=body, headers={
                    "Content-Type": "application/json", "Authorization": "Bearer " + backend_token})
                response = connection.getresponse()
                if 300 <= response.status < 400:
                    return self.fail(502, "upstream_redirect_refused")
                if response.status >= 400:
                    return self.fail(502, "upstream_http_error")
                result = response.read(MAX_BYTES + 1)
                if len(result) > MAX_BYTES:
                    return self.fail(502, "response_too_large")
                payload = json.loads(result)
                if not isinstance(payload, dict):
                    return self.fail(502, "invalid_upstream_response")
                return self.respond(response.status, payload)
            except (OSError, http.client.HTTPException):
                return self.fail(503, "model_not_ready")
            except (ValueError, TypeError, UnicodeError, RecursionError):
                return self.fail(502, "invalid_upstream_response")
            finally:
                connection.close()

        do_GET = do_POST = dispatch  # noqa: N815
        do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = dispatch  # noqa: N815

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8001)
    parser.add_argument("--token-env", default="VLLM_API_KEY")
    parser.add_argument("--backend-token-env")
    parser.add_argument("--max-requests", type=int, default=96)
    args = parser.parse_args()
    try:
        server = make_server(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
