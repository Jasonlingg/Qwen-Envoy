"""Bounded model HTTP for research MCP and evaluation; no redirects or retries."""

from __future__ import annotations

import json
from http.client import HTTPException
from urllib import error, request
from urllib.parse import urlsplit

from src.policies.openai_compatible import MODEL_USER_AGENT

MAX_MODEL_RESPONSE_BYTES = 1_000_000


def validate_endpoint(endpoint):
    if not isinstance(endpoint, str):
        raise ValueError("Model endpoint must be a URL")
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        raise ValueError("Model endpoint must be a valid URL") from None
    if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.scheme not in {"http", "https"}
            or (parsed.scheme == "http"
                and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})):
        raise ValueError("Use authenticated HTTPS or a loopback HTTP tunnel; no URL credentials")
    return endpoint


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _ResponseTooLargeError(Exception):
    pass


def bounded_transport(url, payload, headers, timeout):
    validate_endpoint(url)
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
    model_headers = {**headers, "User-Agent": MODEL_USER_AGENT}
    req = request.Request(url, data=json.dumps(payload).encode(), headers=model_headers,
                          method="POST")
    try:
        with opener.open(req, timeout=timeout) as response:
            size = response.headers.get("Content-Length")
            if size is not None and int(size) > MAX_MODEL_RESPONSE_BYTES:
                raise _ResponseTooLargeError
            body = response.read(MAX_MODEL_RESPONSE_BYTES + 1)
    except error.HTTPError as exc:
        code = exc.code
        exc.close()
        if 300 <= code < 400:
            raise ValueError("Model endpoint redirects are forbidden") from None
        # Error bodies and exception reasons can echo credentials or prompts.
        raise RuntimeError(f"Model endpoint HTTP {code}; no retry") from None
    except _ResponseTooLargeError:
        raise ValueError("Model response exceeds byte limit") from None
    except (OSError, ValueError, HTTPException, error.URLError):
        raise RuntimeError("Model endpoint connection failed; no retry") from None
    if len(body) > MAX_MODEL_RESPONSE_BYTES:
        raise ValueError("Model response exceeds byte limit")
    try:
        result = json.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("Model response is invalid JSON") from None
    if not isinstance(result, dict):
        raise ValueError("Model response is not an object")
    return result
