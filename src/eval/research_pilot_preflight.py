"""Zero-inference endpoint checks with bounded, credential-free diagnostics."""

from __future__ import annotations

import json
import os
import re
from http.client import HTTPException
from urllib import error, request

from src.eval.research_pilot_runtime import MAX_MODEL_RESPONSE_BYTES, validate_endpoint
from src.policies.openai_compatible import MODEL_USER_AGENT


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _safe_header(value, secret):
    if not isinstance(value, str):
        return None
    if secret:
        value = value.replace(secret, "[redacted]")
    return "".join(char for char in value[:120] if 32 <= ord(char) < 127)


def probe_models(settings, *, timeout=10.0):
    """GET only /models; advertised identity does not attest remote weights."""
    result = {"status": "not_ready", "http_status": None, "server": None,
              "content_type": None, "cloudflare_error_code": None,
              "model_present": False, "readiness_requests": 0, "error": None}
    if not isinstance(settings, dict):
        return {**result, "error": "invalid_config"}
    endpoint, model, key_env = (settings.get(name) for name in
                                ("endpoint", "model", "api_key_env"))
    try:
        validate_endpoint(endpoint)
    except (ValueError, TypeError):
        return {**result, "error": "invalid_config"}
    if not isinstance(model, str) or not model.strip() or not isinstance(key_env, str):
        return {**result, "error": "invalid_config"}
    # An empty selection explicitly means no credential. Never inherit the
    # generic ENVOY_MODEL_API_KEY or another provider's key.
    secret = os.environ.get(key_env) if key_env else None
    if key_env and (not secret or not secret.strip()):
        return {**result, "error": "missing_credentials"}
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[:-len("/chat/completions")]
    headers = {"Accept": "application/json", "User-Agent": MODEL_USER_AGENT}
    if secret:
        headers["Authorization"] = "Bearer " + secret
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
    req = request.Request(base + "/models", headers=headers, method="GET")
    result["readiness_requests"] = 1
    try:
        try:
            response = opener.open(req, timeout=timeout)
        except error.HTTPError as exc:
            response = exc
        with response:
            result.update(http_status=response.status,
                          server=_safe_header(response.headers.get("Server"), secret),
                          content_type=_safe_header(response.headers.get("Content-Type"), secret))
            size = response.headers.get("Content-Length")
            if size is not None and int(size) > MAX_MODEL_RESPONSE_BYTES:
                return {**result, "error": "response_too_large"}
            body = response.read(MAX_MODEL_RESPONSE_BYTES + 1)
    except (OSError, ValueError, HTTPException, error.URLError):
        return {**result, "error": "connection_failed"}
    if len(body) > MAX_MODEL_RESPONSE_BYTES:
        return {**result, "error": "response_too_large"}
    if result["http_status"] != 200:
        # Retain no body excerpts: a server can echo request credentials.
        code = re.search(rb"\berror code:\s*(\d{4})\b", body[:500])
        if code:
            result["cloudflare_error_code"] = int(code.group(1))
        return {**result, "error": ("redirect_refused" if 300 <= result["http_status"] < 400
                                    else "http_error")}
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        return {**result, "error": "invalid_response"}
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return {**result, "error": "invalid_response"}
    result["model_present"] = any(isinstance(item, dict) and item.get("id") == model
                                  for item in data)
    if not result["model_present"]:
        return {**result, "error": "model_missing"}
    return {**result, "status": "ready"}


def preflight(config):
    """Check both selected endpoints without inference, Docker, or infra calls."""
    if not isinstance(config, dict) or not all(role in config for role in ("host", "qwen")):
        return {"status": "not_ready", "error": "invalid_config", "model_requests": 0,
                "readiness_requests": 0, "models": {}}
    models = {role: probe_models(config[role]) for role in ("host", "qwen")}
    return {"status": "ready" if all(row["status"] == "ready" for row in models.values())
            else "not_ready", "model_requests": 0,
            "readiness_requests": sum(row["readiness_requests"] for row in models.values()),
            "user_agent": MODEL_USER_AGENT, "models": models}
