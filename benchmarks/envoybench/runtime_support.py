"""Model configuration, sandbox preflight, and safe run-file utilities."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from benchmarks.envoybench.frozen_dataset import _read_json

SANDBOX_IMAGE = "rlm-sandbox"
SANDBOX_LABEL = "org.envoybench.sandbox=v1"
MODEL_SCHEMA = "envoybench-models-v1"
_MODEL_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SECRET_WORDS = {"api_key", "apikey", "authorization", "bearer", "password", "secret", "token"}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(word in str(key).lower() for word in _SECRET_WORDS):
                return True
            if _contains_secret_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def _endpoint(spec: dict) -> tuple[str, dict]:
    env_name = spec.get("endpoint_env")
    if env_name is not None and (not isinstance(env_name, str) or not env_name):
        raise ValueError("endpoint_env must name a non-empty environment variable")
    endpoint = os.environ.get(env_name, "") if env_name else spec.get("endpoint", "")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError(f"{spec['key']}: endpoint is missing; set {env_name or 'endpoint'}")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"{spec['key']}: endpoint must be an HTTP(S) URL without credentials/query"
        )
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme == "http" and not local:
        raise ValueError(f"{spec['key']}: non-local endpoint must use HTTPS")
    # The URL may contain a private hostname; store its hash, never the raw URL.
    metadata = {
        "endpoint_env": env_name,
        "endpoint_sha256": _sha256_text(endpoint.rstrip("/")),
        "endpoint_kind": "local" if local else "remote",
    }
    return endpoint, metadata


def load_models(path: Path, selected: list[str] | None = None) -> list[dict]:
    """Resolve endpoints and secrets in memory, retaining only safe metadata."""
    config = _read_json(path)
    if config.get("schema_version") != MODEL_SCHEMA:
        raise ValueError(f"model config schema_version must be {MODEL_SCHEMA}")
    specs = config.get("models")
    if not isinstance(specs, list) or not specs:
        raise ValueError("model config needs a non-empty models list")
    keys = [item.get("key") for item in specs if isinstance(item, dict)]
    if (
        len(keys) != len(specs)
        or any(not isinstance(key, str) or not _MODEL_KEY.fullmatch(key) for key in keys)
        or len(set(keys)) != len(keys)
    ):
        raise ValueError("model keys must be unique non-empty identifiers")
    requested = selected or keys
    if len(set(requested)) != len(requested) or set(requested) - set(keys):
        raise ValueError("selected model keys are duplicate or unknown")
    result = []
    for spec in specs:
        if spec["key"] not in requested:
            continue
        if "api_key" in spec or "token" in spec or "password" in spec:
            raise ValueError(f"{spec['key']}: put credentials in an environment variable")
        model_id, revision = spec.get("model_id"), spec.get("revision")
        if (
            not isinstance(model_id, str)
            or not model_id
            or not isinstance(revision, str)
            or not revision
        ):
            raise ValueError(f"{spec['key']}: model_id and revision are required")
        endpoint, endpoint_metadata = _endpoint(spec)
        api_key_env = spec.get("api_key_env")
        if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env):
            raise ValueError(f"{spec['key']}: invalid api_key_env")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if api_key_env and not api_key:
            raise ValueError(f"{spec['key']}: environment variable {api_key_env} is unset")
        decoding = spec.get("decoding", {})
        if not isinstance(decoding, dict):
            raise ValueError(f"{spec['key']}: decoding must be an object")
        max_tokens = decoding.get("max_tokens", 1024)
        temperature = decoding.get("temperature", 0.0)
        top_p = decoding.get("top_p", 1.0)
        if type(max_tokens) is not int or max_tokens < 1 or max_tokens > 8192:
            raise ValueError(f"{spec['key']}: max_tokens must be 1..8192")
        if type(temperature) not in {int, float} or not 0 <= temperature <= 2:
            raise ValueError(f"{spec['key']}: temperature must be 0..2")
        if type(top_p) not in {int, float} or not 0 < top_p <= 1:
            raise ValueError(f"{spec['key']}: top_p must be in (0,1]")
        extra_body = spec.get("extra_body", {})
        if not isinstance(extra_body, dict) or _contains_secret_key(extra_body):
            raise ValueError(f"{spec['key']}: extra_body must be a secret-free object")
        controlled = ("model", "messages", "temperature", "max_tokens", "top_p", "seed")
        if any(key in extra_body for key in controlled):
            raise ValueError(f"{spec['key']}: extra_body overrides controlled decoding fields")
        identity = {
            key: spec[key]
            for key in (
                "base_model_id",
                "base_revision",
                "adapter_id",
                "adapter_revision",
                "adapter_sha256",
            )
            if key in spec
        }
        if bool(identity.get("adapter_id")) != bool(identity.get("adapter_revision")):
            raise ValueError(f"{spec['key']}: adapter_id and adapter_revision must appear together")
        send_seed = spec.get("send_seed", False)
        if type(send_seed) is not bool:
            raise ValueError(f"{spec['key']}: send_seed must be boolean")
        request_timeout = spec.get("request_timeout_seconds", 120)
        if type(request_timeout) is not int or not 1 <= request_timeout <= 600:
            raise ValueError(f"{spec['key']}: request_timeout_seconds must be 1..600")
        serving_hardware = spec.get("serving_hardware", "unreported")
        serving_runtime = spec.get("serving_runtime", "unreported")
        if not all(isinstance(item, str) and item for item in (serving_hardware, serving_runtime)):
            raise ValueError(f"{spec['key']}: serving_hardware/runtime must be non-empty strings")
        serving_hourly_usd = spec.get("serving_hourly_usd")
        if serving_hourly_usd is not None and (
            type(serving_hourly_usd) not in {int, float} or serving_hourly_usd < 0
        ):
            raise ValueError(f"{spec['key']}: serving_hourly_usd must be non-negative")
        context_window = spec.get("serving_context_window_tokens")
        if context_window is not None and (
            type(context_window) is not int or context_window <= max_tokens
        ):
            raise ValueError(f"{spec['key']}: serving_context_window_tokens must exceed max_tokens")
        safe = {
            "key": spec["key"],
            "model_id": model_id,
            "revision": revision,
            "identity_source": "operator_declared_not_endpoint_attested",
            **identity,
            **endpoint_metadata,
            "api_key_env": api_key_env,
            "decoding": {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p},
            "send_seed": send_seed,
            "request_timeout_seconds": request_timeout,
            "serving_hardware": serving_hardware,
            "serving_runtime": serving_runtime,
            "serving_hourly_usd": serving_hourly_usd,
            "extra_body": extra_body,
        }
        if context_window is not None:
            safe["serving_context_window_tokens"] = context_window
        result.append({"safe": safe, "endpoint": endpoint, "api_key": api_key})
    return result


def preflight_sandbox() -> dict:
    """Refuse both missing Docker and the older image that bakes in gold data."""
    try:
        process = subprocess.run(
            ["docker", "image", "inspect", SANDBOX_IMAGE],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if process.returncode != 0:
            # Some Docker Desktop installations list/run a tag but fail to
            # inspect by tag. Resolve its exact image ID and inspect that.
            listing = subprocess.run(
                [
                    "docker",
                    "image",
                    "ls",
                    "--no-trunc",
                    "--format",
                    "{{.Repository}}:{{.Tag}} {{.ID}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if listing.returncode == 0:
                matches = [
                    line.split(" ", 1)[1].strip()
                    for line in listing.stdout.splitlines()
                    if line.startswith(f"{SANDBOX_IMAGE}:latest ")
                ]
                if len(matches) == 1:
                    process = subprocess.run(
                        ["docker", "image", "inspect", matches[0]],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Docker sandbox unavailable; no model request was sent") from exc
    try:
        images = json.loads(process.stdout) if process.returncode == 0 else []
        image = images[0] if isinstance(images, list) and images else {}
        label = image.get("Config", {}).get("Labels", {}).get("org.envoybench.sandbox")
    except (json.JSONDecodeError, AttributeError, TypeError, IndexError):
        image, label = {}, None
    if label != "v1" or f"{SANDBOX_IMAGE}:latest" not in image.get("RepoTags", []):
        raise RuntimeError(
            f"required {SANDBOX_IMAGE} image with {SANDBOX_LABEL} is unavailable; "
            "build it with docker build -f benchmarks/envoybench/Dockerfile -t rlm-sandbox ."
        )
    return {"image_id": image.get("Id"), "repo_digests": image.get("RepoDigests") or []}


def _redact(value: object, secrets: list[str]) -> object:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    return value


def _write_json(path: Path, value: object) -> None:
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
