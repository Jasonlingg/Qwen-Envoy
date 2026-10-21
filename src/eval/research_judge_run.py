"""One-shot paid dispatch with durable reservation, bounded I/O and no retry."""
from __future__ import annotations

import hashlib
import json
import os
import time
from decimal import Decimal
from pathlib import Path

from src.eval.research_judge_checks import check_bundle, sha256
from src.eval.research_judge_plan import CATALOG, LIMITS, MODEL, URL, encoded, input_reservation


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def bounded_request(url, *, payload=None, api_key=None):
    import httpx

    allowed = {URL, CATALOG}
    if url not in allowed or ((url == URL) != (payload is not None and bool(api_key))):
        raise ValueError("only the pinned catalog GET or inference POST is allowed")
    headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.monotonic()
    with httpx.Client(transport=httpx.HTTPTransport(retries=0), trust_env=False,
                      follow_redirects=False, timeout=httpx.Timeout(240, connect=10)) as client:
        with client.stream("POST" if payload is not None else "GET", url,
                           content=None if payload is None else encoded(payload),
                           headers=headers) as response:
            response.raise_for_status()  # redirects are never followed, including with credentials
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if (len(body) > LIMITS["response_bytes"]
                        or time.monotonic() - started > LIMITS["soft_elapsed_guard_seconds"]):
                    raise ValueError("response size or soft elapsed guard exceeded")
    return json.loads(body)


def validate_preflight(directory, root, repo, catalog):
    directory, root, repo = map(Path, (directory, root, repo))
    manifest = json.loads((directory / "manifest.json").read_text())
    payload = json.loads((directory / "request.json").read_text())
    check_bundle(root)
    if (manifest["endpoint"] != URL or manifest["limits"] != LIMITS
            or manifest["request_sha256"] != hashlib.sha256(encoded(payload)).hexdigest()
            or manifest["fixture_lock_sha256"] != sha256(root / "freeze-lock.json")
            or manifest["prior_plan_sha256"] != sha256(repo /
                "examples/reader-learning-diagnosis/judge-validation-plan.json")
            or manifest["pre_dispatch_amendment_sha256"] != sha256(repo /
                "examples/research-judge-super-check/pre-dispatch-amendment.json")
            or manifest["original_immutable_judge_readiness_condition_met"] is not False
            or manifest["status"] != "prepared_not_run"):
        raise ValueError("frozen plan/request/fixture binding differs")
    for name, digest in manifest["implementation_sha256"].items():
        if sha256(repo / name) != digest:
            raise ValueError("implementation changed after freeze")
    model = next(m for m in catalog if any(f["model_id"] == MODEL for f in m["flavors"]))
    flavor = next(f for f in model["flavors"] if f["model_id"] == MODEL)
    if (model["status"] != "active"
            or flavor["input_price_per_million_tokens"] != 0.3
            or flavor["output_price_per_million_tokens"] != 0.9
            or flavor["max_model_len"] != LIMITS["input_tokens"]):
        raise ValueError("live price/context changed; skip paid call")
    if (payload["model"] != MODEL or payload["n"] != 1 or payload["stream"] is not False
            or payload["max_completion_tokens"] != 16_384
            or "max_tokens" in payload or payload["temperature"] != 0.0
            or input_reservation(payload) != manifest["input_tokens_reserved"]
            or input_reservation(payload) > LIMITS["input_tokens"]):
        raise ValueError("request/token budget differs")
    cost = (Decimal(LIMITS["input_tokens"]) * Decimal("0.3")
            + Decimal(16_384) * Decimal("0.9")) / 1_000_000
    if cost > Decimal(LIMITS["usd"]) or str(cost) != manifest["maximum_catalog_cost_usd"]:
        raise ValueError("cannot establish <=$0.50 local conservative bound; skip paid call")
    return manifest, payload


def run(directory, root, repo, api_key, *, transport=bounded_request, catalog=None,
        allowance_path=None):
    """A failed/ambiguous request consumes the only attempt, including after crashes."""
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("NEBIUS_API_KEY is required")
    if catalog is None:
        catalog = transport(CATALOG)
    manifest, payload = validate_preflight(directory, root, repo, catalog)
    directory = Path(directory)
    if (directory / "dispatch.json").exists():
        raise FileExistsError("this run already consumed its attempt")
    allowance = (Path(allowance_path) if allowance_path is not None else Path(repo) /
                 "examples/research-judge-super-check/paid-allowance-consumed.json")
    # CLI output-directory changes cannot create a new paid allowance.
    write_new(allowance, {"authorization": "research-judge-super-alias-diagnostic-2026-10-21",
        "requests_reserved": 1, "maximum_catalog_cost_usd": "0.0933888",
        "manifest_sha256": sha256(directory / "manifest.json"),
        "request_sha256": manifest["request_sha256"],
        "fixture_lock_sha256": manifest["fixture_lock_sha256"],
        "consumed_even_if_transport_or_storage_fails": True})
    write_new(directory / "dispatch.json", {"requests_reserved": 1,
        "input_tokens_reserved": manifest["input_tokens_reserved"],
        "completion_tokens_reserved": 16_384, "request_sha256": manifest["request_sha256"],
        "reservation_consumed_even_if_transport_fails": True})
    started = time.monotonic()
    try:
        response = transport(URL, payload=payload, api_key=api_key)
        write_new(directory / "response.json", response)
        data = response if isinstance(response, dict) else {}
        usage = data.get("usage")
        prompt = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
        complete = type(prompt) is int and type(completion) is int and min(prompt, completion) >= 0
        within = complete and prompt <= manifest["input_tokens_reserved"] and completion <= 16_384
        cost = str((Decimal(prompt) * Decimal("0.3")
                    + Decimal(completion) * Decimal("0.9")) / 1_000_000) if complete else None
        receipt = {"status": "response_saved" if within else "usage_unknown_or_bound_violated",
            "elapsed_seconds": time.monotonic() - started, "usage": usage,
            "usage_complete_and_within_reservation": within,
            "catalog_cost_usd": cost, "invoice_cost_usd": None,
            "served_model": data.get("model"),
            "served_model_matches_request": data.get("model") == MODEL,
            "response_sha256": hashlib.sha256(encoded(response)).hexdigest(),
            "request_sha256": manifest["request_sha256"],
            "fixture_lock_sha256": manifest["fixture_lock_sha256"],
            "manifest_sha256": sha256(directory / "manifest.json"),
            "system_fingerprint": data.get("system_fingerprint"), "requests_dispatched": 1}
    except Exception as exc:
        receipt = {"status": "request_or_storage_failed_no_retry", "exception_class":
            type(exc).__name__, "elapsed_seconds": time.monotonic() - started,
            "usage": None, "catalog_cost_usd": None, "requests_dispatched": 1}
    write_new(directory / "receipt.json", receipt)
    return receipt
