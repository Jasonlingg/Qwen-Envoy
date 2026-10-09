"""Explain saved Qwen code-execution evidence with Nemotron on Nebius Token Factory.

Use --packet-only to inspect the exact model input without an API key or call.
The live path requires NEBIUS_API_KEY and writes a reviewable JSON/Markdown pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.artifacts import configuration_hash  # noqa: E402
from src.research.code_exec_packet import build_code_exec_packet  # noqa: E402
from src.research.explainer import (  # noqa: E402
    EXPLANATION_VERSION,
    EndpointExplainer,
    check_explanation,
    explanation_markdown,
    parse_explanation,
)

DEFAULT_ENDPOINT = "https://api.tokenfactory.nebius.com/v1"
DEFAULT_MODEL = "nvidia/Nemotron-3_5-Lightning"
_NEBIUS_HOST = re.compile(r"api\.tokenfactory(?:\.[a-z0-9-]+)?\.nebius\.com\Z")


def _read_row(transcript: Path, question_id: str | None) -> dict:
    payload = json.loads(transcript.read_text())
    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("transcript must be one EvalResult row or a list of rows")
    if question_id is not None:
        rows = [row for row in rows if isinstance(row, dict)
                and row.get("question_id") == question_id]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("select exactly one transcript row with --question-id")
    return rows[0]


def _validate_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "https" or not parsed.hostname
            or not _NEBIUS_HOST.fullmatch(parsed.hostname)
            or parsed.path.rstrip("/") != "/v1"
            or parsed.username or parsed.password or parsed.port
            or parsed.query or parsed.fragment):
        raise ValueError("endpoint must be a Nebius Token Factory HTTPS /v1 base URL")
    return endpoint.rstrip("/")


def explain_with_nebius(
    packet: dict,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    revision: str = "provider_catalog_unpinned",
    max_tokens: int = 2_400,
) -> dict:
    """Use the existing explainer, never making a request without a Nebius key."""
    api_key = os.environ.get("NEBIUS_API_KEY", "").strip()
    if not api_key:
        raise ValueError("NEBIUS_API_KEY is required for a live explanation")
    endpoint = _validate_endpoint(endpoint)
    if not model.startswith("nvidia/") or "nemotron" not in model.lower():
        raise ValueError("model must be an NVIDIA Nemotron Token Factory model ID")
    if max_tokens < 128:
        raise ValueError("max_tokens must be at least 128")

    explainer = EndpointExplainer(
        endpoint=endpoint, model=model, revision=revision, max_tokens=max_tokens,
    )
    previous_key = os.environ.get("EXPLAINER_API_KEY")
    os.environ["EXPLAINER_API_KEY"] = api_key
    try:
        raw = explainer.explain(packet)
    finally:
        if previous_key is None:
            os.environ.pop("EXPLAINER_API_KEY", None)
        else:
            os.environ["EXPLAINER_API_KEY"] = previous_key

    explanation = parse_explanation(raw)
    checks = check_explanation(explanation, packet)
    return {
        "schema_version": EXPLANATION_VERSION,
        "run_id": str(uuid4()),
        "status": "submitted",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_packet": packet,
        "evidence_packet_hash": configuration_hash(packet),
        "explainer": {**explainer.config, "endpoint": endpoint},
        "server_hardware": "Nebius Token Factory; provider managed",
        "explanation": explanation,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--question-id", help="Required when the transcript contains multiple rows")
    parser.add_argument("--corpus", type=Path, required=True, help="Frozen corpus JSON directory")
    parser.add_argument("--output", type=Path, required=True, help="New JSON artifact path")
    parser.add_argument("--packet-only", action="store_true", help="No key or API call")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help="Exact Nebius Token Factory /v1 base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Exact case-sensitive Nemotron model ID from the Nebius catalog")
    parser.add_argument("--revision", default="provider_catalog_unpinned")
    parser.add_argument("--max-tokens", type=int, default=2_400)
    args = parser.parse_args(argv)
    if args.output.suffix != ".json" or args.output.exists():
        raise ValueError("--output must be a new .json path")
    row = _read_row(args.transcript, args.question_id)
    packet = build_code_exec_packet(row, args.corpus)
    if args.packet_only:
        result = packet
    else:
        if args.output.with_suffix(".md").exists():
            raise FileExistsError(args.output.with_suffix(".md"))
        result = explain_with_nebius(
            packet, endpoint=args.endpoint, model=args.model,
            revision=args.revision, max_tokens=args.max_tokens,
        )
        result["source_transcript_sha256"] = hashlib.sha256(
            args.transcript.read_bytes()
        ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if not args.packet_only:
        args.output.with_suffix(".md").write_text(explanation_markdown(result))
    print(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
