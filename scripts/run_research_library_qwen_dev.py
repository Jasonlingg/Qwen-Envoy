"""Development-only product-path Qwen smoke on four frozen public-paper questions.

Dry-run validates the corpus/questions and writes a pending human reference-review
form without contacting a model. Live mode requires that review record, an
explicit served checkpoint identity, and the Docker sandbox. Neither mode uses
held-out questions or treats exact quotations as semantic support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_learning_loop_pilot import validate as validate_sources  # noqa: E402
from src.env.repl import PersistentREPL  # noqa: E402
from src.env.tools import SEARCH_PROTOCOL_VERSION, TOOL_PREAMBLE  # noqa: E402
from src.policies.code_execution import DEFAULT_MAX_TOKENS  # noqa: E402
from src.policies.openai_compatible import OpenAICompatiblePolicy  # noqa: E402
from src.product.chat import _verified_review  # noqa: E402
from src.product.qwen_investigator import (  # noqa: E402
    PUBLIC_PAPER_SYSTEM_PROMPT,
    QwenInvestigator,
)
from src.research.agent import load_snapshot  # noqa: E402
from src.research.benchmark import load_benchmark  # noqa: E402

BENCHMARK = ROOT / "data/research/research_library_transfer_dev_v1.json"
SNAPSHOT = ROOT / "out/research/ai-agents-development-v1-20260912"
BENCHMARK_ID = "research-library-transfer-development-v1"
CORPUS_HASH = "9a9c1d750daf5647898eb18681a5f75775ba76047713316dcc5583f6ed201598"
QUESTION_IDS = ("D01", "D02", "D03", "D04")
REVIEW_SCHEMA = "research-library-dev-reference-review-v1"
RUN_SCHEMA = "research-library-qwen-product-dev-v1"
MAX_STEPS = 15
SEED = 42
DOCKER_IMAGE = "rlm-sandbox"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _source_check(benchmark_path: Path, snapshot: Path) -> tuple[dict, list[str], dict]:
    manifest, _ = load_snapshot(snapshot)
    benchmark = load_benchmark(benchmark_path, manifest)
    if benchmark["benchmark_id"] != BENCHMARK_ID:
        raise ValueError(f"expected development benchmark {BENCHMARK_ID}")
    if manifest["corpus_hash"] != CORPUS_HASH:
        raise ValueError("snapshot is not the pinned 20-paper development corpus")
    if len(manifest.get("papers", [])) != 20:
        raise ValueError("development snapshot must contain exactly 20 papers")
    if tuple(item["id"] for item in benchmark["questions"]) != QUESTION_IDS:
        raise ValueError("development questions must be D01 through D04 in order")
    summary, source_lines = validate_sources(benchmark_path, snapshot)
    if summary["question_count"] != len(QUESTION_IDS):
        raise ValueError("development source check has the wrong question count")
    return benchmark, source_lines, manifest


def _pending_review(benchmark_sha256: str, corpus_hash: str) -> dict:
    return {
        "schema_version": REVIEW_SCHEMA,
        "status": "pending",
        "benchmark_sha256": benchmark_sha256,
        "corpus_hash": corpus_hash,
        "reviewer": {"kind": None, "name": None, "reviewed_at": None},
        "questions": [
            {
                "id": question_id,
                "reference_answer": "pending",
                "answerability": "pending",
                "required_paper_ids": "pending",
                "anchors_support_reference": "pending",
                "notes": "",
            }
            for question_id in QUESTION_IDS
        ],
        "instructions": (
            "A person must inspect source-check.md and the frozen papers, then mark every "
            "field approved only after checking the reference answer, answerability, required "
            "papers, and whether each exact anchor actually supports its claim. Set status "
            "to approved, reviewer.kind to human, and provide name and reviewed_at. "
            "Agent-authored checks alone do not approve these references."
        ),
    }


def _approved_review(path: Path, benchmark_sha256: str, corpus_hash: str) -> dict:
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or record.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("wrong development reference-review schema")
    if record.get("benchmark_sha256") != benchmark_sha256:
        raise ValueError("reference review does not match the exact benchmark bytes")
    if record.get("corpus_hash") != corpus_hash:
        raise ValueError("reference review does not match the frozen corpus")
    if record.get("status") != "approved":
        raise ValueError("development reference review is not approved")
    reviewer = record.get("reviewer")
    if not isinstance(reviewer, dict) or reviewer.get("kind") != "human":
        raise ValueError("reference review needs an identified human reviewer")
    if not isinstance(reviewer.get("name"), str) or not reviewer["name"].strip():
        raise ValueError("reference review needs a reviewer name")
    reviewed_at = reviewer.get("reviewed_at")
    if not isinstance(reviewed_at, str):
        raise ValueError("reference review needs a timezone-aware reviewed_at")
    try:
        reviewed_time = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reference review has invalid reviewed_at") from exc
    if reviewed_time.tzinfo is None:
        raise ValueError("reference review needs a timezone-aware reviewed_at")
    reviews = record.get("questions")
    if (not isinstance(reviews, list) or len(reviews) != len(QUESTION_IDS)
            or any(not isinstance(item, dict) for item in reviews)
            or tuple(item["id"] for item in reviews) != QUESTION_IDS):
        raise ValueError("reference review must cover D01 through D04 exactly")
    fields = (
        "reference_answer", "answerability", "required_paper_ids",
        "anchors_support_reference",
    )
    for item in reviews:
        if any(item.get(field) != "approved" for field in fields):
            raise ValueError(f"reference review for {item['id']} is incomplete")
    return record


def _check_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("endpoint must be an HTTP(S) URL without credentials or query")


def _worker(endpoint: str, model: str, checkpoint: str) -> QwenInvestigator:
    return QwenInvestigator(
        policy_factory=lambda: OpenAICompatiblePolicy(
            endpoint=endpoint,
            model=model,
            api_key=os.environ.get("ENVOY_MODEL_API_KEY"),
            system_prompt=PUBLIC_PAPER_SYSTEM_PROMPT,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False},
                        "top_p": 1.0, "seed": SEED},
        ),
        model_identity={
            "checkpoint": checkpoint, "served_model": model,
            "endpoint": endpoint, "identity_attestation": "operator_supplied_not_server_verified",
            "temperature": 0.0, "top_p": 1.0, "seed": SEED,
            "max_tokens": DEFAULT_MAX_TOKENS, "thinking": False,
        },
        max_steps=MAX_STEPS,
        image=DOCKER_IMAGE,
        source_domain="public_papers",
    )


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run(
    *,
    output: Path,
    dry_run: bool,
    benchmark_path: Path = BENCHMARK,
    snapshot: Path = SNAPSHOT,
    endpoint: str | None = None,
    model: str | None = None,
    checkpoint: str | None = None,
    hardware: str | None = None,
    run_label: str | None = None,
    review_record: Path | None = None,
) -> dict:
    """Validate first; only live mode can construct a model-backed worker."""
    benchmark_path = benchmark_path.expanduser().resolve(strict=True)
    snapshot = snapshot.expanduser().resolve(strict=True)
    benchmark, source_lines, snapshot_manifest = _source_check(benchmark_path, snapshot)
    benchmark_sha256 = _sha256(benchmark_path)
    review_sha256 = None
    if not dry_run:
        if not all(isinstance(value, str) and value.strip() for value in
                   (endpoint, model, checkpoint, hardware, run_label)):
            raise ValueError(
                "live mode needs --endpoint, --model, --checkpoint, --hardware, and --run-label"
            )
        _check_endpoint(endpoint)
        if review_record is None:
            raise ValueError("live mode needs --review-record with human-approved references")
        review_record = review_record.expanduser().resolve(strict=True)
        _approved_review(review_record, benchmark_sha256, snapshot_manifest["corpus_hash"])
        review_sha256 = _sha256(review_record)
        if not PersistentREPL._docker_available(DOCKER_IMAGE):
            raise ValueError(f"Docker sandbox image {DOCKER_IMAGE!r} is unavailable")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": RUN_SCHEMA,
        "mode": "dry_run" if dry_run else "live_development",
        "status": "dry_run_ready" if dry_run else "running",
        "started_at": _utc_now(),
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": benchmark_sha256,
        "question_ids": list(QUESTION_IDS),
        "snapshot": str(snapshot),
        "snapshot_manifest_sha256": _sha256(snapshot / "manifest.json"),
        "corpus_hash": snapshot_manifest["corpus_hash"],
        "parser_version": snapshot_manifest.get("parser_version"),
        "paper_doc_ids": [item["doc_id"] for item in snapshot_manifest["papers"]],
        "source_domain": "public_papers",
        "system_prompt_sha256": hashlib.sha256(PUBLIC_PAPER_SYSTEM_PROMPT.encode()).hexdigest(),
        "code_sha256": {
            relative: _sha256(ROOT / relative)
            for relative in (
                "scripts/run_research_library_qwen_dev.py",
                "src/product/qwen_investigator.py",
                "src/policies/openai_compatible.py",
                "src/env/tools.py",
                "src/env/repl.py",
            )
        },
        "model_identity": {
            "endpoint": endpoint, "served_model": model, "checkpoint": checkpoint,
            "identity_attestation": "operator_supplied_not_server_verified",
        },
        "run_label": run_label,
        "decoding": {"temperature": 0.0, "top_p": 1.0, "seed": SEED,
                     "max_tokens_per_action": DEFAULT_MAX_TOKENS,
                     "enable_thinking": False},
        "max_steps": MAX_STEPS,
        "docker_image": DOCKER_IMAGE,
        "hardware": hardware,
        "reward": "none_product_path; semantic support needs blind review",
        "reference_review_sha256": review_sha256,
        "source_check": "mechanically_valid; human_reference_review_required",
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model_requests_attempted": 0,
        "completed_question_ids": [],
        "semantic_output_review": "pending",
    }
    _write_json(output / "manifest.json", manifest)
    if dry_run:
        (output / "source-check.md").write_text("\n".join(source_lines) + "\n")
        _write_json(
            output / "reference-review-template.json",
            _pending_review(benchmark_sha256, snapshot_manifest["corpus_hash"]),
        )
        manifest["finished_at"] = _utc_now()
        _write_json(output / "manifest.json", manifest)
        return manifest

    try:
        worker = _worker(endpoint, model, checkpoint)
    except Exception as exc:
        manifest.update(
            status="worker_initialization_failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=_utc_now(),
        )
        _write_json(output / "manifest.json", manifest)
        return manifest
    for case in benchmark["questions"]:
        question_id = case["id"]
        started = time.monotonic()
        packet = None
        try:
            # Only the question reaches Qwen. Gold answers, grader notes, and
            # source anchors remain in the local benchmark file.
            packet = worker.investigate(case["question"], snapshot)
            identity = packet.get("model_identity")
            checks = {
                "model_called": packet.get("model_requests_attempted", 0) > 0,
                "docker_only": packet.get("execution") == "docker_only",
                "document_tool_used": packet.get("document_tool_steps", 0) > 0,
                "source_inspected": packet.get("inspection_steps", 0) > 0,
                "acceptable_status": packet.get("status") in {"evidence_found", "no_evidence"},
                "public_paper_prompt": (
                    packet.get("source_domain") == "public_papers"
                    and packet.get("system_prompt_sha256")
                    == hashlib.sha256(PUBLIC_PAPER_SYSTEM_PROMPT.encode()).hexdigest()
                ),
                "pinned_document_tools": (
                    packet.get("tool_search_version") == SEARCH_PROTOCOL_VERSION
                    and packet.get("tool_preamble_sha256")
                    == hashlib.sha256(TOOL_PREAMBLE.encode()).hexdigest()
                ),
                "served_identity": (
                    isinstance(identity, dict)
                    and identity.get("checkpoint") == checkpoint
                    and identity.get("served_model") == model
                ),
            }
            handoff = None
            if all(checks.values()):
                handoff = _verified_review(snapshot, packet, case["question"])
            checks["verified_handoff"] = handoff is not None
        except Exception as exc:
            if not isinstance(packet, dict):
                packet = {"status": "error", "model_requests_attempted": 0,
                          "trajectory": []}
            packet["runner_error"] = f"{type(exc).__name__}: {exc}"
            checks, handoff = {"verified_handoff": False}, None
        episode = {
            "question_id": question_id,
            "question": case["question"],
            "duration_seconds": time.monotonic() - started,
            "packet": packet,
            "mechanical_checks": checks,
            "verified_handoff": handoff,
            "semantic_review": "pending",
        }
        _write_json(output / "packets" / f"{question_id}.json", episode)
        manifest["completed_question_ids"].append(question_id)
        manifest["model_requests_attempted"] += packet.get("model_requests_attempted", 0)
        manifest["last_question_id"] = question_id
        _write_json(output / "manifest.json", manifest)
        if not all(checks.values()):
            manifest["status"] = "stopped_on_mechanical_failure"
            break
    else:
        manifest["status"] = "packets_saved_semantic_review_pending"
    manifest["finished_at"] = _utc_now()
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory; existing output is never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="validate and draft review only")
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--endpoint", help="OpenAI-compatible /v1 endpoint")
    parser.add_argument("--model", help="served Qwen model ID")
    parser.add_argument("--checkpoint", help="exact operator-supplied checkpoint/revision")
    parser.add_argument("--hardware", help="GPU/runtime name for live run")
    parser.add_argument("--run-label", help="model-arm label, e.g. base or v5")
    parser.add_argument("--review-record", type=Path,
                        help="human-approved dev reference-review JSON; required for live mode")
    args = parser.parse_args()
    try:
        result = run(
            output=args.output, dry_run=args.dry_run,
            benchmark_path=args.benchmark, snapshot=args.snapshot,
            endpoint=args.endpoint, model=args.model, checkpoint=args.checkpoint,
            hardware=args.hardware, run_label=args.run_label,
            review_record=args.review_record,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": result["status"], "output": str(args.output),
        "question_ids": result["completed_question_ids"] if not args.dry_run
                        else result["question_ids"],
        "model_requests_attempted": result["model_requests_attempted"],
    }))
    return 0 if result["status"] in {
        "dry_run_ready", "packets_saved_semantic_review_pending",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
