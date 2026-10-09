"""Run a two-question, live Qwen product-path smoke on the synthetic vault.

This checks the model-server -> QwenInvestigator -> Docker tools -> verified
handoff path. It is not the ten-question readiness evaluation or a semantic
answer review. No reference answer, anchor, or grader note enters the prompt.

Example:
  python scripts/smoke_qwen_memory_worker.py \
    --endpoint http://127.0.0.1:8000/v1 --model qwen-v5 \
    --checkpoint jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5@27b912a.../checkpoint-50 \
    --snapshot out/research/personal-memory-synthetic-v2 \
    --output out/research/qwen-product-smoke-v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.personal_memory_web import make_qwen_investigator  # noqa: E402
from src.product.chat import _verified_review  # noqa: E402
from src.research.agent import load_snapshot  # noqa: E402

QUESTIONS = ROOT / "data/product_memory/questions_v2.json"
QUESTION_IDS = ("pm03_life_missing_outcome", "pm04_homework_correction")
BENCHMARK_ID = "personal-memory-synthetic-development-v2"
SMOKE_VERSION = "personal-memory-product-smoke-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases(snapshot: Path) -> tuple[dict, list[dict]]:
    manifest, _ = load_snapshot(snapshot)
    benchmark = json.loads(QUESTIONS.read_text())
    if benchmark.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("wrong synthetic development benchmark")
    if benchmark.get("corpus_hash") != manifest["corpus_hash"]:
        raise ValueError("question file and frozen snapshot have different corpus hashes")
    by_id = {item["id"]: item for item in benchmark["questions"]}
    if len(by_id) != len(benchmark["questions"]):
        raise ValueError("duplicate question ID")
    cases = [by_id[question_id] for question_id in QUESTION_IDS]
    if any(item.get("split") != "pilot_evaluation" for item in cases):
        raise ValueError("smoke question split changed")
    return manifest, cases


def _check_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("endpoint must be an HTTP(S) URL without embedded credentials or query")


def _check_packet(
    snapshot: Path, question: str, packet: dict, *, temporal: bool,
) -> tuple[dict, dict | None]:
    """Require real product tool use; exact spans still need semantic review."""
    checks = {
        "model_called": packet.get("model_requests_attempted", 0) > 0,
        "docker_only": packet.get("execution") == "docker_only",
        "document_tool_used": packet.get("document_tool_steps", 0) > 0,
        "source_inspected": packet.get("inspection_steps", 0) > 0,
        "acceptable_status": packet.get("status") == "evidence_found" if temporal else
                             packet.get("status") in {"evidence_found", "no_evidence"},
    }
    review = None
    if all(checks.values()):
        try:
            review = _verified_review(snapshot, packet, question)
        except (TypeError, ValueError, KeyError) as exc:
            checks["verified_handoff"] = False
            checks["handoff_error"] = f"{type(exc).__name__}: {exc}"
        else:
            checks["verified_handoff"] = True
    else:
        checks["verified_handoff"] = False
    checks["mechanical_pass"] = all(value for key, value in checks.items()
                                    if key not in {"handoff_error", "mechanical_pass"})
    return checks, review


def _persist(path: Path, artifact: dict, *, initial: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if initial:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(artifact, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def run_smoke(*, endpoint: str, model: str, checkpoint: str, snapshot: Path,
              output: Path, served_hardware: str | None = None) -> dict:
    """Use the same worker factory as the web app; preserve failed traces."""
    _check_endpoint(endpoint)
    if not model.strip() or not checkpoint.strip():
        raise ValueError("model and checkpoint identifiers are required")
    snapshot = snapshot.expanduser().resolve(strict=True)
    manifest, cases = _cases(snapshot)
    worker = make_qwen_investigator(endpoint, model, checkpoint)
    artifact = {
        "schema_version": SMOKE_VERSION,
        "screen": "synthetic_development_product_path_smoke_not_readiness_eval",
        "status": "running",
        "started_at": _utc_now(),
        "config": {
            "benchmark_id": BENCHMARK_ID,
            "question_split": "pilot_evaluation",
            "question_ids": list(QUESTION_IDS),
            "questions_sha256": _sha256(QUESTIONS),
            "corpus_hash": manifest["corpus_hash"],
            "snapshot": str(snapshot),
            "model_identity": worker.model_identity,
            "model_identity_attestation": "operator_supplied_not_verified_by_server",
            "product_prompt_sha256": hashlib.sha256(
                worker.policy_factory().system_prompt.encode()
            ).hexdigest(),
            "max_steps": worker.max_steps,
            "sandbox_image": worker.image,
            "reward": "none_product_smoke",
            "served_hardware": served_hardware or "operator_unspecified",
            "client_hardware": platform.platform(),
        },
        "episodes": [],
        "semantic_review": "not_performed",
        "readiness_decision": "not_assessed_by_two_question_smoke",
    }
    _persist(output, artifact, initial=True)
    for case in cases:
        started_at = _utc_now()
        started = time.monotonic()
        try:
            packet = worker.investigate(case["question"], snapshot)
        except Exception as exc:
            packet = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "model_requests_attempted": 0,
                "trajectory": [],
            }
        duration = time.monotonic() - started
        checks, handoff = _check_packet(
            snapshot, case["question"], packet,
            temporal=case["id"] == "pm04_homework_correction",
        )
        artifact["episodes"].append({
            "question_id": case["id"],
            "question": case["question"],
            "started_at": started_at,
            "duration_seconds": duration,
            "packet": packet,
            "verified_handoff": handoff,
            "checks": checks,
        })
        if not checks["mechanical_pass"]:
            artifact["status"] = "failed"
            artifact["stopped_after_question_id"] = case["id"]
            artifact["finished_at"] = _utc_now()
            _persist(output, artifact)
            return artifact
        _persist(output, artifact)
    artifact["status"] = "mechanical_smoke_pass"
    artifact["finished_at"] = _utc_now()
    _persist(output, artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True,
                        help="OpenAI-compatible /v1 endpoint serving the Qwen checkpoint")
    parser.add_argument("--model", required=True, help="served model or adapter ID")
    parser.add_argument("--checkpoint", required=True,
                        help="operator-supplied exact checkpoint and revision; not server-attested")
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "out/research/personal-memory-synthetic-v2")
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON artifact; existing files are never overwritten")
    parser.add_argument("--served-hardware", default=None,
                        help="operator-declared GPU/runtime, if known")
    args = parser.parse_args()
    try:
        result = run_smoke(**vars(args))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": result["status"],
        "question_ids_run": [item["question_id"] for item in result["episodes"]],
        "output": str(args.output),
    }))
    return 0 if result["status"] == "mechanical_smoke_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
