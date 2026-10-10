"""Run the same frozen EnvoyBench questions through declared model endpoints.

This is a local, trusted-operator CLI, not a public code-execution service. The
model's Python actions run only in the labeled Docker sandbox. The legacy
environment reward is recorded as a diagnostic, never as a research-support
benchmark score; independent claim review is required for that judgment.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from benchmarks.envoybench.budget import InferenceBudget
from benchmarks.envoybench.frozen_dataset import (
    DEFAULT_DATASET,
    ROOT,
    _dataset_path,  # noqa: F401 - compatibility for existing runner imports
    _file_sha256,
    _read_json,
    load_split,
)
from benchmarks.envoybench.runtime_support import (
    _MODEL_KEY,  # noqa: F401 - compatibility for existing runner imports
    _SECRET_WORDS,  # noqa: F401 - compatibility for existing runner imports
    MODEL_SCHEMA,  # noqa: F401 - compatibility for existing runner imports
    SANDBOX_IMAGE,  # noqa: F401 - compatibility for existing runner imports
    SANDBOX_LABEL,
    _contains_secret_key,  # noqa: F401 - compatibility for existing runner imports
    _endpoint,  # noqa: F401 - compatibility for existing runner imports
    _redact,
    _sha256_text,
    _write_json,
    load_models,
    preflight_sandbox,
)
from src.env.corpus import Corpus
from src.env.reward import REWARD_VERSION
from src.env.tools import SEARCH_PROTOCOL_VERSION, TOOL_PREAMBLE
from src.eval.artifacts import configuration_hash
from src.eval.harness import EvalResult, run_eval
from src.policies.code_execution import QASPER_SYSTEM_PROMPT
from src.policies.openai_compatible import OpenAICompatiblePolicy

RUN_SCHEMA = "envoybench-run-v1"


def _hardware() -> dict:
    hardware = {"platform": platform.platform(), "python": platform.python_version(), "gpu": None}
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        if process.returncode == 0:
            hardware["gpu"] = process.stdout.splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return hardware


def _git_commit() -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _implementation_hashes() -> dict[str, str]:
    paths = [
        "src/env/document_env.py", "src/env/repl.py", "src/env/repl_worker.py",
        "src/env/tools.py", "src/env/reward.py", "src/eval/harness.py",
        "src/policies/code_execution.py", "src/policies/openai_compatible.py",
        "benchmarks/envoybench/run.py", "benchmarks/envoybench/budget.py",
        "benchmarks/envoybench/frozen_dataset.py",
        "benchmarks/envoybench/runtime_support.py",
    ]
    return {name: _file_sha256(ROOT / name) for name in paths}


def _model_factory(model: dict, seed: int, transport=None):
    spec = model["safe"]
    extra_body = {**spec["extra_body"], "top_p": spec["decoding"]["top_p"]}
    if spec.get("send_seed"):
        extra_body["seed"] = seed

    def factory() -> OpenAICompatiblePolicy:
        return OpenAICompatiblePolicy(
            endpoint=model["endpoint"], model=spec["model_id"], api_key=model["api_key"],
            max_tokens=spec["decoding"]["max_tokens"],
            temperature=spec["decoding"]["temperature"],
            timeout=spec["request_timeout_seconds"],
            extra_body=extra_body, system_prompt=QASPER_SYSTEM_PROMPT,
            **({"transport": transport} if transport is not None else {}),
        )

    return factory


def _result_row(result: EvalResult, model_key: str, manifest: dict, secrets: list[str]) -> dict:
    trace = [
        {**step.model_dump(exclude_none=True), "output": step.observation}
        for step in result.trajectory
    ]
    submitted = bool(trace and re.search(r"(?im)^\s*SUBMIT:\s*", trace[-1]["action"]))
    status = ("error" if result.status == "error" else
              "escalated" if result.status == "escalated" else
              "submitted" if submitted else "no_submission")
    row = {
        "schema_version": RUN_SCHEMA,
        "run_id": manifest["run_id"], "comparison_id": manifest["comparison_id"],
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_hash": manifest["benchmark_hash"],
        "corpus_hash": manifest["corpus_hash"],
        "split": manifest["split"], "split_status": manifest["split_status"],
        "question_id": result.question_id, "question": result.question,
        "model_key": model_key, "status": status,
        "environment_status": result.status, "error": result.error,
        "predicted_answer": result.predicted_answer,
        "predicted_citations": result.predicted_citations,
        "predicted_evidence": result.predicted_evidence,
        "trajectory": trace, "steps": result.steps,
        "duration_seconds": result.duration_seconds,
        "policy_metadata": result.policy_metadata,
        "verifier_events": [event.model_dump(mode="json") for event in result.verifier_events],
        "escalation": result.escalation.model_dump(mode="json") if result.escalation else None,
        "environment_diagnostics_not_benchmark_score": {
            "legacy_reward": result.reward, "legacy_answer_score": result.answer_score,
            "legacy_citation_precision": result.citation_precision,
            "legacy_citation_recall": result.citation_recall,
            "reward_version": result.reward_version,
        },
    }
    return _redact(row, secrets)


def run(
    dataset: Path, split: str, models_path: Path, output: Path, *,
    model_keys: list[str] | None = None, question_ids: list[str] | None = None,
    seed: int = 42, max_steps: int = 15, validate_only: bool = False,
    budget: InferenceBudget | None = None,
    verifier_protocol: str = "fail-closed-v2",
) -> dict:
    """Run a frozen split once per endpoint model, saving a complete trace matrix.

    ``validate_only`` inspects data/model configuration without Docker or model
    calls. A real run preflights the labeled image before creating output files.
    """
    if max_steps < 1 or max_steps > 30:
        raise ValueError("max_steps must be 1..30")
    if verifier_protocol not in {"legacy-v1", "fail-closed-v2"}:
        raise ValueError("verifier_protocol must be legacy-v1 or fail-closed-v2")
    fail_closed = verifier_protocol == "fail-closed-v2"
    benchmark, snapshot, benchmark_path, corpus_path, entry = load_split(dataset, split)
    questions = benchmark["questions"]
    available_ids = [question["id"] for question in questions]
    selected_ids = question_ids or available_ids
    if len(set(selected_ids)) != len(selected_ids) or set(selected_ids) - set(available_ids):
        raise ValueError("selected question IDs are duplicate or unknown")
    questions = [question for question in questions if question["id"] in selected_ids]
    models = load_models(models_path, model_keys)
    if validate_only:
        return {
            "benchmark_id": benchmark["benchmark_id"], "split": split,
            "split_status": snapshot.get("split_status", entry.get("split_status", "unreviewed")),
            "corpus_hash": benchmark["corpus_hash"],
            "question_ids": [question["id"] for question in questions],
            "models": [model["safe"] for model in models],
            "verifier_protocol": verifier_protocol,
            "validation_only": True,
        }
    sandbox_image = preflight_sandbox()
    if output.exists():
        raise ValueError(f"output already exists: {output}")

    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    started = time.monotonic()
    corpus = Corpus(corpus_path=str(corpus_path))
    corpus.load(build_index=False)
    protocol = {
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": configuration_hash(benchmark),
        "corpus_hash": benchmark["corpus_hash"],
        "question_ids": [question["id"] for question in questions],
        "split": split, "max_steps": max_steps, "seed": seed,
        "seed_note": (
            "Local runner seed only; remote determinism requires endpoint support "
            "and is not guaranteed."
        ),
        "system_prompt_sha256": _sha256_text(QASPER_SYSTEM_PROMPT),
        "system_prompt_name": "QASPER_SYSTEM_PROMPT",
        "tool_preamble_sha256": _sha256_text(TOOL_PREAMBLE),
        "tool_search_version": SEARCH_PROTOCOL_VERSION,
        "require_evidence": True, "evidence_verifier": True,
        "verifier_feedback_budget": 1,
        "escalate_after_verifier_failure": fail_closed,
        "docker_sandbox": SANDBOX_LABEL,
    }
    if fail_closed:
        # Keep legacy-v1's protocol hash reproducible from the archived manifest.
        protocol["verifier_protocol"] = verifier_protocol
    manifest = {
        "schema_version": RUN_SCHEMA,
        "run_id": uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        **protocol, "comparison_id": configuration_hash(protocol),
        "split_question_count": len(available_ids),
        "full_split": len(questions) == len(available_ids),
        "subset_smoke": len(questions) != len(available_ids),
        "split_status": snapshot.get("split_status", entry.get("split_status", "unreviewed")),
        "benchmark_file_sha256": _file_sha256(benchmark_path),
        "dataset_index_sha256": _file_sha256(dataset / "manifest.json"),
        "model_config_sha256": _file_sha256(models_path),
        "models": [model["safe"] for model in models],
        "model_identity_note": (
            "Model/revision IDs are operator declarations; the endpoint does not attest weights."
        ),
        "legacy_reward_version": REWARD_VERSION,
        "legacy_reward_is_benchmark_score": False,
        "sandbox_image": sandbox_image,
        "implementation_sha256": _implementation_hashes(),
        "git_commit": _git_commit(), "runner_hardware": _hardware(),
    }
    if budget is not None:
        manifest["inference_budget"] = budget.config
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "manifest.json", manifest)
    rows: list[dict] = []
    secrets = [model["api_key"] for model in models if model["api_key"]]
    try:
        for model in models:
            key = model["safe"]["key"]
            consecutive_errors = 0
            for question in questions:
                result_set = run_eval(
                    corpus=corpus, questions=[question],
                    policies={key: _model_factory(model, seed, transport=budget)},
                    max_steps=max_steps, use_docker=True,
                    corpus_path=str(corpus_path), workers=1,
                    require_evidence=True, include_preamble=True,
                    evidence_verifier=True, verifier_feedback_budget=1,
                    escalate_after_verifier_failure=fail_closed,
                )
                if [item.question_id for item in result_set] != [question["id"]]:
                    raise RuntimeError(
                        f"{key}: incomplete question matrix; refusing complete artifact"
                    )
                rows.extend(_result_row(item, key, manifest, secrets) for item in result_set)
                _write_json(output / "results.partial.json", rows)
                if budget is not None:
                    _write_json(output / "usage-budget.json", budget.snapshot())
                    if budget.halted_reason:
                        raise RuntimeError(budget.halted_reason)
                    consecutive_errors = (consecutive_errors + 1
                                          if result_set[0].status == "error" else 0)
                    if consecutive_errors >= 3:
                        raise RuntimeError("three consecutive error episodes; bounded run stopped")
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - started
        _write_json(output / "results.json", rows)
        _write_json(output / "manifest.json", manifest)
        (output / "results.partial.json").unlink(missing_ok=True)
    except BaseException:
        manifest["status"] = "incomplete"
        _write_json(output / "manifest.json", manifest)
        raise
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("dev", "test_candidate"), default="dev")
    parser.add_argument("--models", type=Path, required=True, help="envoybench-models-v1 JSON file")
    parser.add_argument(
        "--model", action="append", dest="models_selected", help="model key; repeatable"
    )
    parser.add_argument(
        "--question-id", action="append", dest="question_ids", help="ID; repeatable"
    )
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument(
        "--verifier-protocol", choices=("fail-closed-v2", "legacy-v1"),
        default="fail-closed-v2",
        help="fail-closed for new runs; legacy-v1 only to reproduce archived runs",
    )
    parser.add_argument("--validate-only", action="store_true", help="no Docker or model calls")
    parser.add_argument("--budget", type=Path, help="local estimated-cost/request limit JSON")
    args = parser.parse_args(argv)
    try:
        result = run(
            args.dataset, args.split, args.models, args.output,
            model_keys=args.models_selected, question_ids=args.question_ids,
            seed=args.seed, max_steps=args.max_steps, validate_only=args.validate_only,
            budget=InferenceBudget(_read_json(args.budget)) if args.budget else None,
            verifier_protocol=args.verifier_protocol,
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"EnvoyBench: {exc}\n")
    if args.validate_only:
        print(json.dumps(result, indent=2))
    else:
        print(f"EnvoyBench complete: {args.output} ({len(result['question_ids'])} questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
