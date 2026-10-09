"""Run the same frozen EnvoyBench questions through declared model endpoints.

This is a local, trusted-operator CLI, not a public code-execution service. The
model's Python actions run only in the labeled Docker sandbox. The legacy
environment reward is recorded as a diagnostic, never as a research-support
benchmark score; independent claim review is required for that judgment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from src.env.corpus import Corpus
from src.env.reward import REWARD_VERSION
from src.env.tools import SEARCH_PROTOCOL_VERSION, TOOL_PREAMBLE
from src.eval.artifacts import configuration_hash, content_hash
from src.eval.harness import EvalResult, run_eval
from src.policies.code_execution import QASPER_SYSTEM_PROMPT
from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.research.benchmark import load_benchmark

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "benchmarks/envoybench/data"
SANDBOX_IMAGE = "rlm-sandbox"
SANDBOX_LABEL = "org.envoybench.sandbox=v1"
RUN_SCHEMA = "envoybench-run-v1"
MODEL_SCHEMA = "envoybench-models-v1"
_MODEL_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SECRET_WORDS = {"api_key", "apikey", "authorization", "bearer", "password", "secret", "token"}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_path(dataset: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("dataset split entry is missing a path")
    base = dataset.resolve(strict=True)
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"dataset path escapes the frozen dataset: {relative}")
    if not path.exists():
        raise ValueError(
            f"frozen dataset file missing: {path}; run "
            "python -m benchmarks.envoybench.build_data --output benchmarks/envoybench/data"
        )
    return path


def load_split(dataset: Path, split: str) -> tuple[dict, dict, Path, Path, dict]:
    """Load and hash-check a materialized split before any model request."""
    index = _read_json(dataset / "manifest.json")
    if index.get("schema_version") != "envoybench-data-manifest-v1":
        raise ValueError("unsupported frozen dataset manifest")
    splits_path = _dataset_path(dataset, "splits.json")
    if _file_sha256(splits_path) != index.get("frozen_splits_sha256"):
        raise ValueError("frozen split specification hash differs from dataset manifest")
    split_spec = (_read_json(splits_path).get("splits") or {}).get(split)
    if not isinstance(split_spec, dict):
        raise ValueError(f"{split}: absent from frozen split specification")
    entry = (index.get("splits") or {}).get(split)
    if not isinstance(entry, dict):
        available = sorted(index.get("splits") or {})
        raise ValueError(f"unknown split {split!r}; available: {available}")
    benchmark_path = _dataset_path(dataset, entry.get("benchmark"))
    corpus_path = _dataset_path(dataset, entry.get("corpus"))
    snapshot_path = _dataset_path(dataset, entry.get("manifest"))
    if not corpus_path.is_dir():
        raise ValueError(f"corpus is not a directory: {corpus_path}")
    snapshot = _read_json(snapshot_path)
    benchmark = load_benchmark(benchmark_path, snapshot)
    benchmark_file_hash = _file_sha256(benchmark_path)
    if (benchmark_file_hash != entry.get("benchmark_sha256")
            or benchmark_file_hash != snapshot.get("benchmark_sha256")):
        raise ValueError("generated benchmark bytes differ from frozen manifests")
    selected_ids = [question["id"] for question in benchmark["questions"]]
    if (selected_ids != split_spec.get("question_ids")
            or selected_ids != snapshot.get("selected_question_ids")
            or len(selected_ids) != entry.get("question_count")):
        raise ValueError("generated benchmark question IDs differ from frozen selection")
    target_papers = sorted({
        question.get("source_paper_id") for question in benchmark["questions"]
    })
    if target_papers != snapshot.get("selected_target_paper_ids"):
        raise ValueError("generated benchmark target papers differ from frozen selection")
    if snapshot.get("split_status") != entry.get("status"):
        raise ValueError("snapshot split status differs from dataset manifest")
    actual_hash = content_hash(corpus_path)
    if (actual_hash != benchmark["corpus_hash"]
            or actual_hash != snapshot.get("corpus_hash")
            or actual_hash != entry.get("corpus_hash")):
        raise ValueError("frozen corpus contents differ from benchmark corpus_hash")
    return benchmark, snapshot, benchmark_path, corpus_path, entry


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
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or
            parsed.username or parsed.password or parsed.query or parsed.fragment):
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
    if len(keys) != len(specs) or any(not isinstance(key, str) or not _MODEL_KEY.fullmatch(key)
                                     for key in keys) or len(set(keys)) != len(keys):
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
        if (not isinstance(model_id, str) or not model_id or
                not isinstance(revision, str) or not revision):
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
        identity = {key: spec[key] for key in (
            "base_model_id", "base_revision", "adapter_id", "adapter_revision", "adapter_sha256"
        ) if key in spec}
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
        if (serving_hourly_usd is not None and
                (type(serving_hourly_usd) not in {int, float} or serving_hourly_usd < 0)):
            raise ValueError(f"{spec['key']}: serving_hourly_usd must be non-negative")
        context_window = spec.get("serving_context_window_tokens")
        if context_window is not None and (
            type(context_window) is not int or context_window <= max_tokens
        ):
            raise ValueError(
                f"{spec['key']}: serving_context_window_tokens must exceed max_tokens"
            )
        safe = {
            "key": spec["key"], "model_id": model_id, "revision": revision,
            "identity_source": "operator_declared_not_endpoint_attested",
            **identity, **endpoint_metadata, "api_key_env": api_key_env,
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
            capture_output=True, text=True, timeout=10,
        )
        if process.returncode != 0:
            # Some Docker Desktop installations list/run a tag but fail to
            # inspect by tag. Resolve its exact image ID and inspect that.
            listing = subprocess.run(
                ["docker", "image", "ls", "--no-trunc", "--format",
                 "{{.Repository}}:{{.Tag}} {{.ID}}"],
                capture_output=True, text=True, timeout=10,
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
                        capture_output=True, text=True, timeout=10,
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
        "benchmarks/envoybench/run.py",
    ]
    return {name: _file_sha256(ROOT / name) for name in paths}


def _model_factory(model: dict, seed: int):
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
        )

    return factory


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


def _write_json(path: Path, value: object) -> None:
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def run(
    dataset: Path, split: str, models_path: Path, output: Path, *,
    model_keys: list[str] | None = None, question_ids: list[str] | None = None,
    seed: int = 42, max_steps: int = 15, validate_only: bool = False,
) -> dict:
    """Run a frozen split once per endpoint model, saving a complete trace matrix.

    ``validate_only`` inspects data/model configuration without Docker or model
    calls. A real run preflights the labeled image before creating output files.
    """
    if max_steps < 1 or max_steps > 30:
        raise ValueError("max_steps must be 1..30")
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
        "verifier_feedback_budget": 1, "escalate_after_verifier_failure": False,
        "docker_sandbox": SANDBOX_LABEL,
    }
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
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "manifest.json", manifest)
    rows: list[dict] = []
    secrets = [model["api_key"] for model in models if model["api_key"]]
    try:
        for model in models:
            key = model["safe"]["key"]
            result_set = run_eval(
                corpus=corpus, questions=questions,
                policies={key: _model_factory(model, seed)},
                max_steps=max_steps, use_docker=True,
                corpus_path=str(corpus_path), workers=1,
                require_evidence=True, include_preamble=True,
                evidence_verifier=True, verifier_feedback_budget=1,
                escalate_after_verifier_failure=False,
            )
            expected = [question["id"] for question in questions]
            if sorted(item.question_id for item in result_set) != sorted(expected):
                raise RuntimeError(f"{key}: incomplete question matrix; refusing complete artifact")
            rows.extend(_result_row(item, key, manifest, secrets) for item in result_set)
            _write_json(output / "results.partial.json", rows)
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
    parser.add_argument("--validate-only", action="store_true", help="no Docker or model calls")
    args = parser.parse_args(argv)
    try:
        result = run(
            args.dataset, args.split, args.models, args.output,
            model_keys=args.models_selected, question_ids=args.question_ids,
            seed=args.seed, max_steps=args.max_steps, validate_only=args.validate_only,
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
