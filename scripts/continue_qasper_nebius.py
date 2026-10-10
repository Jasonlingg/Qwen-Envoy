"""Safely finish the two interrupted QASPER/Nebius questions.

Dry-run by default. ``--execute`` makes paid requests, with the original
usage carried into a $25 *total* catalog-rate estimate guard. Question 39
restarts at step one because the original REPL/conversation cannot be restored.
The original partial run is never modified.
"""

# ruff: noqa: E402 -- allow direct `python scripts/...` invocation from the repo.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.envoybench import run as runner
from benchmarks.envoybench.budget import InferenceBudget
from src.env.tools import SEARCH_PROTOCOL_VERSION, TOOL_PREAMBLE
from src.eval.artifacts import configuration_hash
from src.policies.code_execution import QASPER_SYSTEM_PROMPT

RELEASE = ROOT / "release/qasper-agent-study"
ORIGINAL = RELEASE / "nebius-run"
DATASET = ROOT / "benchmarks/envoybench/data"
MODELS = RELEASE / "models.json"
DEFAULT_OUTPUT = RELEASE / "nebius-continuation"
MODEL_KEY = "nemotron_ultra_nebius"
TOTAL_ESTIMATED_USD = 25.0
MAX_REQUESTS = 600

# This is the sole known source difference from the original recorded runner.
# Replacing this exact line wrap must recover the original file hash.
_OLD_RUNNER_LINES = (
    '                    raise RuntimeError('
    'f"{key}: incomplete question matrix; refusing complete artifact")\n'
)
_NEW_RUNNER_LINES = (
    "                    raise RuntimeError(\n"
    '                        f"{key}: incomplete question matrix; refusing complete artifact"\n'
    "                    )\n"
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _json(path: Path) -> dict | list:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid JSON: {path}") from exc


def _sha256(path: Path) -> str:
    return runner._file_sha256(path)


def _pinned_source_hashes(original: Path) -> dict[str, str]:
    """Check the three immutable source files against the release checksum list."""
    release = original.parent
    try:
        lines = (release / "SHA256SUMS").read_text().splitlines()
    except OSError as exc:
        raise ValueError("original release SHA256SUMS is missing") from exc
    expected = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) == 2:
            expected[parts[1]] = parts[0]
    result = {}
    for name in ("manifest.json", "results.partial.json", "usage-budget.json"):
        relative = f"{original.name}/{name}"
        _require(relative in expected, f"source checksum is missing: {relative}")
        actual = _sha256(original / name)
        _require(actual == expected[relative], f"source checksum differs: {relative}")
        result[name] = actual
    return result


def _implementation_variance(manifest: dict) -> dict | None:
    expected = manifest.get("implementation_sha256")
    _require(isinstance(expected, dict) and expected, "original implementation hashes are missing")
    actual = runner._implementation_hashes()
    _require(
        set(actual) == set(expected),
        "implementation differs: file set differs from original",
    )
    variance = None
    for name, current_hash in actual.items():
        original_hash = expected[name]
        if current_hash == original_hash:
            continue
        _require(name == "benchmarks/envoybench/run.py", f"implementation differs: {name}")
        source = (ROOT / name).read_text()
        _require(source.count(_NEW_RUNNER_LINES) == 1, "runner differs beyond allowed formatting")
        restored = source.replace(_NEW_RUNNER_LINES, _OLD_RUNNER_LINES, 1)
        restored_hash = hashlib.sha256(restored.encode()).hexdigest()
        _require(restored_hash == original_hash, "runner differs beyond allowed formatting")
        variance = {
            "file": name,
            "original_sha256": original_hash,
            "continuation_sha256": current_hash,
            "reason": "exact line wrap of an unchanged RuntimeError statement",
        }
    return variance


def prepare_continuation(
    *, original: Path = ORIGINAL, dataset: Path = DATASET,
    models: Path = MODELS, output: Path = DEFAULT_OUTPUT,
) -> dict:
    """Return a validated, secret-free plan without Docker or model calls."""
    original = original.resolve(strict=True)
    output = output.resolve()
    _require(not output.exists(), f"output already exists: {output}")
    _require(not output.is_relative_to(original), "output must not be inside the original run")
    source_hashes = _pinned_source_hashes(original)
    manifest = _json(original / "manifest.json")
    rows = _json(original / "results.partial.json")
    usage = _json(original / "usage-budget.json")
    _require(isinstance(manifest, dict) and isinstance(rows, list) and isinstance(usage, dict),
             "original run has invalid artifact types")
    ids = manifest.get("question_ids")
    _require(manifest.get("schema_version") == runner.RUN_SCHEMA, "original run schema differs")
    _require(manifest.get("status") == "incomplete", "original run is not incomplete")
    _require(not (original / "results.json").exists(), "original run has a complete results file")
    _require(isinstance(ids, list) and len(ids) == 40 and len(set(ids)) == 40,
             "original run does not have 40 unique ordered questions")
    _require(manifest.get("split") == "test_candidate" and
             manifest.get("split_question_count") == 40 and
             manifest.get("full_split") is True and manifest.get("subset_smoke") is False,
             "original run is not the full 40-question candidate split")
    _require(manifest.get("max_steps") == 15 and manifest.get("seed") == 42,
             "original step limit or seed differs")
    _require(manifest.get("models") and len(manifest["models"]) == 1 and
             manifest["models"][0].get("key") == MODEL_KEY and
             manifest["models"][0].get("api_key_env") == "NEBIUS_API_KEY",
             "original model identity differs")
    _require(manifest.get("model_config_sha256") == _sha256(models),
             "model configuration differs from original")

    benchmark, snapshot, benchmark_path, _corpus_path, entry = runner.load_split(
        dataset, "test_candidate"
    )
    frozen_ids = [question["id"] for question in benchmark["questions"]]
    _require(ids == frozen_ids, "original question order differs from frozen split")
    _require(manifest.get("benchmark_id") == benchmark["benchmark_id"] and
             manifest.get("benchmark_hash") == configuration_hash(benchmark) and
             manifest.get("corpus_hash") == benchmark["corpus_hash"] and
             manifest.get("benchmark_file_sha256") == _sha256(benchmark_path) and
             manifest.get("dataset_index_sha256") == _sha256(dataset / "manifest.json") and
             manifest.get("split_status") == snapshot.get(
                 "split_status", entry.get("split_status")
             ),
             "dataset/source bindings differ from original")
    _require(manifest.get("system_prompt_sha256") ==
             hashlib.sha256(QASPER_SYSTEM_PROMPT.encode()).hexdigest() and
             manifest.get("tool_preamble_sha256") ==
             hashlib.sha256(TOOL_PREAMBLE.encode()).hexdigest() and
             manifest.get("tool_search_version") == SEARCH_PROTOCOL_VERSION and
             manifest.get("require_evidence") is True and
             manifest.get("evidence_verifier") is True and
             manifest.get("verifier_feedback_budget") == 1 and
             manifest.get("escalate_after_verifier_failure") is False and
             manifest.get("docker_sandbox") == runner.SANDBOX_LABEL,
             "prompt, tools, or execution protocol differs from original")
    variance = _implementation_variance(manifest)

    _require(len(rows) == 39, "expected exactly 39 saved original rows")
    for index, row in enumerate(rows):
        _require(isinstance(row, dict) and row.get("question_id") == ids[index] and
                 row.get("run_id") == manifest.get("run_id") and
                 row.get("comparison_id") == manifest.get("comparison_id") and
                 row.get("model_key") == MODEL_KEY and
                 row.get("benchmark_hash") == manifest["benchmark_hash"] and
                 row.get("corpus_hash") == manifest["corpus_hash"],
                 f"original row {index + 1} does not bind to manifest/order")
        if index < 38:
            _require(row.get("status") in {"submitted", "no_submission", "escalated"} and
                     row.get("environment_status") != "error",
                     f"original question {index + 1} was not terminal")
    interrupted = rows[38]
    _require(interrupted.get("status") == "error" and
             interrupted.get("environment_status") == "error" and
             interrupted.get("error") == (
                 "RuntimeError: estimated inference cost budget exhausted"
             ) and
             interrupted.get("steps") == 11,
             "original question 39 is not the recorded budget interruption")

    original_config = manifest.get("inference_budget")
    _require(isinstance(original_config, dict) and usage.get("config") == original_config and
             original_config.get("max_estimated_usd") == 2.0 and
             original_config.get("max_requests") == MAX_REQUESTS and
             original_config.get("input_usd_per_million") == 1.0 and
             original_config.get("output_usd_per_million") == 3.0 and
             original_config.get("max_input_utf8_bytes") == 100000,
             "original budget configuration differs")
    counters = {
        "requests": 366,
        "estimated_usd": 1.947862,
        "prompt_tokens": 1819534,
        "completion_tokens": 42776,
        "missing_usage_requests": 0,
    }
    for key, expected in counters.items():
        actual = usage.get(key)
        _require(type(actual) in (int, float) and math.isfinite(actual) and actual == expected,
                 f"original usage {key} differs")
    _require(usage.get("halted_reason") == "estimated inference cost budget exhausted",
             "original usage halt reason differs")
    _require(counters["requests"] < MAX_REQUESTS and
             counters["estimated_usd"] < TOTAL_ESTIMATED_USD,
             "original usage already exceeds continuation bounds")

    total_config = {**original_config, "max_estimated_usd": TOTAL_ESTIMATED_USD,
                    "max_requests": MAX_REQUESTS}
    # Validate the extended guard before any paid action.
    InferenceBudget(total_config)
    return {
        "schema_version": "qasper-nebius-continuation-v1",
        "original_run_id": manifest["run_id"],
        "original_files_sha256": source_hashes,
        "original_usage": counters,
        "continuation_question_ids": ids[38:40],
        "question_39_restarted_at_step": 1,
        "question_39_original_interrupted_after_steps": 11,
        "total_inference_budget": total_config,
        "implementation_variance": variance,
        "output": str(output),
    }


def _seed_budget(plan: dict) -> InferenceBudget:
    budget = InferenceBudget(plan["total_inference_budget"])
    for key, value in plan["original_usage"].items():
        setattr(budget, key, value)
    # The previous halt applied to the old $2 guard, not the new $25 guard.
    budget.halted_reason = None
    return budget


def execute_continuation(
    plan: dict, *, dataset: Path = DATASET, models: Path = MODELS,
    env_file: Path = ROOT / ".env",
) -> dict:
    """Perform the two paid episodes after an explicit CLI ``--execute`` gate."""
    load_dotenv(env_file, override=False)
    _require(bool(os.environ.get("NEBIUS_API_KEY", "").strip()),
             "NEBIUS_API_KEY is required in the environment or repository .env")
    output = Path(plan["output"])
    _require(not output.exists(), f"output already exists: {output}")
    budget = _seed_budget(plan)
    try:
        return runner.run(
            dataset, "test_candidate", models, output,
            model_keys=[MODEL_KEY], question_ids=plan["continuation_question_ids"],
            seed=42, max_steps=15, budget=budget, verifier_protocol="legacy-v1",
        )
    finally:
        if output.is_dir():
            lineage = {key: value for key, value in plan.items() if key != "output"}
            lineage["continuation_usage"] = budget.snapshot()
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                extension = _json(manifest_path)
                lineage["continuation_run_id"] = extension.get("run_id")
                lineage["continuation_status"] = extension.get("status")
            runner._write_json(output / "continuation-lineage.json", lineage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="permit the two paid episodes")
    args = parser.parse_args(argv)
    try:
        plan = prepare_continuation(
            original=args.original, dataset=args.dataset,
            models=args.models, output=args.output,
        )
        if not args.execute:
            print(json.dumps({**plan, "mode": "dry_run"}, indent=2))
            return 0
        result = execute_continuation(plan, dataset=args.dataset, models=args.models)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"QASPER Nebius continuation: {exc}\n")
    print(f"QASPER Nebius continuation complete: {args.output} "
          f"({len(result['question_ids'])} questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
