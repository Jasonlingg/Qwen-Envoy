"""Combine the frozen 38 completed Nebius episodes with a two-question retry.

This is an offline artifact operation. It never resumes an agent or contacts a
model. The original 39-row partial attempt remains the record of what happened
under the initial $2 protocol; the output is a separately identified, amended
40-question result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.envoybench.run import DEFAULT_DATASET, RUN_SCHEMA, load_split
from benchmarks.envoybench.score import _verify_artifacts
from src.eval.artifacts import configuration_hash

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RUN = ROOT / "release/qasper-agent-study/nebius-run"
TOTAL_ESTIMATED_USD_CAP = 25.0
DERIVATION_SCHEMA = "qasper-nebius-continuation-merge-v1"
CONTINUATION_SCHEMA = "qasper-nebius-continuation-v1"
ORIGINAL_RUNNER_SHA256 = "345b6d766eaa0d023418386b0931c94fb4cd40fa974d1576dce79a46ce10b463"
FORMATTED_RUNNER_SHA256 = "5169a7acc243a92c472bd380fbb42fc5f98561391bc974b3c99642e24e963bf5"

PROTOCOL_FIELDS = (
    "benchmark_id", "benchmark_hash", "corpus_hash", "question_ids", "split",
    "max_steps", "seed", "seed_note", "system_prompt_sha256",
    "system_prompt_name", "tool_preamble_sha256", "tool_search_version",
    "require_evidence", "evidence_verifier", "verifier_feedback_budget",
    "escalate_after_verifier_failure", "docker_sandbox",
)
SAME_PROTOCOL_FIELDS = tuple(field for field in PROTOCOL_FIELDS if field != "question_ids") + (
    "schema_version", "split_status", "split_question_count",
    "benchmark_file_sha256", "dataset_index_sha256", "model_config_sha256",
    "models", "model_identity_note", "legacy_reward_version",
    "legacy_reward_is_benchmark_score", "sandbox_image", "runner_hardware",
)
ROW_IDENTITY_FIELDS = ("run_id", "comparison_id")
ROW_BINDING_FIELDS = (
    "schema_version", "run_id", "comparison_id", "benchmark_id",
    "benchmark_hash", "corpus_hash", "split", "split_status",
)
USAGE_COUNTERS = (
    "requests", "estimated_usd", "prompt_tokens", "completion_tokens",
    "missing_usage_requests",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path, expected: type) -> dict | list:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid JSON: {path}") from exc
    _require(isinstance(value, expected), f"unexpected JSON shape: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _check_original_release_hashes(original_run: Path, file_hashes: dict[str, str]) -> None:
    """Reject drift in the published partial artifact used for this extension."""
    if original_run != ORIGINAL_RUN.resolve():
        return
    try:
        lines = (original_run.parent / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("original release checksum list is missing") from exc
    expected = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) == 2:
            expected[parts[1]] = parts[0]
    for name, actual in file_hashes.items():
        release_name = f"{original_run.name}/{name}"
        _require(expected.get(release_name) == actual,
                 f"original release checksum differs: {release_name}")


def _check_comparison_id(manifest: dict, name: str) -> None:
    try:
        protocol = {field: manifest[field] for field in PROTOCOL_FIELDS}
    except KeyError as exc:
        raise ValueError(f"{name} is missing protocol field {exc.args[0]}") from exc
    _require(
        manifest.get("comparison_id") == configuration_hash(protocol),
        f"{name} comparison_id does not bind its declared protocol",
    )


def _check_rows(manifest: dict, rows: list, question_ids: list[str],
                questions: dict[str, str], name: str) -> None:
    _require(len(rows) == len(question_ids), f"{name} row count differs from question selection")
    _require(
        [row.get("question_id") if isinstance(row, dict) else None for row in rows]
        == question_ids,
        f"{name} question IDs/order differ from declared selection",
    )
    _require(len(set(question_ids)) == len(question_ids), f"{name} repeats a question ID")
    model_key = manifest["models"][0]["key"]
    for row in rows:
        qid = row["question_id"]
        _require(row.get("model_key") == model_key, f"{name} {qid} model key mismatch")
        _require(row.get("question") == questions[qid], f"{name} {qid} question text mismatch")
        for field in ROW_BINDING_FIELDS:
            _require(row.get(field) == manifest.get(field), f"{name} {qid} {field} mismatch")
        _require(row["schema_version"] == RUN_SCHEMA, f"{name} {qid} schema mismatch")
        _require(row.get("status") in {"submitted", "no_submission", "error", "escalated"},
                 f"{name} {qid} invalid status")
        _require(isinstance(row.get("predicted_answer"), str), f"{name} {qid} invalid answer")
        _require(isinstance(row.get("predicted_citations"), list),
                 f"{name} {qid} invalid citations")
        _require(isinstance(row.get("predicted_evidence"), list),
                 f"{name} {qid} invalid evidence")
        trajectory = row.get("trajectory")
        _require(isinstance(trajectory, list), f"{name} {qid} invalid trajectory")
        _require(type(row.get("steps")) is int and row["steps"] == len(trajectory),
                 f"{name} {qid} steps differ from saved trajectory")
        duration = row.get("duration_seconds")
        _require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
                 f"{name} {qid} invalid duration")


def _check_compatibility(original: dict, continuation: dict) -> dict | None:
    for field in SAME_PROTOCOL_FIELDS:
        _require(
            original.get(field) == continuation.get(field),
            f"continuation {field} differs from original protocol",
        )
    original_hashes = original.get("implementation_sha256")
    continued_hashes = continuation.get("implementation_sha256")
    _require(isinstance(original_hashes, dict) and isinstance(continued_hashes, dict)
             and original_hashes.keys() == continued_hashes.keys(),
             "continuation implementation hash set differs from original")
    runner = "benchmarks/envoybench/run.py"
    for path in original_hashes:
        if path == runner:
            continue
        _require(original_hashes[path] == continued_hashes[path],
                 f"continuation implementation hash differs: {path}")
    old_hash, new_hash = original_hashes[runner], continued_hashes[runner]
    if old_hash == new_hash:
        return None
    _require(
        (old_hash, new_hash) == (ORIGINAL_RUNNER_SHA256, FORMATTED_RUNNER_SHA256),
        "continuation runner implementation hash differs beyond the pinned line-wrap",
    )
    return {
        "file": runner, "original_sha256": old_hash, "continuation_sha256": new_hash,
        "scope": (
            "Only the RuntimeError line wrapping changed; the executed statement is identical."
        ),
    }


def _check_usage(original: dict, continuation: dict, original_manifest: dict,
                 continuation_manifest: dict) -> dict:
    old_config, new_config = original.get("config"), continuation.get("config")
    _require(isinstance(old_config, dict) and isinstance(new_config, dict),
             "usage ledgers need budget configurations")
    _require(old_config == original_manifest.get("inference_budget"),
             "original usage config differs from manifest")
    _require(new_config == continuation_manifest.get("inference_budget"),
             "continuation usage config differs from manifest")
    _require(old_config.get("max_estimated_usd") == 2,
             "original local estimated cost ceiling is not $2")
    _require(new_config.get("max_estimated_usd") == TOTAL_ESTIMATED_USD_CAP,
             "continuation local estimated cost ceiling must be $25 cumulative")
    for field in ("input_usd_per_million", "output_usd_per_million", "max_requests",
                  "max_input_utf8_bytes", "price_source"):
        _require(old_config.get(field) == new_config.get(field),
                 f"continuation budget {field} differs from original")
    _require(original.get("halted_reason") == "estimated inference cost budget exhausted",
             "original ledger does not identify the recorded budget stop")
    _require(continuation.get("halted_reason") is None,
             "continuation ledger records a halt; cannot claim complete merge")
    deltas = {}
    for field in USAGE_COUNTERS:
        old, new = original.get(field), continuation.get(field)
        numeric = (int, float) if field == "estimated_usd" else (int,)
        _require(type(old) in numeric and type(new) in numeric,
                 f"usage {field} has invalid numbers")
        _require(math.isfinite(old) and math.isfinite(new) and 0 <= old <= new,
                 f"continuation usage {field} does not carry forward original ledger")
        deltas[field] = round(new - old, 8) if field == "estimated_usd" else new - old
    _require(deltas["requests"] > 0, "continuation records no new provider requests")
    _require(continuation["estimated_usd"] <= TOTAL_ESTIMATED_USD_CAP,
             "cumulative estimated usage exceeds $25")
    _require(continuation["requests"] <= new_config["max_requests"],
             "cumulative request count exceeds the declared cap")
    return deltas


def merge_continuation(continuation_run: Path, output: Path, *,
                       original_run: Path = ORIGINAL_RUN,
                       dataset: Path = DEFAULT_DATASET) -> dict:
    """Validate and write a new complete run without altering either source."""
    original_run = original_run.resolve(strict=True)
    continuation_run = continuation_run.resolve(strict=True)
    output = output.resolve()
    _require(not output.exists(), f"output already exists: {output}")
    _require(output != original_run and output != continuation_run,
             "output must differ from both source runs")
    _require(not output.is_relative_to(original_run)
             and not output.is_relative_to(continuation_run),
             "output must not be nested inside a source run")
    _require(not (original_run / "results.json").exists(),
             "original run unexpectedly has completed results")
    _require(not (continuation_run / "results.partial.json").exists(),
             "continuation unexpectedly retains partial results")

    old_manifest = _read_json(original_run / "manifest.json", dict)
    old_rows = _read_json(original_run / "results.partial.json", list)
    old_usage = _read_json(original_run / "usage-budget.json", dict)
    new_manifest = _read_json(continuation_run / "manifest.json", dict)
    new_rows = _read_json(continuation_run / "results.json", list)
    new_usage = _read_json(continuation_run / "usage-budget.json", dict)
    _require(old_manifest.get("status") == "incomplete", "original run is not incomplete")
    _require(new_manifest.get("status") == "complete", "continuation run is not complete")
    _require(old_manifest.get("run_id") != new_manifest.get("run_id"),
             "continuation must have a separate source run ID")
    _require(len(old_manifest.get("models", [])) == len(new_manifest.get("models", [])) == 1,
             "merge requires exactly one declared model in each run")
    _require(old_manifest.get("full_split") is True and old_manifest.get("subset_smoke") is False,
             "original run did not declare the full split")
    _require(new_manifest.get("full_split") is False and new_manifest.get("subset_smoke") is True,
             "continuation must declare the two-question subset")

    benchmark, _, _, corpus_path, _ = load_split(dataset, old_manifest["split"])
    all_ids = [question["id"] for question in benchmark["questions"]]
    _require(len(all_ids) == 40 and old_manifest.get("question_ids") == all_ids,
             "original question selection/order is not the frozen 40-question split")
    last_ids = all_ids[38:]
    _require(new_manifest.get("question_ids") == last_ids,
             "continuation must contain only questions 39 and 40, in frozen order")
    _check_comparison_id(old_manifest, "original")
    _check_comparison_id(new_manifest, "continuation")
    runner_variance = _check_compatibility(old_manifest, new_manifest)
    questions = {question["id"]: question["question"] for question in benchmark["questions"]}
    _check_rows(old_manifest, old_rows, all_ids[:39], questions, "original")
    _check_rows(new_manifest, new_rows, last_ids, questions, "continuation")
    _require(all(row["status"] in {"submitted", "no_submission", "escalated"}
                 and row.get("environment_status") != "error" for row in new_rows),
             "continuation contains an error episode; cannot claim a completed extension")
    _require(all(row["status"] != "error" and row.get("environment_status") != "error"
                 for row in old_rows[:38]),
             "original first 38 episodes were not all terminal before the budget stop")
    _verify_artifacts(
        benchmark, old_manifest, old_rows, corpus_path,
        require_paired=False, allow_incomplete=True,
    )
    interrupted = old_rows[38]
    _require(interrupted["status"] == "error" and interrupted["error"] ==
             "RuntimeError: estimated inference cost budget exhausted",
             "original question 39 is not the recorded budget interruption")
    _require(interrupted["steps"] == 11 and len(interrupted["trajectory"]) == 11,
             "original question 39 interrupted trace length changed")
    _require(new_rows[0]["trajectory"] != interrupted["trajectory"],
             "question 39 continuation reuses the interrupted trace")
    _require(new_rows[0].get("error") != interrupted["error"],
             "question 39 continuation still has the original budget error")
    usage_delta = _check_usage(old_usage, new_usage, old_manifest, new_manifest)

    original_files = {
        name: _file_sha256(original_run / name)
        for name in ("manifest.json", "results.partial.json", "usage-budget.json")
    }
    _check_original_release_hashes(original_run, original_files)
    continuation_files = {
        name: _file_sha256(continuation_run / name)
        for name in ("manifest.json", "results.json", "usage-budget.json")
    }
    continuation_lineage_path = continuation_run / "continuation-lineage.json"
    _require(continuation_lineage_path.is_file(),
             "continuation is missing its source-binding lineage record")
    continuation_lineage = _read_json(continuation_lineage_path, dict)
    _require(continuation_lineage.get("schema_version") == CONTINUATION_SCHEMA,
             "continuation lineage schema mismatch")
    _require(continuation_lineage.get("original_run_id") == old_manifest["run_id"]
             and continuation_lineage.get("original_files_sha256") == original_files,
             "continuation lineage does not bind the original source files")
    _require(continuation_lineage.get("original_usage") ==
             {field: old_usage[field] for field in USAGE_COUNTERS}
             and continuation_lineage.get("total_inference_budget") == new_usage["config"],
             "continuation lineage does not bind the carried-forward budget")
    _require(continuation_lineage.get("continuation_usage") == new_usage,
             "continuation lineage does not bind the final cumulative usage")
    _require(continuation_lineage.get("continuation_question_ids") == last_ids
             and continuation_lineage.get("question_39_restarted_at_step") == 1
             and continuation_lineage.get("question_39_original_interrupted_after_steps") == 11,
             "continuation lineage does not disclose the exact retry")
    _require(continuation_lineage.get("continuation_run_id") == new_manifest["run_id"]
             and continuation_lineage.get("continuation_status") == "complete",
             "continuation lineage does not bind the completed subset run")
    declared_variance = continuation_lineage.get("implementation_variance")
    if runner_variance is None:
        _require(declared_variance is None,
                 "continuation lineage declares an unobserved runner variance")
    else:
        _require(isinstance(declared_variance, dict) and all(
            declared_variance.get(field) == runner_variance[field]
            for field in ("file", "original_sha256", "continuation_sha256")
        ), "continuation lineage runner variance differs from validated hashes")
    continuation_files["continuation-lineage.json"] = _file_sha256(continuation_lineage_path)

    amendment = {
        "schema_version": DERIVATION_SCHEMA,
        "kind": "cumulative_budget_increase_and_question_39_restart",
        "original_local_estimated_usd_cap": 2.0,
        "amended_cumulative_estimated_usd_cap": TOTAL_ESTIMATED_USD_CAP,
        "original_no_retry_protocol_amended": True,
        "question_39_restarted_at_step": 1,
        "original_question_39_partial_trace_excluded_from_derived_result": True,
        "question_39_source": "continuation retry",
        "question_40_source": "continuation first attempt",
        "first_38_source": "original completed episodes",
    }
    amended_protocol = {
        **{field: old_manifest[field] for field in PROTOCOL_FIELDS},
        "amendment": amendment,
        "original_comparison_id": old_manifest["comparison_id"],
        "continuation_comparison_id": new_manifest["comparison_id"],
    }
    comparison_id = configuration_hash(amended_protocol)
    run_id = _value_sha256({
        "derivation_schema": DERIVATION_SCHEMA,
        "comparison_id": comparison_id,
        "original_files_sha256": original_files,
        "continuation_files_sha256": continuation_files,
    })[:32]
    derived_rows = []
    for source in [*old_rows[:38], *new_rows]:
        row = copy.deepcopy(source)
        row["run_id"] = run_id
        row["comparison_id"] = comparison_id
        derived_rows.append(row)
    _require(
        all({key: value for key, value in row.items() if key not in ROW_IDENTITY_FIELDS}
            == {key: value for key, value in source.items() if key not in ROW_IDENTITY_FIELDS}
            for row, source in zip(derived_rows[:38], old_rows[:38])),
        "first 38 source rows changed beyond derived identity fields",
    )
    lineage = {
        "schema_version": DERIVATION_SCHEMA,
        "original": {
            "run_id": old_manifest["run_id"],
            "comparison_id": old_manifest["comparison_id"],
            "path": os.path.relpath(original_run, output),
            "files_sha256": original_files,
            "retained_question_ids": all_ids[:38],
            "retained_row_sha256": [_value_sha256(row) for row in old_rows[:38]],
            "implementation_sha256": old_manifest["implementation_sha256"],
            "git_commit": old_manifest.get("git_commit"),
            "interrupted_question_39": {
                "question_id": last_ids[0],
                "source_file": "results.partial.json", "source_row_index": 38,
                "row_sha256": _value_sha256(interrupted),
                "trajectory_sha256": _value_sha256(interrupted["trajectory"]),
                "steps": 11, "status": interrupted["status"], "error": interrupted["error"],
            },
            "usage": {field: old_usage[field] for field in USAGE_COUNTERS},
        },
        "continuation": {
            "run_id": new_manifest["run_id"],
            "comparison_id": new_manifest["comparison_id"],
            "path": os.path.relpath(continuation_run, output),
            "files_sha256": continuation_files,
            "question_ids": last_ids,
            "implementation_sha256": new_manifest["implementation_sha256"],
            "git_commit": new_manifest.get("git_commit"),
            "usage_delta": usage_delta,
        },
        "row_identity_rebinding": {
            "fields": list(ROW_IDENTITY_FIELDS),
            "reason": "The verifier binds every saved row to the derived run/protocol IDs.",
        },
    }
    if runner_variance is not None:
        lineage["runner_hash_variance"] = runner_variance
    derived_manifest = copy.deepcopy(old_manifest)
    derived_manifest.update({
        "run_id": run_id,
        "comparison_id": comparison_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": new_manifest.get("completed_at_utc"),
        "status": "complete",
        "inference_budget": copy.deepcopy(new_usage["config"]),
        "git_commit": None,
        "implementation_sha256_scope": (
            "First 38 rows use the original implementation; questions 39 and 40 use "
            "the continuation implementation. See lineage for both hash sets."
        ),
        "protocol_amendment": amendment,
        "lineage": lineage,
    })
    derived_manifest.pop("elapsed_seconds", None)
    _verify_artifacts(benchmark, derived_manifest, derived_rows, corpus_path,
                      require_paired=False)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "manifest.json", derived_manifest)
    _write_json(output / "results.json", derived_rows)
    _write_json(output / "usage-budget.json", new_usage)
    return derived_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = merge_continuation(args.continuation, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Nebius continuation merge: {exc}\n")
    print(json.dumps({
        "run_id": manifest["run_id"], "comparison_id": manifest["comparison_id"],
        "status": manifest["status"], "questions": len(manifest["question_ids"]),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
