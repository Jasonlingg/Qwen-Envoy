"""Sanitize and summarize saved EnvoyBench generated-token logprobs.

This is an offline execution diagnostic. It does not score answer quality,
calibrate confidence, or request model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.envoybench.score import (
    HARNESS_ERROR_MARKER,
    RUNTIME_ERROR_MARKER,
    _error_diagnostics,
    _structured_tool_error_count,
)

SCHEMA_VERSION = "envoybench-logprob-analysis-v1"
COVERAGE_FLOOR = 0.8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(directory: Path) -> tuple[dict, list[dict], dict]:
    manifest_path = directory / "manifest.json"
    results_path = directory / "results.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(results, list) or not isinstance(manifest.get("models"), list):
        raise ValueError(f"invalid run shape: {directory}")
    model_keys = [model["key"] for model in manifest["models"]]
    question_ids = manifest["question_ids"]
    if len(set(model_keys)) != len(model_keys) or len(set(question_ids)) != len(question_ids):
        raise ValueError(f"duplicate model or question ID: {directory}")
    expected = {
        (question_id, model_key)
        for question_id in question_ids
        for model_key in model_keys
    }
    actual = [(row["question_id"], row["model_key"]) for row in results]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f"results do not contain exactly the declared pairs: {directory}")
    for row in results:
        for key in ("run_id", "comparison_id", "benchmark_hash", "corpus_hash", "split"):
            if row.get(key) != manifest.get(key):
                raise ValueError(f"result/manifest {key} mismatch: {directory}")
    hashes = {"manifest_sha256": _sha256(manifest_path), "results_sha256": _sha256(results_path)}
    return manifest, results, hashes


def _model_provenance(model: dict) -> dict:
    body = model.get("extra_body") or {}
    chat = body.get("chat_template_kwargs") or {}
    return {
        "key": model["key"],
        "model_id": model["model_id"],
        "revision": model.get("revision"),
        "serving_runtime": model.get("serving_runtime"),
        "serving_hardware": model.get("serving_hardware"),
        "decoding": {
            key: model["decoding"].get(key)
            for key in ("temperature", "top_p", "max_tokens")
        },
        "logprobs_requested": body.get("logprobs"),
        "top_logprobs_requested": body.get("top_logprobs"),
        "thinking_enabled": chat.get("enable_thinking"),
    }


def _provenance(manifest: dict, hashes: dict) -> dict:
    keys = (
        "run_id", "comparison_id", "created_at_utc", "completed_at_utc",
        "split", "split_status", "subset_smoke", "full_split", "question_ids",
        "benchmark_hash", "corpus_hash", "benchmark_file_sha256",
        "dataset_index_sha256", "implementation_sha256", "system_prompt_sha256",
        "tool_preamble_sha256", "git_commit", "seed", "seed_note", "max_steps",
        "legacy_reward_version", "runner_hardware", "sandbox_image",
    )
    return {
        **{key: manifest.get(key) for key in keys},
        "models": [_model_provenance(model) for model in manifest["models"]],
        **hashes,
    }


def _error_class(observation: str) -> str:
    returned = _structured_tool_error_count(observation) > 0
    stderr = observation.split("\nSTDERR:\n", 1)[-1] if "\nSTDERR:\n" in observation else ""
    runtime = bool(RUNTIME_ERROR_MARKER.search(stderr) or HARNESS_ERROR_MARKER.search(observation))
    if returned and runtime:
        return "returned_tool_and_runtime_error"
    if returned:
        return "returned_tool_error"
    if runtime:
        return "runtime_or_harness_error"
    return "no_detected_execution_error"


def _logprob_summary(step: dict) -> dict:
    values = step.get("logprob_diagnostics")
    if not isinstance(values, dict):
        return {"usable": False, "reason": "missing_diagnostics"}
    reported, valid = values.get("reported_token_count"), values.get("valid_logprob_count")
    mean, total = values.get("mean_logprob"), values.get("sum_logprob")
    if (
        values.get("schema_version") != "sampled-content-logprobs-v1"
        or values.get("scope") != (
            "provider generated content; not aligned to cleaned action or answer correctness"
        )
        or type(reported) is not int or type(valid) is not int
        or reported < valid or valid < 1
        or not isinstance(mean, (int, float)) or not isinstance(total, (int, float))
        or not math.isfinite(mean) or not math.isfinite(total)
        or not math.isclose(mean * valid, total, rel_tol=1e-5, abs_tol=1e-4)
    ):
        return {"usable": False, "reason": "invalid_or_empty_diagnostics"}
    return {
        "usable": True,
        "reported_token_count": reported,
        "valid_logprob_count": valid,
        "mean_logprob": round(float(mean), 6),
        "sum_logprob": round(float(total), 6),
        "prefix_truncated": bool(values.get("prefix_truncated")),
    }


def _step_summary(step: dict) -> dict:
    action = step.get("action")
    if not isinstance(action, str):
        raise ValueError("saved action must be a string")
    kind = "submission" if action.lstrip().upper().startswith("SUBMIT:") else "code"
    observation = step.get("observation", "")
    if not isinstance(observation, str):
        raise ValueError("saved observation must be a string")
    return {
        "step": step["step"],
        "kind": kind,
        "immediate_error_class": _error_class(observation) if kind == "code" else "not_applicable",
        **_logprob_summary(step),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 6)


def _distribution(values: list[float]) -> dict:
    return {
        "n": len(values),
        "q25": _percentile(values, 0.25),
        "median": round(statistics.median(values), 6) if values else None,
        "q75": _percentile(values, 0.75),
    }


def _analyze(manifest: dict, rows: list[dict], *, include_cases: bool) -> dict:
    model_keys = [model["key"] for model in manifest["models"]]
    counters: dict[str, Counter] = {key: Counter() for key in model_keys}
    groups: dict[str, dict[str, list[float]]] = {
        key: defaultdict(list) for key in model_keys
    }
    error_classes: dict[str, Counter] = {key: Counter() for key in model_keys}
    by_pair: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = row["model_key"]
        count = counters[key]
        count["rows"] += 1
        count[f"status_{row['status']}"] += 1
        summaries = []
        all_error_classes = []
        for step in row["trajectory"]:
            summary = _step_summary(step)
            summaries.append(summary)
            all_error_classes.append(_error_class(step.get("observation", "")))
            count["generated_turns"] += 1
            if summary["usable"]:
                count["usable_generated_turns"] += 1
                count["reported_tokens"] += summary["reported_token_count"]
                count["valid_token_logprobs"] += summary["valid_logprob_count"]
            if summary["kind"] == "code":
                count["code_turns"] += 1
                has_error = summary["immediate_error_class"] != "no_detected_execution_error"
                count["code_turns_with_immediate_error"] += int(has_error)
                error_classes[key][summary["immediate_error_class"]] += 1
                if summary["usable"]:
                    count["usable_code_turns"] += 1
                    group = "immediate_error" if has_error else "clean_observation"
                    groups[key][group].append(summary["mean_logprob"])
        scorer_counts = _error_diagnostics(row)
        count["endpoint_error_rows"] += int(scorer_counts["endpoint_error_episode"])
        count["context_limit_error_rows"] += int(scorer_counts["context_limit_error_episode"])
        count["other_harness_error_rows"] += int(scorer_counts["other_harness_error_episode"])
        if (
            sum("returned_tool" in name for name in all_error_classes)
            != scorer_counts["tool_error_steps"]
            or sum(name in {"runtime_or_harness_error", "returned_tool_and_runtime_error"}
                   for name in all_error_classes)
            != scorer_counts["runtime_error_steps"]
        ):
            raise ValueError("step error classification diverges from EnvoyBench scorer")
        by_pair[(row["question_id"], key)] = {
            "status": row["status"],
            "generated_turn_count": len(summaries),
            "steps": summaries,
        }

    model_summary = {}
    for key in model_keys:
        count = counters[key]
        eligible = count["code_turns"]
        covered = count["usable_code_turns"]
        coverage = covered / eligible if eligible else 0.0
        error = groups[key]["immediate_error"]
        clean = groups[key]["clean_observation"]
        comparable = coverage >= COVERAGE_FLOOR and bool(error) and bool(clean)
        model_summary[key] = {
            **{name: count[name] for name in (
                "rows", "status_submitted", "status_error",
                "endpoint_error_rows", "context_limit_error_rows",
                "other_harness_error_rows", "generated_turns",
                "usable_generated_turns", "reported_tokens", "valid_token_logprobs",
                "code_turns", "code_turns_with_immediate_error", "usable_code_turns",
            )},
            "code_turn_logprob_coverage": round(coverage, 4),
            "code_turn_immediate_error_class_counts": dict(sorted(error_classes[key].items())),
            "comparison_status": "descriptive_only" if comparable else "coverage_only",
            "immediate_error_step_mean_logprob": _distribution(error),
            "clean_step_mean_logprob": _distribution(clean),
            "median_error_minus_clean": (
                round(statistics.median(error) - statistics.median(clean), 6)
                if comparable else None
            ),
        }

    paired_submissions = sum(
        all(by_pair[(qid, key)]["status"] == "submitted" for key in model_keys)
        for qid in manifest["question_ids"]
    )
    result = {
        "declared_question_count": len(manifest["question_ids"]),
        "paired_submission_question_count": paired_submissions,
        "model_summary": model_summary,
    }
    if include_cases:
        result["cases"] = [
            {"question_id": qid, "systems": {
                key: by_pair[(qid, key)] for key in model_keys
            }}
            for qid in manifest["question_ids"]
        ]
    return result


def build_analysis(smoke_dir: Path, dev_full_dir: Path) -> dict:
    smoke_manifest, smoke_rows, smoke_hashes = _load_run(smoke_dir)
    dev_manifest, dev_rows, dev_hashes = _load_run(dev_full_dir)
    if smoke_manifest["split"] != dev_manifest["split"]:
        raise ValueError("smoke and full run must use the same development split")
    if set(smoke_manifest["question_ids"]) - set(dev_manifest["question_ids"]):
        raise ValueError("smoke questions are not in the full run")
    if [m["key"] for m in smoke_manifest["models"]] != [m["key"] for m in dev_manifest["models"]]:
        raise ValueError("smoke and full model keys differ")
    for key in ("benchmark_hash", "corpus_hash", "seed", "max_steps", "system_prompt_sha256"):
        if smoke_manifest.get(key) != dev_manifest.get(key):
            raise ValueError(f"smoke and full {key} differ")
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_code_sha256": _sha256(Path(__file__)),
        "scope": "saved_development_run_execution_diagnostic_only",
        "hypothesis": (
            "Generated-action token logprobs may be lower on code turns with immediate "
            "tool or runtime errors than on code turns with clean observations."
        ),
        "decision_rule": (
            "Compare within each model only if at least 80% of code turns have usable "
            "logprobs and both error and clean groups are nonempty; otherwise report coverage. "
            "Any comparison is descriptive, never a correctness/confidence threshold."
        ),
        "caveats": [
            "Logprobs are for emitted action tokens, not answer correctness or evidence support.",
            "Step means pool different action lengths and positions; "
            "errors are observational, not randomized.",
            "No task answerability labels or model-assisted quality grades are used.",
            "Episode-level endpoint errors have no aligned generated-action logprob.",
            "Raw prompts, actions, observations, answers, token strings, "
            "and endpoints are omitted.",
        ],
        "smoke": {
            "provenance": _provenance(smoke_manifest, smoke_hashes),
            **_analyze(smoke_manifest, smoke_rows, include_cases=True),
        },
        "dev_full": {
            "provenance": _provenance(dev_manifest, dev_hashes),
            **_analyze(dev_manifest, dev_rows, include_cases=False),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--dev-full-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = build_analysis(args.smoke_dir, args.dev_full_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote sanitized offline diagnostic: {args.output}")


if __name__ == "__main__":
    main()
