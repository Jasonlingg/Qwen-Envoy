"""Prospective, same-checkpoint prefix-cache timing diagnostic (never a quality score).

Preparation is offline. Execution runs one explicitly declared server phase through
the ordinary multi-step EnvoyBench runner; it never provisions or restarts a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.envoybench import run as runner
from benchmarks.envoybench.score import _error_diagnostics
from src.eval.artifacts import configuration_hash

SCHEMA = "envoybench-prefix-cache-ablation-v1"
PINNED_IDENTITY = {
    "model_id": "envoy",
    "revision": "27b912a863ff914ad45baa03624d8911dc1e17fb",
    "base_model_id": "Qwen/Qwen3-8B",
    "base_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "adapter_id": "jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5",
    "adapter_revision": "27b912a863ff914ad45baa03624d8911dc1e17fb",
    "adapter_sha256": "7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3",
}
DECODING = {"max_tokens": 1024, "temperature": 0.0, "top_p": 1.0}
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
PHASES = [
    {"phase": "off-1", "prefix_caching": "off", "repetition": 1},
    {"phase": "on-1", "prefix_caching": "on", "repetition": 1},
    {"phase": "on-2", "prefix_caching": "on", "repetition": 2},
    {"phase": "off-2", "prefix_caching": "off", "repetition": 2},
]
WARMUP = {
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "max_tokens": 8, "temperature": 0.0, "top_p": 1.0,
    **EXTRA_BODY,
}


def _model_settings(model: dict) -> dict:
    for name, expected in PINNED_IDENTITY.items():
        if model.get(name) != expected:
            raise ValueError(f"pinned v5 {name} differs")
    if model.get("decoding") != DECODING or model.get("extra_body") != EXTRA_BODY:
        raise ValueError("pinned decoding or chat template differs")
    if model.get("send_seed", False) is not False:
        raise ValueError("send_seed must remain false, matching the source policy")
    if not isinstance(model.get("key"), str) or not model["key"]:
        raise ValueError("model key is missing")
    return {"key": model["key"], **PINNED_IDENTITY, "decoding": DECODING,
            "extra_body": EXTRA_BODY, "send_seed": False}


def _implementation() -> dict:
    return {**runner._implementation_hashes(),
            "benchmarks/envoybench/inference_ablation.py": runner._file_sha256(Path(__file__)),
            "benchmarks/envoybench/score.py": runner._file_sha256(
                runner.ROOT / "benchmarks/envoybench/score.py"),
            "scripts/serve_vllm.sh": runner._file_sha256(runner.ROOT / "scripts/serve_vllm.sh")}


def build_plan(dataset: Path, source_run: Path, *, serving_hardware: str | None = None,
               serving_runtime: str | None = None) -> dict:
    """Freeze the first three dev IDs without reading any model outputs or endpoints."""
    source = runner._read_json(source_run / "manifest.json")
    if source.get("schema_version") != runner.RUN_SCHEMA or source.get("status") != "complete":
        raise ValueError("source run must be a complete EnvoyBench run")
    candidates = [m for m in source.get("models", [])
                  if m.get("adapter_id") == PINNED_IDENTITY["adapter_id"]]
    if len(candidates) != 1:
        raise ValueError("source must identify exactly one pinned v5 model")
    model = candidates[0]
    settings = _model_settings(model)
    hardware = serving_hardware or model.get("serving_hardware")
    runtime = serving_runtime or model.get("serving_runtime")
    if any(not isinstance(v, str) or not v or v == "unreported" for v in (hardware, runtime)):
        raise ValueError("serving hardware and exact runtime must be declared")
    benchmark, snapshot, benchmark_path, _, _ = runner.load_split(dataset, "dev")
    questions = benchmark["questions"][:3]
    if len(questions) != 3:
        raise ValueError("the frozen dev split must have at least three questions")
    plan = {
        "schema_version": SCHEMA, "status": "prepared_not_run", "requests_executed": 0,
        "scope": "development serving diagnostic; not held-out quality or model promotion",
        "source_run_id": source["run_id"],
        "source_manifest_sha256": runner._file_sha256(source_run / "manifest.json"),
        "split": "dev", "split_status": snapshot["split_status"],
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": configuration_hash(benchmark),
        "benchmark_file_sha256": runner._file_sha256(benchmark_path),
        "dataset_index_sha256": runner._file_sha256(dataset / "manifest.json"),
        "corpus_hash": benchmark["corpus_hash"],
        "question_ids": [q["id"] for q in questions],
        "selected_questions_sha256": configuration_hash({"questions": questions}),
        "selection_rule": "first three IDs in frozen dev order, before reading outputs",
        "model": settings, "serving_hardware": hardware, "serving_runtime": runtime,
        "serving_context_window_tokens": 8192, "max_steps": 15, "seed": 42, "workers": 1,
        "seed_note": "Local seed 42; source send_seed=false retained; "
                     "remote determinism not guaranteed.",
        "phases": PHASES, "minimum_median_paired_reduction": 0.15,
        "hypothesis": "Prefix caching reduces same-checkpoint multi-step inference latency.",
        "expected_signal": "Lower paired elapsed time with identical outputs and no failures.",
        "decision_rule": "A >=15% median paired request-time reduction is a timing signal "
                         "only when "
                         "all four phases have complete timing and token counts, identical "
                         "answers/evidence/actions and per-request token counts, and no errors.",
        "server_control": "Operator declared, not endpoint attested. "
                          "Fresh process for every phase; "
                          "run phases in listed order with no concurrent traffic.",
        "warmup": {"request": {"model": settings["model_id"], **WARMUP},
                   "protocol": "On the fresh process, run this same disjoint warmup once; exclude "
                               "its time, then reset the prefix cache before measured requests. "
                               "With VLLM_SERVER_DEV_MODE=1, POST /reset_prefix_cache and require "
                               "the JSON response success:true; a restart after warmup also clears "
                               "kernel warmup and is not equivalent.",
                   "reset_command": "curl --fail --silent --show-error -X POST "
                                    "http://127.0.0.1:8000/reset_prefix_cache",
                   "required_reset_response": {"success": True}},
        "implementation_sha256": _implementation(),
    }
    plan["plan_hash"] = configuration_hash(plan)
    return plan


def prepare(dataset: Path, source_run: Path, output: Path, **kwargs) -> dict:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    plan = build_plan(dataset, source_run, **kwargs)
    output.mkdir(parents=True)
    runner._write_json(output / "plan.json", plan)
    runner._write_json(output / "models.json", {
        "schema_version": runner.MODEL_SCHEMA,
        "models": [{**plan["model"], "endpoint_env": "ENVOY_QWEN_ENDPOINT",
                    **{key: plan[key] for key in ("serving_hardware", "serving_runtime",
                                                 "serving_context_window_tokens")}}],
    })
    return plan


def _read_plan(prepared: Path) -> dict:
    plan = runner._read_json(prepared / "plan.json")
    unhashed = {k: v for k, v in plan.items() if k != "plan_hash"}
    if (plan.get("schema_version") != SCHEMA
            or configuration_hash(unhashed) != plan.get("plan_hash")):
        raise ValueError("prepared plan hash differs")
    return plan


def execute(prepared: Path, dataset: Path, source_run: Path, models_path: Path,
            output: Path, *, phase: str, server_prefix_caching: str,
            server_process_id: str, fresh_server_process: bool,
            warmup_cache_reset_complete: bool) -> dict:
    plan = _read_plan(prepared)
    rebuilt = build_plan(dataset, source_run, serving_hardware=plan["serving_hardware"],
                        serving_runtime=plan["serving_runtime"])
    if rebuilt != plan:
        raise ValueError("prepared source, dataset, implementation or settings changed")
    selected = next((p for p in PHASES if p["phase"] == phase), None)
    if selected is None or selected["prefix_caching"] != server_prefix_caching:
        raise ValueError("declared server prefix caching differs from phase")
    if not server_process_id.strip() or not fresh_server_process or not warmup_cache_reset_complete:
        raise ValueError(
            "fresh process and completed disjoint warmup/cache reset declarations required")
    raw = runner._read_json(models_path)
    if len(raw.get("models", [])) != 1:
        raise ValueError("execute requires exactly one v5 model in its config")
    models = runner.load_models(models_path)
    model = models[0]["safe"]
    if _model_settings(model) != plan["model"]:
        raise ValueError("execution model differs from plan")
    for field in ("serving_hardware", "serving_runtime", "serving_context_window_tokens"):
        if model.get(field) != plan[field]:
            raise ValueError(f"execution {field} differs from plan")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    sidecar = {
        "schema_version": SCHEMA, "plan_hash": plan["plan_hash"], **selected,
        "server_process_id": server_process_id, "fresh_server_process": True,
        "warmup_cache_reset_complete": True,
        "server_control_source": "operator_declared_not_endpoint_attested",
        "source_run_id": plan["source_run_id"],
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "phase_started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        result = runner.run(dataset, "dev", models_path, output,
                            model_keys=[plan["model"]["key"]], question_ids=plan["question_ids"],
                            seed=plan["seed"], max_steps=plan["max_steps"])
    finally:
        if output.is_dir():
            sidecar["artifact_sha256"] = {
                name: runner._file_sha256(output / name)
                for name in ("manifest.json", "results.json") if (output / name).is_file()
            }
            runner._write_json(output / "ablation.json", sidecar)
    return result


def _positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _request_timing(row: dict) -> tuple[float | None, int, int]:
    usage = row.get("policy_metadata", {}).get("token_usage", {})
    requests = usage.get("requests", [])
    expected = usage.get("request_count")
    valid = [r.get("duration_seconds") for r in requests if _positive(r.get("duration_seconds"))]
    complete = type(expected) is int and expected > 0 and len(requests) == expected == len(valid)
    return (sum(valid) if complete else None, len(valid), expected or 0)


def _behavior(row: dict) -> dict:
    return {"status": row.get("status"), "predicted_answer": row.get("predicted_answer"),
            "predicted_citations": row.get("predicted_citations"),
            "predicted_evidence": row.get("predicted_evidence"),
            "actions": [step["action"] for step in row["trajectory"]]}


def _request_usage(row: dict) -> dict:
    usage = row.get("policy_metadata", {}).get("token_usage", {})
    requests = usage.get("requests", [])
    result = {"request_count": usage.get("request_count", 0)}
    for field in ("prompt_tokens", "completion_tokens", "cached_prompt_tokens"):
        values = [r.get(field) for r in requests]
        observed = [v for v in values if type(v) is int and v >= 0]
        complete = (bool(values) and len(observed) == len(values) == result["request_count"]
                    and not any(r.get("failed") for r in requests))
        result[field] = sum(observed) if complete else None
        result[f"observed_{field}"] = sum(observed) if observed else None
        result[f"{field}_observed_requests"] = len(observed)
        result[f"{field}_complete"] = complete
    result["per_request_workload"] = [
        [r.get("prompt_tokens"), r.get("completion_tokens")] for r in requests
    ] if result["prompt_tokens_complete"] and result["completion_tokens_complete"] else None
    return result


def compare(prepared: Path, phase_dirs: list[Path]) -> dict:
    """Require a complete four-phase matrix; withhold a speed claim on any confound."""
    plan = _read_plan(prepared)
    if len(phase_dirs) != 4:
        raise ValueError("compare requires exactly four phase directories")
    matrix, process_ids, run_ids, starts = {}, set(), set(), {}
    sandbox = None
    common_protocol = None
    for directory in phase_dirs:
        sidecar = runner._read_json(directory / "ablation.json")
        phase = sidecar.get("phase")
        expected_phase = next((p for p in PHASES if p["phase"] == phase), None)
        if (expected_phase is None or phase in matrix
                or sidecar.get("plan_hash") != plan["plan_hash"]
                or any(sidecar.get(k) != v for k, v in expected_phase.items())
                or sidecar.get("source_run_id") != plan["source_run_id"]
                or sidecar.get("source_manifest_sha256") != plan["source_manifest_sha256"]):
            raise ValueError("phase identity or plan hash differs")
        process_id = sidecar.get("server_process_id")
        if (not process_id or process_id in process_ids or
                sidecar.get("fresh_server_process") is not True or
                sidecar.get("warmup_cache_reset_complete") is not True):
            raise ValueError(
                "each phase must declare a unique fresh process and warmup/cache reset")
        process_ids.add(process_id)
        try:
            starts[phase] = datetime.fromisoformat(sidecar["phase_started_at_utc"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("phase start timestamp is missing or invalid") from exc
        for name in ("manifest.json", "results.json"):
            if (runner._file_sha256(directory / name)
                    != sidecar.get("artifact_sha256", {}).get(name)):
                raise ValueError(f"phase {name} artifact hash differs")
        manifest = runner._read_json(directory / "manifest.json")
        if manifest.get("status") != "complete" or manifest.get("run_id") in run_ids:
            raise ValueError("phase run must be complete and unique")
        run_ids.add(manifest.get("run_id"))
        for field in ("benchmark_id", "benchmark_hash", "corpus_hash", "split", "question_ids",
                      "max_steps", "seed", "benchmark_file_sha256", "dataset_index_sha256"):
            if manifest.get(field) != plan[field]:
                raise ValueError(f"phase {field} differs from plan")
        if manifest.get("implementation_sha256") != {
            k: v for k, v in plan["implementation_sha256"].items()
            if k not in {"benchmarks/envoybench/inference_ablation.py", "scripts/serve_vllm.sh",
                         "benchmarks/envoybench/score.py"}
        }:
            raise ValueError("phase implementation differs from plan")
        if len(manifest.get("models", [])) != 1:
            raise ValueError("phase must contain one model")
        model = manifest["models"][0]
        if _model_settings(model) != plan["model"] or any(model.get(f) != plan[f] for f in (
                "serving_hardware", "serving_runtime", "serving_context_window_tokens")):
            raise ValueError("phase checkpoint, hardware, runtime or context differs")
        if sandbox is None:
            sandbox = manifest.get("sandbox_image")
        elif manifest.get("sandbox_image") != sandbox:
            raise ValueError("phase sandbox image differs")
        protocol = {key: manifest.get(key) for key in (
            "comparison_id", "system_prompt_sha256", "tool_preamble_sha256", "tool_search_version",
            "require_evidence", "evidence_verifier", "verifier_feedback_budget",
            "escalate_after_verifier_failure", "docker_sandbox",
        )}
        if common_protocol is None:
            common_protocol = protocol
        elif protocol != common_protocol:
            raise ValueError("phase environment protocol differs")
        rows = json.loads((directory / "results.json").read_text())
        if (not isinstance(rows, list) or len(rows) != len(plan["question_ids"]) or
                sorted(r.get("question_id", "") for r in rows) != sorted(plan["question_ids"])):
            raise ValueError("phase question matrix is incomplete or duplicated")
        for row in rows:
            if (row.get("run_id") != manifest["run_id"] or
                    row.get("model_key") != plan["model"]["key"] or
                    row.get("comparison_id") != manifest["comparison_id"]):
                raise ValueError("row identity differs from phase manifest")
        matrix[phase] = {row["question_id"]: row for row in rows}
    if sorted(starts, key=starts.get) != [phase["phase"] for phase in PHASES]:
        raise ValueError("phase execution order differs from OFF, ON, ON, OFF")
    per_question, reductions, request_reductions = [], [], []
    failures, changed, missing_episode, missing_requests, missing_usage, changed_workload = (
        [], [], [], [], [], [],
    )
    coverage = []
    for qid in plan["question_ids"]:
        rows = {phase: matrix[phase][qid] for phase in matrix}
        if any(_behavior(row) != _behavior(rows["off-1"]) for row in rows.values()):
            changed.append(qid)
        workloads = [_request_usage(row)["per_request_workload"] for row in rows.values()]
        if all(w is not None for w in workloads) and any(w != workloads[0] for w in workloads):
            changed_workload.append(qid)
        pairs = []
        for phase, row in rows.items():
            request_seconds, observed, expected = _request_timing(row)
            token_usage = _request_usage(row)
            coverage.append({"question_id": qid, "phase": phase,
                             "episode_timing_observed": _positive(row.get("duration_seconds")),
                             "request_timing_observed": observed, "request_count": expected,
                             "request_timing_complete": request_seconds is not None,
                             "token_usage": token_usage})
            usage = row.get("policy_metadata", {}).get("token_usage", {})
            errors = _error_diagnostics(row)
            if (errors["execution_error_episode"] or row.get("error")
                    or usage.get("failed_request_count", 0)):
                failures.append({"question_id": qid, "phase": phase,
                                 "error_categories": errors["error_categories"]})
            if not _positive(row.get("duration_seconds")):
                missing_episode.append({"question_id": qid, "phase": phase})
            if request_seconds is None:
                missing_requests.append({"question_id": qid, "phase": phase})
            if token_usage["per_request_workload"] is None:
                missing_usage.append({"question_id": qid, "phase": phase})
        for repetition in (1, 2):
            off, on = rows[f"off-{repetition}"], rows[f"on-{repetition}"]
            a, b = off.get("duration_seconds"), on.get("duration_seconds")
            req_a, req_b = _request_timing(off)[0], _request_timing(on)[0]
            delta = (1 - b / a) if _positive(a) and _positive(b) else None
            request_delta = (1 - req_b / req_a) if req_a and req_b else None
            if delta is not None:
                reductions.append(delta)
            if request_delta is not None:
                request_reductions.append(request_delta)
            pairs.append({"repetition": repetition, "off_elapsed_seconds": a,
                          "on_elapsed_seconds": b, "elapsed_reduction_fraction": delta,
                          "off_request_seconds": req_a, "on_request_seconds": req_b,
                          "request_reduction_fraction": request_delta})
        per_question.append({"question_id": qid, "pairs": pairs,
                             "median_paired_elapsed_reduction_fraction": statistics.median(
                                 [p["elapsed_reduction_fraction"] for p in pairs]
                             ) if all(p["elapsed_reduction_fraction"] is not None for p in pairs)
                             else None})
    median = statistics.median(reductions) if reductions else None
    request_median = statistics.median(request_reductions) if request_reductions else None
    eligible = not (failures or changed or missing_episode or missing_requests
                    or missing_usage or changed_workload)
    met = eligible and request_median is not None and request_median >= 0.15
    return {"schema_version": SCHEMA, "plan_hash": plan["plan_hash"],
            "scope": plan["scope"],
            "server_control_source": "operator_declared_not_endpoint_attested",
            "per_question": per_question, "timing_coverage": coverage,
            "failures": failures, "changed_output_question_ids": changed,
            "missing_episode_timing": missing_episode, "missing_request_timing": missing_requests,
            "missing_token_usage": missing_usage,
            "changed_token_workload_question_ids": changed_workload,
            "workload_note": "Exact cleaned actions and per-request prompt/completion token counts "
                             "are compared; raw assistant responses are not saved by this runner.",
            "median_paired_elapsed_reduction_fraction": median,
            "median_paired_request_reduction_fraction": request_median,
            "timing_gain_claim_eligible": eligible, "minimum_reduction": 0.15,
            "threshold_met": met, "quality_promotion": False,
            "decision": "diagnostic_timing_signal" if met else
                        "inconclusive_confounded" if not eligible else "threshold_not_met"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true", help="send model requests; may incur cost")
    modes.add_argument("--compare", action="store_true")
    parser.add_argument("--dataset", type=Path, default=runner.DEFAULT_DATASET)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serving-hardware")
    parser.add_argument("--serving-runtime")
    parser.add_argument("--phase", choices=[p["phase"] for p in PHASES])
    parser.add_argument("--phase-dir", type=Path, action="append")
    parser.add_argument("--server-prefix-caching", choices=["on", "off"])
    parser.add_argument("--server-process-id")
    parser.add_argument("--fresh-server-process", action="store_true")
    parser.add_argument("--warmup-cache-reset-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            if not args.source_run:
                raise ValueError("--prepare requires --source-run")
            result = prepare(args.dataset, args.source_run, args.output,
                             serving_hardware=args.serving_hardware,
                             serving_runtime=args.serving_runtime)
        elif args.execute:
            if not all((args.prepared, args.source_run, args.models, args.phase,
                        args.server_prefix_caching, args.server_process_id)):
                raise ValueError("--execute requires --prepared, --source-run, --models, --phase, "
                                 "--server-prefix-caching and --server-process-id")
            result = execute(args.prepared, args.dataset, args.source_run, args.models, args.output,
                             phase=args.phase, server_prefix_caching=args.server_prefix_caching,
                             server_process_id=args.server_process_id,
                             fresh_server_process=args.fresh_server_process,
                             warmup_cache_reset_complete=args.warmup_cache_reset_complete)
        else:
            if not args.prepared or not args.phase_dir:
                raise ValueError("--compare requires --prepared and four --phase-dir values")
            if args.output.exists():
                raise ValueError(f"output already exists: {args.output}")
            result = compare(args.prepared, args.phase_dir)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            runner._write_json(args.output, result)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Prefix cache ablation: {exc}\n")
    print(f"Prefix cache ablation: {args.output} ({result.get('decision', result.get('status'))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
