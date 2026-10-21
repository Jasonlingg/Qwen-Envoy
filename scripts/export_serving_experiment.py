"""Build a small retrospective experiment from checked, committed observations.

This example selects an existing research note and serving run. It never fetches
sources, invokes a model, executes generated code, or changes the original data.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product.ml_experiment_measurements import Measurement  # noqa: E402

PROFILE = "examples/two-reader-profile/automatic-live-1"
RESEARCH = "examples/experiment-mode-walkthrough"
POPULATION = "Two measured rounds over the same pinned historical input pair"
QUESTION = "Can two Qwen v5 reader requests share one A5000, and should we claim a latency win?"
DECISION = ("Keep one shared model with up to two requests; concurrent admission worked. "
            "Do not claim a speedup or choose another latency optimization from these two "
            "short-output rounds. The next useful decision needs representative reader "
            "evidence and answer checks, not more serving changes or automatic training.")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def build_example(source_root=ROOT):
    """Recheck manifest hashes before deriving any observation or measurement."""
    source_root = Path(source_root)
    parents = {}

    def load(relative, expected=None):
        raw = (source_root / relative).read_bytes()
        if len(raw) > 2_000_000 or expected is not None and sha(raw) != expected:
            raise ValueError("Saved source bytes differ: " + relative)
        parents[relative] = sha(raw)
        return json.loads(raw)

    profile_manifest = load(PROFILE + "/validation.json")
    research_manifest = load(RESEARCH + "/validation.json")

    def profile(name):
        return load(PROFILE + "/" + name, profile_manifest["public_artifacts_sha256"][name])

    research_path = RESEARCH + "/research-note.json"
    research = load(research_path, research_manifest["artifact_sha256"][research_path])
    result, outcome = profile("result.json"), profile("outcome.json")
    plan, identity = profile("plan.json"), profile("worker-identity.json")
    if (outcome["assessment"] != result["assessment"]
            or outcome["model_jobs_completed"] != 10
            or outcome["telemetry_jobs_completed"] != 9
            or not outcome["startup_automation_passed"]
            or not outcome["shutdown_verified"]
            or not outcome["full_endpoint_configuration_restored"]
            or result["assessment"]["comparable"]
            or result["assessment"]["reasons"] != [
                "difference does not exceed the 0.4-second polling uncertainty margin"]):
        raise ValueError("Saved profile does not describe the recorded inconclusive run")
    if [(b["mode"], b["repeat"]) for b in result["batches"]] != [
            ("serial", 1), ("concurrent", 1), ("concurrent", 2), ("serial", 2)]:
        raise ValueError("Counterbalanced batch coverage differs")
    signatures = {}
    for row in [*result["warmups"], *[r for b in result["batches"] for r in b["requests"]]]:
        response = row["response"]
        signature = encoded({k: response[k] for k in ("choices", "usage")})
        if signatures.setdefault(row["sample_id"], signature) != signature:
            raise ValueError("Saved output, usage or finish reason changed")
    if len(result["warmups"]) != 2 or set(signatures) != set(plan["payload_hashes"]):
        raise ValueError("Saved input population differs")
    for batch in result["batches"]:
        telemetry = batch["telemetry"]
        if len(batch["requests"]) != 2 or telemetry["errors"]:
            raise ValueError("Incomplete batch or failed telemetry")
        metrics = [r["metrics"] for r in [telemetry["before"],
                   *telemetry["samples"], telemetry["after"]]]
        peak = max(m["vllm:num_requests_running"] for m in metrics)
        if peak != batch["engine_peak_running_sampled"]:
            raise ValueError("Saved engine overlap summary differs")
        phases = ("request_queue_time_seconds", "request_prefill_time_seconds",
                  "request_decode_time_seconds", "time_to_first_token_seconds")
        if any(telemetry["after"]["metrics"]["vllm:" + name + "_count"]
               - telemetry["before"]["metrics"]["vllm:" + name + "_count"] != 2
               for name in phases):
            raise ValueError("Phase telemetry does not cover the two requests")

    def known(value):
        return {"value": value}
    evaluation = {"population": POPULATION, "purpose": "historical", "expected_cases": 2,
                  "sample_ids": ["pair_round_1", "pair_round_2"],
                  "dataset_revision": known(sha(encoded(plan["payload_hashes"])))}
    shared = {"model_revision": known(identity["base_revision"] + "/" + identity["adapter_sha256"]),
              "workload": known(evaluation["dataset_revision"]["value"]),
              "hardware": known(identity["gpu"]),
              "software_revision": known("vLLM " + identity["vllm_version"]
                                         + "; torch " + identity["torch_version"])}
    metrics = [
        {"metric_id": "completed_calls", "unit": "count", "aggregation": "total",
         "population": POPULATION, "direction": "higher", "measurement_kind": "other"},
        {"metric_id": "engine_overlap", "unit": "count", "aggregation": "minimum round peak",
         "population": POPULATION, "direction": "higher", "measurement_kind": "other"},
        {"metric_id": "pair_seconds", "unit": "seconds", "aggregation": "mean",
         "population": POPULATION, "direction": "lower", "measurement_kind": "latency",
         "timing_scope": "warm"}]
    measurements = {}
    for label, mode in (("baseline", "serial"), ("candidate", "concurrent")):
        batches = [b for b in result["batches"] if b["mode"] == mode]
        measurements[label] = {}
        for metric in metrics:
            name = metric["metric_id"]
            value = (sum(len(b["requests"]) for b in batches) if name == "completed_calls"
                     else min(b["engine_peak_running_sampled"] for b in batches)
                     if name == "engine_overlap" else None)
            fields = {k: metric[k] for k in ("metric_id", "unit", "aggregation", "population")}
            fields.update(value=float(value) if value is not None else None, sample_count=2,
                          coverage="complete" if value is not None else "unknown")
            if value is None:
                fields.update(unknown_reason="No admissible latency comparison: one round is "
                    "within the predeclared 0.4-second polling margin. Raw timings retained.",
                    timing_scope="warm")
            measurements[label][name] = Measurement(**fields).model_dump(
                mode="json", exclude={"reference_id"})
    paper = next(r for r in research["references"] if r["kind"] == "paper_span")
    export = {"schema_version": "recorded-serving-experiment-export-v1",
        "source_sha256": parents, "measurement_derivation":
        "Counts and minimum sampled round peaks from original batches; latency stays unknown "
        "because the original comparability check failed. Raw pair timings are not erased.",
        "saved_paper_research": {"claim": research["findings"][0]["claim"],
            "arxiv_id": paper["arxiv_id"], "start": paper["start"], "end": paper["end"],
            "quote": paper["quote"], "provenance": "Imported earlier research artifact; "
            "not a fresh paper-span verification or independent semantic review."},
        "raw_rounds": result["assessment"]["rounds"], "per_sample": outcome["per_sample"],
        "live_identity": identity, "config": plan["config"], "measurements": measurements,
        "operational": {k: outcome[k] for k in ("startup_automation_passed",
            "model_jobs_completed", "telemetry_jobs_completed", "new_failed_jobs",
            "shutdown_verified", "full_endpoint_configuration_restored")},
        "original_assessment": result["assessment"], "decision": DECISION}
    export_hash = sha(encoded(export))

    def reference(name, pointer):
        return {"kind": "local_artifact", "reference_id": name,
                "relative_path": "serving-observations.json", "sha256": export_hash,
                "json_pointer": pointer}

    research_refs = [reference("prior_paper_research", "/saved_paper_research"),
        {"kind": "external_reference", "reference_id": "original_paper",
         "url": "https://arxiv.org/abs/" + paper["arxiv_id"],
         "description": "Original paper from earlier live discovery; not fetched by this replay."}]
    brief = {"question": QUESTION, "goal": "latency", "current_setup": {
        "task": known("Retrospective review of a fixed-input serving diagnostic; not full agents"),
        "model": known(identity["base_repo"]),
        "checkpoint": known(identity["adapter_repo"] + "/" + identity["adapter_subfolder"]),
        "revision": shared["model_revision"], "hardware": shared["hardware"],
        "workload": known(POPULATION), "software": {"serving": shared["software_revision"]}},
        "proposed_change": "Compare previously recorded concurrent dispatch with serial dispatch.",
        "hypothesis": "Concurrent requests can share one engine; latency benefit is uncertain.",
        "expected_signal": "Complete matched outputs and two-request overlap; admit latency only "
            "when the original uncertainty/comparability check allows it.",
        "applicability": "Two short fixed inputs only; reconstruction after results were seen.",
        "baseline": {"description": "Recorded serial pair", "identity": shared},
        "candidate": {"description": "Recorded concurrent pair", "identity": shared},
        "evaluation": evaluation,
        "conditions": [{"name": "dispatch_mode", "role": "proposed_change",
            "baseline": known("serial"), "candidate": known("concurrent")},
            {"name": "runtime_config", "role": "held_constant", "baseline": known(
                json.dumps(plan["config"], sort_keys=True)), "candidate": known(
                json.dumps(plan["config"], sort_keys=True))}],
        "metrics": metrics, "criteria": [
            {"criterion_id": "all_timed_calls_completed", "metric_id": "completed_calls",
             "operation": "absolute_gte", "threshold": 4.0},
            {"criterion_id": "same_engine_overlap", "metric_id": "engine_overlap",
             "operation": "absolute_gte", "threshold": 2.0},
            {"criterion_id": "resolved_pair_time_nonregression", "metric_id": "pair_seconds",
             "operation": "delta_lte", "threshold": 0.0}],
        "budget": {"max_model_requests": 0, "max_input_tokens": 0, "max_output_tokens": 0,
            "max_gpu_seconds": 0.0, "max_elapsed_seconds": 120.0, "max_cost_usd": 0.0,
            "stop_conditions": ["Source hash mismatch or MCP error; never dispatch inference."]},
        "procedure": ["Verify original saved-source checksums and derive selected observations.",
            "Import prior research, freeze this retrospective analysis, and record measurements.",
            "Restart MCP and recover the same result, evidence and unresolved performance gate."],
        "references": [reference("observed_runtime", "/live_identity"),
                       reference("original_comparability", "/original_assessment")],
        "unknowns": ["Representative reader evidence quality remains unmeasured.",
                     "Reliable latency benefit and all-in invoice cost remain unknown."],
        "limitations": ["Reconstruction, not preregistration or a new model/optimization run.",
                        "No human usefulness review or fine-tuning improvement demonstrated."],
        "candidate_outcomes_already_seen": "yes"}
    note = {"note_id": "prior_research_context", "question": QUESTION,
        "findings": [{"claim": export["saved_paper_research"]["claim"],
                      "reference_ids": ["prior_paper_research", "original_paper"]}],
        "references": research_refs, "uncertainties": [
            "Paper mechanisms do not establish local speed or useful evidence extraction."],
        "origin": "host_research", "data_kind": "recorded"}
    recorded = {"run_id": "recorded_automatic_profile", "timing": "retrospective",
                "data_kind": "recorded", "references": [], "interpretation": DECISION,
                "limitations": outcome["limitations"] + [
                    "This MCP record was reconstructed after observing the original run."]}
    for label, mode in (("baseline", "serial"), ("candidate", "concurrent")):
        actual = []
        for name, measurement in measurements[label].items():
            ref = reference(label + "_" + name, "/measurements/" + label + "/" + name)
            recorded["references"].append(ref)
            actual.append({**measurement, "reference_id": ref["reference_id"]})
        recorded[label] = {"identity": shared, "evaluation": evaluation,
            "conditions": {"dispatch_mode": known(mode),
                "runtime_config": known(json.dumps(plan["config"], sort_keys=True))},
            "measurements": actual}
    return {"serving-observations.json": export, "brief.json": brief,
            "research-note.json": note, "result-input.json": recorded}
