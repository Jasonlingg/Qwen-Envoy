"""Replay a complete research-to-result MCP conversation with recorded evidence.

No network, model, Docker, tokenizer or GPU calls. Use a new output directory.
The project-specific question belongs to this example, not the general MCP API.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_serving_experiment import (  # noqa: E402
    DECISION,
    QUESTION,
    build_example,
    encoded,
    sha,
)
from src.product.ml_experiment_models import ExperimentBrief  # noqa: E402
from src.product.ml_experiment_results import ExperimentResult  # noqa: E402

TOOLS = {"save_experiment_brief", "get_experiment", "list_experiments",
         "find_related_experiments",
         "freeze_experiment_plan", "record_experiment_result", "record_experiment_research",
         "resume_experiment", "search_papers", "get_citations", "open_paper",
         "find_evidence", "get_passage", "create_decision_thread",
         "find_decision_threads", "get_decision_thread", "link_decision_experiment",
         "record_decision"}


async def run_demo(output: Path, *, source_root=ROOT):
    bundle = build_example(source_root)
    brief, result = bundle["brief.json"], bundle["result-input.json"]
    if ExperimentBrief.model_validate(brief).missing_fields():
        raise ValueError("Reconstructed plan is incomplete")
    ExperimentResult.model_validate(result)
    output = Path(output).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    exports = output / "exports"
    exports.mkdir()
    responses = output / "responses"
    responses.mkdir()
    for name, value in bundle.items():
        (exports / name).write_bytes(encoded(value))
    parameters = StdioServerParameters(command=sys.executable, cwd=str(ROOT), args=[
        str(ROOT / "scripts/experiment_mode_mcp.py"), "--store-root", str(output / "records"),
        "--artifact-root", str(exports), "--library-root", str(output / "library")],
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    calls = []

    async def call(client, session, tool, arguments, *, rejected=False):
        returned = await client.call_tool(tool, arguments)
        value = returned.model_dump(mode="json", by_alias=True)
        name = f"{len(calls) + 1:02d}-{tool}.json"
        (responses / name).write_bytes(encoded(value))
        calls.append({"session": session, "tool": tool,
            "arguments_sha256": sha(encoded(arguments)), "response_path": "responses/" + name,
            "response_sha256": sha(encoded(value)), "is_error": bool(returned.is_error),
            "expected_rejection": rejected})
        if bool(returned.is_error) != rejected:
            raise ValueError("Unexpected MCP outcome for " + tool)
        return returned.structured_content

    async with asyncio.timeout(120):
        async with Client(parameters) as client:
            inventory = {t.name for t in (await client.list_tools()).tools}
            if inventory != TOOLS:
                raise ValueError("Unexpected combined MCP inventory")
            await call(client, 1, "list_experiments", {})
            saved = await call(client, 1, "save_experiment_brief", {
                "brief": {"question": QUESTION}})
            key = saved["experiment_id"]
            research = await call(client, 1, "record_experiment_research", {
                "experiment_id": key, "expected_revision": 1,
                "note": bundle["research-note.json"]})
            prior_note = research["selected_research_note"]
            await call(client, 1, "save_experiment_brief", {"experiment_id": key,
                "expected_revision": 1, "brief": brief})
            frozen = await call(client, 1, "freeze_experiment_plan", {
                "experiment_id": key, "expected_revision": 2})
            result_args = {"experiment_id": key, "plan_sha256": frozen["plan_sha256"],
                           "result": result}
            recorded = await call(client, 1, "record_experiment_result", result_args)
            reference = result["references"][0]
            decision_refs = [{**reference, "reference_id": name, "json_pointer": pointer}
                for name, pointer in (("completed_serving_run", "/operational"),
                    ("observed_overlap", "/measurements/candidate/engine_overlap"),
                    ("original_uncertainty", "/original_assessment"))]
            decision_note = {"note_id": "measurement_decision", "question": QUESTION,
                "findings": [{"claim": DECISION,
                              "reference_ids": [r["reference_id"] for r in decision_refs]}],
                "references": decision_refs,
                "uncertainties": brief["unknowns"], "origin": "host_research",
                "data_kind": "recorded"}
            decision_args = {"experiment_id": key, "expected_revision": 2,
                             "note": decision_note}
            decision = await call(client, 1, "record_experiment_research", decision_args)
            thread = await call(client, 1, "create_decision_thread", {
                "question": "Should we claim a latency win from concurrent Qwen readers?",
                "experiment_id": key})
            thread_id = thread["thread_id"]
            stated_choice = {"decision_id": "defer_latency_claim", "outcome": "defer",
                "rationale": "Concurrent admission worked, but the recorded pair times "
                             "do not establish a latency win.",
                "revisit_when": "Run representative matched reader tasks with complete "
                                "latency and answer-quality measurements.",
                "author_kind": "agent_proposed",
                "basis": [{"experiment_id": key, "run_id": result["run_id"],
                           "research_note_id": "measurement_decision"}]}
            thread = await call(client, 1, "record_decision", {"thread_id": thread_id,
                "expected_revision": 1, "decision": stated_choice})

        # New stdio process and client; no memory from the first server survives.
        async with Client(parameters) as restarted:
            related = await call(restarted, 2, "find_related_experiments", {
                "question": "What did we learn about A5000 reader request latency?"})
            related_decisions = await call(restarted, 2, "find_decision_threads", {
                "question": "Qwen reader latency claim"})
            resumed_decision = await call(restarted, 2, "get_decision_thread", {
                "thread_id": thread_id})
            resumed = await call(restarted, 2, "resume_experiment", {"experiment_id": key})
            full = await call(restarted, 2, "get_experiment", {"experiment_id": key})
            repeated = await call(restarted, 2, "record_experiment_result", result_args)
            repeated_note = await call(restarted, 2, "record_experiment_research", decision_args)
            old_research = await call(restarted, 2, "get_experiment", {
                "experiment_id": key, "research_note_id": "prior_research_context"})
            changed = json.loads(json.dumps(result_args))
            changed["result"]["interpretation"] = "Changed result must not overwrite history."
            await call(restarted, 2, "record_experiment_result", changed, rejected=True)
            await call(restarted, 2, "save_experiment_brief", {
                "experiment_id": key, "expected_revision": 2, "brief": brief}, rejected=True)
            after = await call(restarted, 2, "get_experiment", {"experiment_id": key})
    selected = full["selected_result"]
    if (related["matches"][0]["experiment_id"] != key
            or related["matches"][0]["latest_assessment"] != "inconclusive"
            or related_decisions["matches"][0]["thread_id"] != thread_id
            or resumed_decision != thread
            or resumed_decision["latest_decision"]["basis_receipts"][0]["result_sha256"]
            != recorded["selected_result"]["sha256"]
            or selected != recorded["selected_result"] or selected != repeated["selected_result"]
            or full != after or old_research["selected_research_note"] != prior_note
            or repeated_note["selected_research_note"] != decision["selected_research_note"]):
        raise ValueError("Restart, retry or rejected changes altered immutable content")
    assessment = selected["assessment"]
    expected = {"all_timed_calls_completed": "met", "same_engine_overlap": "met",
                "resolved_pair_time_nonregression": "unknown"}
    if ({r["criterion_id"]: r["status"] for r in assessment["criteria"]} != expected
            or assessment["overall"] != "inconclusive"
            or assessment["effective_timing"] != "retrospective"
            or set(assessment["measurement_provenance"].values()) != {
                "artifact_measurement_verified"}
            or resumed["latest_result_run_id"] != result["run_id"]
            or resumed["latest_research_note_id"] != "measurement_decision"
            or DECISION not in resumed["context"]):
        raise ValueError("Resume lost evidence, retrospective timing or performance uncertainty")
    report = {"schema_version": "completed-experiment-mode-demo-v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": key, "plan_sha256": full["plan_sha256"],
        "decision_thread_id": thread_id,
        "result_sha256": selected["sha256"], "tools": sorted(inventory), "calls": calls,
        "restart_verified": True, "idempotent_retries_verified": True,
        "cross_session_related_recall_verified": True,
        "conflicting_result_and_frozen_edit_rejected": True,
        "research_note_count": resumed["research_note_count"],
        "result_count": resumed["result_count"], "assessment": assessment,
        "source_sha256": bundle["serving-observations.json"]["source_sha256"],
        "model_calls": 0, "gpu_activations": 0, "network_source_calls": 0,
        "decision": DECISION, "human_usefulness_review": "pending",
        "limitations": ["Recorded-data replay; earlier live source and serving runs are linked.",
            "Imported paper research is artifact-verified; no fresh source or semantic review.",
            "Real MCP stdio client, not a native host adoption or productivity test.",
            "Qwen research usefulness and model improvement remain unestablished."]}
    (output / "report.json").write_bytes(encoded(report))
    (output / "complete-record.json").write_bytes(encoded(full))
    (output / "decision-thread.json").write_bytes(encoded(resumed_decision))
    (output / "resume.json").write_bytes(encoded(resumed))
    (output / "EXPERIMENT.md").write_text(full["markdown"].rstrip() + "\n")
    (output / "RESUME.md").write_text(resumed["context"] + "\n\n" + resumed["warning"] + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for records, exports and the restart report")
    args = parser.parse_args()
    report = asyncio.run(run_demo(args.output))
    print(json.dumps({"output": str(args.output), "experiment_id": report["experiment_id"],
        "assessment": report["assessment"]["overall"], "mcp_calls": len(report["calls"]),
        "model_calls": 0, "gpu_activations": 0}))


if __name__ == "__main__":
    main()
