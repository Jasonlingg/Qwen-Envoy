"""Exercise the real local MCP with a draft proposal and a separate historical import.

No browsing, model inference, saved-code execution, training or GPU provisioning.
Paper references require the three versioned papers already open in the selected cache.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/ml-experiment-qwen-v5"


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


async def run_demo(*, store_root: Path, artifact_root: Path, library_root: Path,
                   example_root: Path = EXAMPLE) -> dict:
    proposal = json.loads((example_root / "proposal.json").read_text())
    history = json.loads((example_root / "historical-plan.json").read_text())
    outcome = json.loads((example_root / "historical-result.json").read_text())
    # A fresh copy is selected explicitly. Never replace a different operator export.
    source = example_root / "exports/qwen-v5-task5-development.json"
    artifact_root.mkdir(parents=True, exist_ok=True)
    target = artifact_root / source.name
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise ValueError("Selected export already exists with different bytes")
    else:
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
    parameters = StdioServerParameters(command=sys.executable, cwd=str(ROOT), args=[
        str(ROOT / "scripts/ml_experiment_mcp.py"), "--store-root", str(store_root),
        "--artifact-root", str(artifact_root), "--library-root", str(library_root)])
    calls = []

    async def call(client, tool, arguments):
        result = await client.call_tool(tool, arguments)
        if result.is_error:
            raise ValueError(f"MCP {tool} failed: {str(result.content)[:600]}")
        value = result.structured_content
        calls.append({"tool": tool, "arguments_sha256": _sha(arguments),
                      "response_sha256": _sha(value), "is_error": False})
        return value

    async with Client(parameters) as client:
        inventory = sorted(t.name for t in (await client.list_tools()).tools)
        if len(inventory) != 6 or not {"record_experiment_result",
                                         "find_related_experiments"}.issubset(inventory):
            raise ValueError("Unexpected planning MCP tool inventory")
        await call(client, "list_experiments", {})
        draft = await call(client, "save_experiment_brief", {
            "brief": {"question": proposal["question"]}})
        draft = await call(client, "save_experiment_brief", {"brief": proposal,
            "experiment_id": draft["experiment_id"], "expected_revision": 1})
        if draft["status"] != "draft" or draft["plan_sha256"] is not None:
            raise ValueError("Unapproved future proposal must stay a draft")
        old = await call(client, "save_experiment_brief", {"brief": history})
        frozen = await call(client, "freeze_experiment_plan", {
            "experiment_id": old["experiment_id"], "expected_revision": 1})
        args = {"experiment_id": old["experiment_id"], "plan_sha256": frozen["plan_sha256"],
                "result": outcome}
        recorded = await call(client, "record_experiment_result", args)
    # Use a second server process to verify durable bindings, not in-memory state.
    async with Client(parameters) as restarted:
        restored = await call(restarted, "get_experiment", {
            "experiment_id": old["experiment_id"], "run_id": outcome["run_id"]})
        repeated = await call(restarted, "record_experiment_result", args)
        reopened = await call(restarted, "get_experiment", {
            "experiment_id": draft["experiment_id"]})
        await call(restarted, "list_experiments", {})
    selected = restored["selected_result"]
    if selected != recorded["selected_result"] or selected != repeated["selected_result"]:
        raise ValueError("Restart or retry changed the historical outcome")
    if reopened["status"] != "draft" or reopened["results"]:
        raise ValueError("Future proposal was frozen or given historical results")
    assessment = selected["assessment"]
    expected = {"original_provisional_gain": "met", "original_error_tolerance": "met",
                "independent_support_gain": "unknown", "complete_latency_nonregression": "unknown",
                "all_in_cost_nonregression": "unknown"}
    if ({r["criterion_id"]: r["status"] for r in assessment["criteria"]} != expected
            or assessment["overall"] != "inconclusive"
            or assessment["effective_timing"] != "retrospective"):
        raise ValueError("Historical unknowns or retrospective timing were not preserved")
    return {"schema_version": "ml-experiment-development-demo-v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "transport": "Python MCP stdio client invoking the locally registered server command",
        "native_host_planning_tool_inventory": "Not loaded in this session; used stdio client",
        "tools": inventory, "calls": calls, "restart_verified": True, "retry_verified": True,
        "proposal": {key: reopened[key] for key in ["experiment_id", "revision", "status",
                                                   "plan_sha256", "results"]},
        "historical": {"experiment_id": old["experiment_id"],
                       "plan_sha256": restored["plan_sha256"], "result_sha256": selected["sha256"],
                       "assessment": assessment},
        "model_requests": 0, "gpu_launches": 0, "paid_model_calls": 0,
        "independent_human_review": "pending", "user_usefulness_review": "pending",
        "limitations": ["No new Qwen outputs or intervention outcomes were collected.",
            "Source research is documented separately; this script reuses those cached papers.",
            "MCP checks export bytes, not faithful parent derivation; run the exporter to recheck.",
            "The proposal's resource ceilings are unapproved estimates, not executed limits."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store-root", "artifact-root", "library-root", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Refusing to replace an existing demonstration report")
    report = asyncio.run(run_demo(store_root=args.store_root.absolute(),
        artifact_root=args.artifact_root.absolute(), library_root=args.library_root.absolute()))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "calls": len(report["calls"]),
        "proposal_status": report["proposal"]["status"],
        "historical_assessment": report["historical"]["assessment"]["overall"]}))


if __name__ == "__main__":
    main()
