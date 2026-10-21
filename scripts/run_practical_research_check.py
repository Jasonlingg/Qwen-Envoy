"""Five project decisions through the existing native collector; no worker/GPU calls."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.native_pilot_case import run_case, write_json  # noqa: E402
from src.eval.native_pilot_config import EFFORT, MODEL  # noqa: E402

ASSISTED = """This answer is the prescribed MCP-assisted condition. Actually use
arxiv-research.search_papers to discover relevant papers, open a relevant paper,
and inspect relevant evidence with find_evidence/get_passage before answering.
Use native web tools for official model/runtime documentation and as fallback
when MCP fails or cannot find applicable material. Report MCP failures honestly;
do not invent retrieved evidence. Do not use experiment-record tools or workers.
"""
DIRECT = "This answer is the ordinary web-research condition; no MCP tools are available."
ORDER = [[f"P{i:02d}", *(('A', 'B') if i % 2 else ('B', 'A'))] for i in range(1, 6)]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "examples/practical-research-check/questions.json"
    data = json.loads(source.read_text())
    by_id = {q["id"]: q for q in data["questions"]}
    args.output.mkdir(parents=True, exist_ok=False)
    files = ["examples/practical-research-check/questions.json",
             "examples/practical-research-check/PLAN.md", "scripts/run_practical_research_check.py",
             "src/eval/native_pilot_case.py", "src/eval/native_pilot_config.py",
             "src/eval/native_pilot_client.py", "src/eval/native_pilot_events.py",
             "scripts/arxiv_research_mcp.py", "src/product/arxiv_mcp_config.py"]
    manifest = {"status": "frozen_before_answers", "model": MODEL, "effort": EFFORT,
        "arm_order": ORDER, "qwen_workers": "off", "paid_worker_api_calls": 0,
        "host_dollar_cap": None, "host_dollar_cost": None, "human_review": "not_performed",
        "hypothesis": "Source MCP helps produce useful supported project-decision briefs.",
        "decision_rule": ("Inspect all five pairs qualitatively; no automatic quality score "
                          "or training approval."),
        "input_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}}
    write_json(args.output / "manifest.json", manifest)
    outcomes = []
    for qid, *arms in ORDER:
        for arm in arms:
            condition = ASSISTED if arm == "B" else DIRECT
            print(json.dumps({"event": "starting", "question": qid, "arm": arm}), flush=True)
            result = await run_case(ROOT, args.output / f"{qid}-{arm}",
                                   data["common_instruction"] + "\n\n" + condition, by_id[qid], arm)
            items = result.get("items", [])
            mcp = [i for i in items if i["type"] == "mcpToolCall"]
            summary = {k: result.get(k) for k in ("status", "answer_present", "elapsed_seconds",
                "research_calls_observed", "research_target_overshoot", "token_usage", "error")}
            summary.update(question_id=qid, arm=arm, mcp_calls=len(mcp),
                           mcp_tool_names=[i.get("tool") for i in mcp])
            outcomes.append(summary)
            write_json(args.output / "collection.json", {"outcomes": outcomes})
            print(json.dumps({"event": "finished", **summary}), flush=True)
            if result["status"] != "completed":
                print("Stopped on incomplete native session; no retry or fallback.", flush=True)
                return


if __name__ == "__main__":
    asyncio.run(main())
