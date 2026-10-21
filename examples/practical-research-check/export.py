"""Export the ten saved answers and ordinary receipts; never grade answer meaning."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    (args.destination / "answers").mkdir()
    rows, preferences = [], []
    for index in range(1, 6):
        qid = f"P{index:02d}"
        preferences.append({"question_id": qid, "would_use": None,
                            "why": None, "changed_next_decision": None})
        for arm in ("A", "B"):
            folder = args.source / f"{qid}-{arm}"
            result_file = folder / "result.json"
            result = json.loads(result_file.read_text()) if result_file.exists() else {}
            answer = (folder / "answer.md").read_bytes() if (folder / "answer.md").exists() else b""
            (args.destination / "answers" / f"{qid}-{arm}.md").write_bytes(answer)
            items = result.get("items", [])
            research = [i for i in items if i["type"] in {"webSearch", "mcpToolCall"}]
            mcp = [i for i in research if i["type"] == "mcpToolCall"]
            data_statuses = [i.get("result", {}).get("structuredContent", {}).get("status")
                             for i in mcp]
            rows.append({"question_id": qid, "arm": arm,
                "status": result.get("status", "unattempted"),
                "elapsed_seconds": result.get("elapsed_seconds"), "answer_present": bool(answer),
                "answer_sha256": hashlib.sha256(answer).hexdigest(),
                "token_usage": result.get("token_usage"), "actual_cost_usd": None,
                "mcp_calls": len(mcp), "mcp_tools": [i.get("tool") for i in mcp],
                "mcp_source_unavailable_calls": data_statuses.count("source_unavailable"),
                "native_web_calls": sum(i["type"] == "webSearch" for i in research),
                "research_calls_observed": result.get("research_calls_observed"),
                "research_target_overshoot": result.get("research_target_overshoot"),
                "recorded_tool_errors": len(result.get("errors", [])),
                "error": result.get("error"),
                "source_urls_observed": sorted(set(re.findall(r'https?://[^\s"<>\\]+',
                                                            json.dumps(research)))),
                "raw_artifact_sha256": {name: hashlib.sha256(
                    (folder / name).read_bytes()).hexdigest()
                    for name in ("events.jsonl", "result.json", "settings.json",
                                 "mcp-inventory.json",
                                 "thread.json", "boundary-checks.json", "prompt.txt")
                    if (folder / name).exists()}})
    for name, data in [("receipts.json", {"rows": rows, "human_review": "not_performed",
            "scope": "Prescribed source-tool development walkthrough; Qwen off. No quality scores.",
            "raw_local_collection": str(args.source.resolve())}),
            ("user-preferences.json", {"reviewer_identity": None, "reviewed_at": None,
                                      "rows": preferences})]:
        (args.destination / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
