"""Collect the approved native A/B development pilot; never score human quality."""

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


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "authorized_frozen_before_answers":
        raise ValueError("a frozen authorized execution manifest is required")
    for name, digest in manifest["input_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen pilot input changed: " + name)
    questions = json.loads((ROOT / manifest["questions_file"]).read_text())
    by_id = {q["id"]: q for q in questions["questions"]}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "execution-manifest.json", manifest)
    outcomes = []
    for question_id, *arms in manifest["arm_order"]:
        for arm in arms:
            print(json.dumps({"event": "starting", "question_id": question_id, "arm": arm}),
                  flush=True)
            case_path = args.output / f"{question_id}-{arm}"
            try:
                result = await run_case(ROOT, case_path, questions["common_instruction"],
                                        by_id[question_id], arm)
            except Exception as exc:
                saved = case_path / "result.json"
                result = (json.loads(saved.read_text()) if saved.exists() else {
                    "question_id": question_id, "arm": arm, "status": "setup_error",
                    "model_turns_attempted": 0, "answer_present": False})
                result["controller_error"] = f"{type(exc).__name__}: {exc}"[:2000]
                write_json(case_path / "controller-error.json", result)
            outcomes.append({key: result.get(key) for key in (
                "question_id", "arm", "status", "answer_present", "elapsed_seconds",
                "research_calls_observed", "research_target_overshoot", "model_turns_attempted",
                "token_usage", "error", "actual_spend_usd")})
            write_json(args.output / "collection.json", {"outcomes": outcomes,
                       "scheduled_answers": 10, "human_review": "not_performed"})
            print(json.dumps({"event": "finished", **outcomes[-1]}), flush=True)
            if result["status"] in {"setup_error", "generation_error", "failed"}:
                print(json.dumps({"event": "collection_stopped", "reason":
                    "Native client/model failure; retain attempt, no retry/fallback."}), flush=True)
                return


if __name__ == "__main__":
    asyncio.run(main())
