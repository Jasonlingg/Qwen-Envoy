"""Prepare or execute one explicitly authorized experiment investigation; always park afterward."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reader_comparison import credential
from src.env.repl import PersistentREPL
from src.eval.assigned_smoke_lifecycle import supervise
from src.eval.assigned_smoke_plan import write_json
from src.eval.experiment_smoke import ENDPOINT, IMAGE, LIMITS, checked_plan, plan, run
from src.product.experiment_tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run", "worker"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--approved-live", action="store_true")
    parser.add_argument("--resume-startup", action="store_true",
                        help="startup-only correction within original absolute deadline")
    args = parser.parse_args()
    output = args.output.resolve()
    plan_name = "plan-amended.json" if args.resume_startup else "plan.json"
    runtime_name = "runtime-resumed.json" if args.resume_startup else "runtime.json"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    tokenizer = load_tokenizer("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
    if not PersistentREPL._docker_available(IMAGE):
        raise ValueError("pinned Docker image unavailable; no GPU activation")
    if args.phase == "prepare":
        output.mkdir(parents=True, exist_ok=True)
        with (output / plan_name).open("x") as stream:
            json.dump(plan(), stream, indent=2)
            stream.write("\n")
        print(json.dumps({"prepared": str(output), "limits": LIMITS, "model_calls": 0}))
        return
    checked_plan(output, plan_name)
    sha = hashlib.sha256((output / plan_name).read_bytes()).hexdigest()
    marker = ROOT / "out/experiment-research-LIVE_ATTEMPT"
    if args.phase == "worker":
        runtime = json.loads((output / runtime_name).read_text())
        if (marker.read_text().strip() != str(output) or runtime["plan_sha256"] != sha
                or runtime["approved_live"] is not True):
            raise ValueError("worker is not bound to the exclusive live session")
        run(output, tokenizer=tokenizer, key=credential(), plan_name=plan_name)
        return
    if not args.approved_live:
        parser.error("run needs --approved-live for this new bounded session")
    deadline = None
    if args.resume_startup:
        original = json.loads((output / "runtime.json").read_text())
        lifecycle = json.loads((output / "lifecycle.json").read_text())
        deadline = datetime.fromisoformat(original["started_at"]).timestamp() + 900
        if (original.get("approved_live") is not True or original.get("limits") != LIMITS
                or original["plan_sha256"] != hashlib.sha256(
                    (output / "plan.json").read_bytes()).hexdigest()
                or marker.read_text().strip() != str(output) or not lifecycle["shutdown_verified"]
                or (output / "WORKER_STARTED").exists()
                or list(output.glob("*.requests.jsonl")) or deadline - time.time() < 60):
            raise ValueError("resume needs verified parking, zero inference and remaining time")
        archive = output / "startup-attempt-1"
        archive.mkdir(exist_ok=False)
        for name in ("lifecycle.json", "endpoint-before.json", "endpoint-activated.json",
                     "health-latest.json", "workers-latest.json", "startup-log-reads.jsonl"):
            if (output / name).exists():
                shutil.move(output / name, archive / name)
    else:
        with marker.open("x") as stream:
            stream.write(str(output) + "\n")
    write_json(output / runtime_name, {"approved_live": True, "plan_sha256": sha,
        "user_authorization": "User asked 'cna we test it', paused, then 'ok contienu then' "
                              "after the one-question/12-request/900-second limits were stated.",
        "started_at": datetime.now(timezone.utc).isoformat(), "limits": LIMITS,
        "catalog_serverless_gpu_hourly_usd": 0.69, "provider_dollar_cap": False,
        "original_deadline_unix": deadline, "startup_readiness_amendment": args.resume_startup})
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output)]
    if args.resume_startup:
        command.append("--resume-startup")
    supervise(ENDPOINT, output, command, key=credential(), seconds=LIMITS["watchdog_seconds"],
              logged_running_ready=args.resume_startup, deadline_unix=deadline)


if __name__ == "__main__":
    main()
