"""Run the separately approved frozen smoke once, with cold-start/shutdown guard."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runpod_qwen_bootstrap import BASE, BASE_REVISION
from src.env.repl import PersistentREPL
from src.eval.assigned_smoke_check import check
from src.eval.assigned_smoke_lifecycle import supervise
from src.eval.assigned_smoke_plan import LIMITS, write_json
from src.eval.assigned_smoke_run import require_completed_cases, run_cases

FILES = ["scripts/run_assigned_reader_smoke.py", "src/eval/assigned_smoke_run.py",
         "src/eval/assigned_smoke_lifecycle.py", "src/eval/assigned_smoke_cleanup.py"]


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["run", "worker"])
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approved-live", action="store_true")
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION,
        local_files_only=True, trust_remote_code=False)
    check(args.prepared, tokenizer=tokenizer)
    plan = json.loads((args.prepared / "plan.json").read_text())
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        key = tomllib.loads((Path.home() / ".runpod/config.toml").read_text()).get("apikey")
    if not key:
        raise ValueError("Runpod credential required; never print its contents")
    if args.phase == "worker":
        runtime = json.loads((args.output / "runtime.json").read_text())
        if (runtime.get("implementation_sha256") != hashes()
                or runtime.get("prepared_sha256") != hashlib.sha256(
                    (args.prepared / "plan.json").read_bytes()).hexdigest()
                or runtime.get("approved_live") is not True):
            raise ValueError("live worker differs from the approved execution manifest")
        with (args.output / "WORKER_STARTED").open("x"):
            pass
        summary = run_cases(args.prepared, args.output, tokenizer=tokenizer, key=key)
        require_completed_cases(summary)
        return
    if not args.approved_live:
        parser.error("run requires separate live activation approval (--approved-live)")
    if not PersistentREPL._docker_available(plan["sandbox_image"]):
        raise ValueError("pinned Docker image unavailable; GPU was not activated")
    args.output.mkdir(parents=True, exist_ok=False)
    # Global exclusive attempt marker, shared across output directories/copies of the plan.
    marker = ROOT / "out/assigned-reader-two-case-LIVE_ATTEMPT"
    with marker.open("x") as stream:
        stream.write(str(args.output.resolve()) + "\n")
    write_json(args.output / "runtime.json", {"approved_live": True,
        "authorization": "User said 'ok go' after the live $0.69/hour A5000 quote.",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "prepared_sha256": hashlib.sha256((args.prepared / "plan.json").read_bytes()).hexdigest(),
        "implementation_sha256": hashes(),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "limits": LIMITS, "catalog_gpu_hourly_usd": 0.69,
        "provider_dollar_cap": False, "actual_spend_usd": None})
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--prepared",
               str(args.prepared.resolve()), "--output", str(args.output.resolve())]
    supervise(plan["model"]["endpoint_id"], args.output, command, key=key,
              seconds=LIMITS["whole_session_watchdog_seconds"])


if __name__ == "__main__":
    main()
