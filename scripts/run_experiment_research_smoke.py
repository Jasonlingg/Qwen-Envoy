"""Prepare or execute one explicitly authorized experiment investigation; always park afterward."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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
    args = parser.parse_args()
    output = args.output.resolve()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    tokenizer = load_tokenizer("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
    if not PersistentREPL._docker_available(IMAGE):
        raise ValueError("pinned Docker image unavailable; no GPU activation")
    if args.phase == "prepare":
        output.mkdir(parents=True, exist_ok=True)
        with (output / "plan.json").open("x") as stream:
            json.dump(plan(), stream, indent=2)
            stream.write("\n")
        print(json.dumps({"prepared": str(output), "limits": LIMITS, "model_calls": 0}))
        return
    checked_plan(output)
    sha = hashlib.sha256((output / "plan.json").read_bytes()).hexdigest()
    marker = ROOT / "out/experiment-research-LIVE_ATTEMPT"
    if args.phase == "worker":
        runtime = json.loads((output / "runtime.json").read_text())
        if (marker.read_text().strip() != str(output) or runtime["plan_sha256"] != sha
                or runtime["approved_live"] is not True):
            raise ValueError("worker is not bound to the exclusive live session")
        run(output, tokenizer=tokenizer, key=credential())
        return
    if not args.approved_live:
        parser.error("run needs --approved-live for this new bounded session")
    with marker.open("x") as stream:
        stream.write(str(output) + "\n")
    write_json(output / "runtime.json", {"approved_live": True, "plan_sha256": sha,
        "user_authorization": "User asked 'cna we test it', paused, then 'ok contienu then' "
                              "after the one-question/12-request/900-second limits were stated.",
        "started_at": datetime.now(timezone.utc).isoformat(), "limits": LIMITS,
        "catalog_serverless_gpu_hourly_usd": 0.69, "provider_dollar_cap": False})
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output)]
    supervise(ENDPOINT, output, command, key=credential(), seconds=LIMITS["watchdog_seconds"])


if __name__ == "__main__":
    main()
