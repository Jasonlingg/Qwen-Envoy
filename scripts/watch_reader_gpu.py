"""Independent local cutoff for this run's explicitly recorded Runpod resources.

Keep this process awake with caffeinate on macOS. It is an estimated-cost/time
watchdog, not a provider-enforced dollar cap, and needs working control-plane access.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def cli(*args):
    result = subprocess.run(["runpodctl", *args], capture_output=True, text=True, timeout=20)
    body = result.stdout if result.returncode == 0 else result.stderr
    # Some CLI warnings precede JSON. Never print raw credential-bearing config.
    start = body.find("{")
    if start < 0:
        raise RuntimeError("Runpod CLI returned no JSON receipt")
    value, _ = json.JSONDecoder().raw_decode(body[start:])
    if result.returncode and value.get("code") != "not_found":
        raise RuntimeError("Runpod control-plane failure: " + str(value.get("code")))
    return value


def validate(record):
    for key in ("started", "deadline", "budget_usd", "hourly_total_upper_bound"):
        if type(record.get(key)) not in {int, float} or not math.isfinite(record[key]):
            raise ValueError("finite lifecycle bounds required")
    maximum_seconds, maximum_budget = 7200, 5
    if record.get("pilot_protocol") is not None:
        import hashlib

        root = Path(__file__).resolve().parents[1]
        expected = "examples/reader-rl-pilot-v2/protocol.json"
        if record["pilot_protocol"] != expected:
            raise ValueError("unknown lifecycle budget protocol")
        path = root / expected
        protocol = json.loads(path.read_text())
        allowance = protocol["budget_proposal"]
        authorization = record.get("paid_authorization_receipt", {})
        if (hashlib.sha256(path.read_bytes()).hexdigest() != record.get("pilot_protocol_sha256")
                or authorization.get("authorized") is not True
                or not authorization.get("authorization")
                or authorization.get("protocol_sha256") != record.get("pilot_protocol_sha256")
                or authorization.get("budget_usd") != allowance["estimated_cost_ceiling_usd"]
                or authorization.get("maximum_lifetime_seconds") != allowance[
                    "maximum_total_owned_GPU_lifetime_seconds"]):
            raise ValueError("explicit matching paid authorization receipt required")
        maximum_seconds = allowance["maximum_total_owned_GPU_lifetime_seconds"]
        maximum_budget = allowance["estimated_cost_ceiling_usd"]
        if record["hourly_total_upper_bound"] > allowance["maximum_combined_hourly_rate_usd"]:
            raise ValueError("fresh combined price exceeds the proposed pilot rate")
    seconds = record["deadline"] - record["started"]
    if (not 0 < seconds <= maximum_seconds or not 0 < record["budget_usd"] <= maximum_budget
            or record["hourly_total_upper_bound"] <= 0
            or seconds * record["hourly_total_upper_bound"] / 3600 > record["budget_usd"] * .9
            or not record.get("volume") or record["volume"].get("id") is None):
        raise ValueError("lifecycle exceeds the declared allowance or lacks owned volume")
    if record.get("pod") and not record["pod"].get("id"):
        raise ValueError("owned pod requires its create receipt")


def cleanup(record, call=cli):
    """Only IDs + creation timestamps from this launch's create receipts."""
    proof = {}
    for kind in ("pod", "network-volume"):
        receipt = record.get("pod" if kind == "pod" else "volume")
        if not receipt:
            continue
        ident = receipt["id"]
        current = call(kind, "get", ident)
        if current.get("code") != "not_found":
            if current.get("id") != ident or (
                    receipt.get("createdAt") and creation_time(current.get("createdAt"))
                    != creation_time(receipt["createdAt"])):
                raise ValueError("resource no longer matches this launch's create receipt")
            call(kind, "delete", ident)
        absent = call(kind, "get", ident).get("code") == "not_found"
        proof[kind] = {"id": ident, "absent": absent}
        if not absent:
            raise RuntimeError("owned resource deletion could not be verified")
    return proof


def creation_time(value):
    """MCP uses RFC3339; runpodctl 2.14 emits Go's UTC timestamp string."""
    if not isinstance(value, str):
        raise ValueError("resource creation timestamp is missing")
    if value.endswith(" +0000 UTC"):
        value = value.removesuffix(" +0000 UTC").replace(" ", "T") + "+00:00"
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def watch(directory, call=cli, clock=time.time, sleep=time.sleep):
    directory = Path(directory)
    while True:
        record = json.loads((directory / "owned.json").read_text())
        validate(record)
        if clock() >= record["deadline"] or (directory / "stop-now").exists():
            proof = cleanup(record, call)
            (directory / "watchdog-cleanup.json").write_text(json.dumps(proof, indent=2) + "\n")
            return proof
        heartbeat = directory / "watchdog-heartbeat.json"
        temporary = heartbeat.with_suffix(".tmp")
        temporary.write_text(json.dumps({"pid": os.getpid(), "time": clock(),
                                        "deadline": record["deadline"]}))
        temporary.replace(heartbeat)
        # Do not depend on the agent, SSH session, training process or its logs.
        sleep(max(0, min(5, record["deadline"] - clock())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    while True:
        try:
            watch(args.directory)
            break
        except (RuntimeError, subprocess.SubprocessError):
            # A transient control-plane failure must never silently disable cleanup.
            time.sleep(5)
