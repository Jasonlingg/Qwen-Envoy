"""Execute the frozen cases once; model Python remains inside the Docker reader."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from src.eval.assigned_smoke_check import check
from src.eval.assigned_smoke_plan import LIMITS, write_json
from src.policies.runpod_queue import RunpodQueueTransport
from src.product.assigned_reader import build_assigned_reader
from src.product.packet_verifier import verify_investigation_packet
from src.research.agent import load_snapshot


def audit_summary(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    counts = {event: sum(row.get("event") == event for row in rows)
              for event in ("reserved", "submitted", "completed", "failed")}
    counts["reported_usage"] = {}
    for row in rows:
        if row.get("event") == "completed":
            for key, value in row.get("response", {}).get("usage", {}).items():
                if type(value) is int:
                    counts["reported_usage"][key] = counts["reported_usage"].get(key, 0) + value
    return counts


def require_completed_cases(summary: dict):
    if summary["unattempted_cases"] or any(
            row["status"] in {"error", "unavailable"}
            or row["packet_verification"] != "passed" or row["audit"]["failed"]
            for row in summary["cases"]):
        raise RuntimeError("case infrastructure/policy/verification failure; results retained")


def track_sandbox(reader, output: Path):
    original = reader._repl

    def create(corpus):
        repl = original(corpus)
        start = repl.start_session

        def tracked_start():
            try:
                return start()
            finally:
                container = getattr(repl._impl, "_container_id", None)
                if container:
                    with (output / "containers.jsonl").open("a") as stream:
                        stream.write(json.dumps({"id": container}) + "\n")
                        stream.flush()
        repl.start_session = tracked_start
        return repl
    reader._repl = create


def run_cases(prepared: Path, output: Path, *, tokenizer, key: str,
              transport_factory=RunpodQueueTransport, reader_factory=build_assigned_reader) -> dict:
    check(prepared, tokenizer=tokenizer)
    plan = json.loads((prepared / "plan.json").read_text())
    inputs = json.loads((prepared / "model_inputs.json").read_text())
    manifest, docs = load_snapshot(Path(plan["snapshot"]))
    endpoint = plan["model"]["endpoint_id"]
    results = []
    for case, initial in zip(plan["cases"], inputs):
        name = case["case_id"]
        audit_path = output / f"{name}.requests.jsonl"
        # No reuse/replacement ledgers inside an attempted session.
        with audit_path.open("x"):
            pass
        transport = transport_factory(endpoint, key, tokenizer, audit_path,
            requests_max=LIMITS["model_requests_per_case"])
        first = True

        def guarded(url, payload, headers, timeout):
            nonlocal first
            if first and payload != initial["payload"]:
                raise ValueError("live initial request differs from frozen model input")
            first = False
            return transport(url, payload, headers, timeout)

        reader = reader_factory("qwen", qwen_endpoint=f"https://api.runpod.ai/v2/{endpoint}",
            qwen_model="envoy", qwen_checkpoint=(f'{plan["model"]["adapter"]}@'
                f'{plan["model"]["adapter_revision"]}/{plan["model"]["subfolder"]}'),
            max_steps=LIMITS["action_steps_per_case"], sandbox_image=plan["sandbox_image"],
            policy_transport=guarded)
        prompt_hash = hashlib.sha256(reader.system_prompt.encode()).hexdigest()
        if prompt_hash != plan["system_prompt_sha256"]:
            raise ValueError("live effective prompt differs from preparation")
        reader.model_identity.update(plan["model"])
        if reader.repl_factory is None:
            track_sandbox(reader, output)
        started = time.monotonic()
        try:
            packet = reader.investigate(case["question"], Path(plan["snapshot"]))
        finally:
            transport.audit.close()
        write_json(output / f"{name}.packet.json", packet)
        result = {"case_id": name, "status": packet["status"],
                  "seconds": time.monotonic() - started, "audit": audit_summary(audit_path),
                  "execution_error_steps": [], "action_rejection_steps": [],
                  "repeated_call_rejections": [],
                  "semantic_support": "not_reviewed", "human_review": "not_performed"}
        for trace in packet.get("trajectory", []):
            observation = trace.get("observation", "")
            if observation.startswith("Action rejected:"):
                result["action_rejection_steps"].append(trace["step"])
            if re.search(r"Traceback|(?:Type|Value|Name|Syntax|Runtime|Key|Index)Error:|ERROR:",
                         observation):
                result["execution_error_steps"].append(trace["step"])
            if "identical" in observation and "already executed" in observation:
                result["repeated_call_rejections"].append(trace["step"])
        try:
            compact = verify_investigation_packet(packet, manifest, docs,
                allowed_prompt_sha256s=frozenset({plan["system_prompt_sha256"]}),
                expected_reader_mode="assigned")
            write_json(output / f"{name}.verified.json", compact)
            result["packet_verification"] = "passed"
        except ValueError as exc:
            result.update(packet_verification="failed", verification_error=str(exc))
        results.append(result)
        summary = {"status": "completed_pending_claim_review", "cases": results,
            "unattempted_cases": [c["case_id"] for c in plan["cases"][len(results):]],
            "model_requests_reserved": sum(r["audit"]["reserved"] for r in results),
            "reward": "not_computed", "human_review": "not_performed"}
        write_json(output / "results.json", summary)
        print(json.dumps(result), flush=True)
        if (packet["status"] in {"error", "unavailable"} or result["audit"]["failed"]
                or result["packet_verification"] != "passed"):
            break
    return summary
