"""Execute the predeclared paired schedule using injected, metered model clients."""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.eval.research_pilot_bundle import load_bundle, source_identity, write_json
from src.eval.research_pilot_host import SYSTEM_PROMPT, run_host
from src.eval.research_pilot_tools import FrozenPilotTools, ReaderAdmission


def _dispatched(records):
    return [row for row in records if row["status"] != "observer_error"]


def run_pilot(bundle: Path, destination: Path, *, host_factory, investigator,
              budget, config: dict, repo: Path, execution_kind="live_models") -> dict:
    """No infrastructure is provisioned here; a caller explicitly supplies endpoints.

    The injected budget must wrap both model transports. Only transport records,
    not reader-reported attempt counters, establish actual model attempts.
    """
    if destination.exists():
        raise ValueError("Run destination already exists; overwriting/resuming is forbidden")
    if execution_kind not in {"live_models", "fake_test"}:
        raise ValueError("Unknown execution kind")
    manifest, data, library = load_bundle(bundle)
    identity = source_identity(repo)
    if identity["implementation_sha256"] != manifest["source_identity"]["implementation_sha256"]:
        raise ValueError("Implementation changed after preparation; create a new frozen bundle")
    destination.mkdir(parents=True)
    packets, packet_lock = [], threading.Lock()
    def save_packet(packet):
        with packet_lock:
            filename = f"reader-packets/{len(packets) + 1:03d}.json"
            write_json(destination / filename, packet)
            packets.append(filename)
    reader = ReaderAdmission(investigator, limit=2, record_packet=save_packet)
    run = {"schema_version": "research-workflow-pilot-run-v1", "status": "running",
           "execution_kind": execution_kind, "bundle_sha256": manifest["bundle_sha256"],
           "questions": data["questions"], "review_strata": data["review_strata"],
           "corpus_hash": manifest["corpus_hash"], "papers": manifest["papers"],
           "plan": manifest["plan"], "seed": manifest["seed"],
           "host_system_prompt": SYSTEM_PROMPT,
           "host_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
           "reader_prompt_sha256": hashlib.sha256(reader.system_prompt.encode()).hexdigest(),
           "configuration": config, "source_identity": identity,
           "started_at": datetime.now(timezone.utc).isoformat(),
           "reward_version": "not_applicable", "semantic_review": "not_performed",
           "actual_model_requests": 0, "actual_qwen_requests": 0,
           "costs": None, "cold_start_seconds": None,
           "timing_scope": "warm tool/inference time; GPU startup and idle excluded",
           "results": []}
    questions = {q["id"]: q for q in data["questions"]}
    policies = []
    def persist():
        run["budget"] = budget.summary()
        records = _dispatched(run["budget"]["records"])
        run["actual_model_requests"] = len(records)
        run["actual_qwen_requests"] = sum(row["label"] == "qwen" for row in records)
        write_json(destination / "run.json", run)
    persist()
    try:
        for slot in manifest["plan"]:
            question, arm = questions[slot["question_id"]], slot["arm"]
            policy = host_factory(question["id"], arm)
            if (policy.system_prompt != SYSTEM_PROMPT or any(policy is p for p in policies)
                    or getattr(policy, "history", [])):
                raise ValueError("Every arm requires a fresh host with the same pinned prompt")
            policies.append(policy)
            tools = FrozenPilotTools(library, question, reader)
            before = len(budget.summary()["records"])
            packet_start = len(packets)
            result = run_host(question, arm, policy, tools.callbacks(arm),
                              max_turns=8, validate_finish=tools.validate_finish)
            records = _dispatched(budget.summary()["records"][before:])
            result.update(question_id=question["id"], provenance_checks=result.pop("checks"),
                          telemetry={"host_requests": sum(r["label"].startswith("host:")
                                                          for r in records),
                                     "qwen_requests": sum(r["label"] == "qwen" for r in records),
                                     "request_ids": [r["request_id"] for r in records]},
                          reader_packets=packets[packet_start:])
            run["results"].append(result)
            persist()
            if budget.summary()["blocked_reason"]:
                run["stop_reason"] = budget.summary()["blocked_reason"]
                break
            if len(run["results"]) <= 2 and (
                    result["status"] != "submitted" or
                    (arm == "delegated" and result["telemetry"]["qwen_requests"] == 0)):
                run["stop_reason"] = "Initial-question smoke failed; later questions unattempted"
                break
        ledger = budget.summary()
        complete = (not ledger["blocked_reason"] and ledger["usage_complete"]
                    and all(r["status"] == "ok" for r in ledger["records"])
                    and len(run["results"]) == len(manifest["plan"])
                    and all(row["status"] == "submitted" for row in run["results"])
                    and all(row["telemetry"]["qwen_requests"] > 0
                            for row in run["results"] if row["arm"] == "delegated"))
        run["status"] = "complete" if complete else "incomplete"
    except BaseException as exc:
        run.update(status="interrupted", exception_class=type(exc).__name__)
        raise
    finally:
        run["finished_at"] = datetime.now(timezone.utc).isoformat()
        persist()
    return run
