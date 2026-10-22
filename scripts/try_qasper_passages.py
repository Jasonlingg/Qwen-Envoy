"""One bounded V5 passage-v2 smoke over three already exposed QASPER questions."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp import Client
from mcp.server import MCPServer

from benchmarks.envoybench.frozen_dataset import DEFAULT_DATASET, load_split
from scripts.reader_comparison import credential
from scripts.runpod_qwen_bootstrap import BASE, BASE_REVISION
from scripts.try_passage_reader import ENDPOINT, IMAGE, LIMITS, save, sha
from scripts.try_passage_reader import FILES as PRIOR_FILES
from src.env.repl import PersistentREPL
from src.eval.artifacts import content_hash
from src.eval.assigned_smoke_lifecycle import supervise
from src.eval.assigned_smoke_plan import MODEL_IDENTITY
from src.eval.assigned_smoke_run import track_sandbox
from src.eval.qasper_passage_smoke import RETRIEVAL, SELECTION, retrieve, select_questions
from src.eval.reader_comparison_run import action_metrics, identity
from src.eval.worker_log_reader import WorkerLogReader
from src.policies.runpod_queue import RunpodQueueTransport
from src.product.assigned_reader import build_assigned_reader
from src.product.experiment_budget import ExperimentBudget
from src.product.experiment_tokenizer import load_tokenizer
from src.product.packet_verifier import verify_investigation_packet
from src.product.passage_scope import PASSAGES_V2
from src.research.agent import load_snapshot

MARKER = ROOT / "out/qasper-passage-v2-LIVE_ATTEMPT"
FILES = sorted(set(PRIOR_FILES + ["scripts/try_qasper_passages.py",
    "src/eval/qasper_passage_smoke.py", "src/product/reader_gaps.py",
    "benchmarks/envoybench/frozen_dataset.py", "src/product/experiment_tokenizer.py",
    "scripts/runpod_qwen_bootstrap.py"]))


def hashes():
    return {name: sha(ROOT / name) for name in FILES}


def prepare(output):
    if not PersistentREPL._docker_available(IMAGE):
        raise ValueError("pinned Docker image unavailable")
    benchmark, source, benchmark_path, corpus, _ = load_split(DEFAULT_DATASET, "test_candidate")
    selected = select_questions(benchmark["questions"])
    output.mkdir(parents=True, exist_ok=False)
    cases, sources = [], {}
    for question in selected:
        # Unanswerable rows intentionally have no required citation IDs.
        doc_id = "qasper_" + re.sub(r"[^A-Za-z0-9_-]+", "_", question["source_paper_id"])
        path = corpus / (doc_id + ".json")
        doc = json.loads(path.read_text())
        if doc["metadata"]["qasper_paper_id"] != question["source_paper_id"]:
            raise ValueError("question/paper identity differs")
        windows, ranking = retrieve(doc["text"], question["source_question"])
        case_id = question["id"]
        snapshot = output / "snapshots" / case_id
        destination = snapshot / "corpus" / path.name
        destination.parent.mkdir(parents=True)
        doc["metadata"]["source_revision"] = source["source_revision"]
        save(destination, doc)
        manifest = {k: source[k] for k in ("source_dataset", "source_revision", "source_split")}
        manifest.update(schema_version="research-snapshot-v1", corpus_hash=content_hash(
            destination.parent), source_corpus_hash=source["corpus_hash"],
            coverage_note="QASPER/S2ORC frozen dataset text, not current arXiv extraction")
        save(snapshot / "manifest.json", manifest)
        save(snapshot / "retrieval.json", {"config": RETRIEVAL, "ranking": ranking,
            "original_document_sha256": sha(path), "selected_windows": windows})
        cases.append({"case_id": case_id, "source_paper_id": question["source_paper_id"],
            "question": question["source_question"], "doc_id": doc_id, "windows": windows,
            "snapshot": str(snapshot.relative_to(output))})
        for item in snapshot.rglob("*.json"):
            sources[str(item.relative_to(output))] = sha(item)
    save(output / "cases.json", cases)
    # Scoring-only record: never read by run_cases or the policy/transport.
    save(output / "review-input.json", {"sent_to_model": False, "human_review": None,
        "cases": selected})
    plan = {"schema_version": "qasper-passage-v2-smoke-v1", "limits": LIMITS,
        "model_arm": "v5", "model_identity": identity({"models": MODEL_IDENTITY}, "v5"),
        "endpoint": ENDPOINT, "sandbox_image": IMAGE, "implementation_sha256": hashes(),
        "source_sha256": sources, "cases_sha256": sha(output / "cases.json"),
        "review_input_sha256": sha(output / "review-input.json"),
        "original_benchmark_sha256": sha(benchmark_path),
        "original_corpus_hash": source["corpus_hash"], "source_revision": source["source_revision"],
        "data_kind": "previously_exposed_development_not_held_out", "selection": SELECTION,
        "retrieval": RETRIEVAL, "reader_guidance": PASSAGES_V2, "seed": 42,
        "remote_seed_enforced": False, "reward": "not_computed",
        "tokenizer": BASE, "tokenizer_revision": BASE_REVISION,
        "authorization": "Preparation is not GPU authorization", "provider_dollar_cap": False}
    save(output / "plan.json", plan)
    return plan


def check(output):
    plan = json.loads((output / "plan.json").read_text())
    if (plan["model_identity"] != identity({"models": MODEL_IDENTITY}, "v5")
            or plan["limits"] != LIMITS or plan["endpoint"] != ENDPOINT
            or plan["sandbox_image"] != IMAGE or plan["implementation_sha256"] != hashes()
            or plan["reader_guidance"] != PASSAGES_V2
            or plan["cases_sha256"] != sha(output / "cases.json")
            or plan["review_input_sha256"] != sha(output / "review-input.json")
            or any(sha(output / p) != h for p, h in plan["source_sha256"].items())):
        raise ValueError("prepared implementation or sources changed")
    return plan


async def run_cases(output, tokenizer, key=None, *, scripted=False):
    plan = check(output)
    model = plan["model_identity"]
    cases = json.loads((output / "cases.json").read_text())
    folder = output / ("scripted" if scripted else "live")
    folder.mkdir(exist_ok=False)
    queue = None
    if not scripted:
        with (output / "session.requests.jsonl").open("x"):
            pass
        queue = RunpodQueueTransport(ENDPOINT, key, tokenizer,
            output / "session.requests.jsonl", requests_max=12,
            wait_seconds=120, served_model=model["served_model"])
    first_inputs, reports = {}, []
    expected = {} if scripted else json.loads((output / "first-inputs.json").read_text())
    try:
        for case in cases:
            name = case["case_id"]
            snapshot = output / case["snapshot"]
            manifest, docs = load_snapshot(snapshot)
            actions = iter([f'print(passage({case["doc_id"]!r}, {w["start"]}, '
                f'{w["end"] - w["start"]}))' for w in case["windows"]] + [
                'submit_gap(missing="Scripted protocol fixture; not a semantic judgment", '
                'follow_up_query="unexecuted scripted follow-up")'])
            first = True

            def transport(url, payload, headers, timeout):
                nonlocal first
                if first:
                    if scripted:
                        first_inputs[name] = payload
                    elif payload != expected[name]:
                        raise ValueError("live first request differs from frozen input")
                    first = False
                if not scripted:
                    return queue(url, payload, headers, timeout)
                content = next(actions)
                count = len(tokenizer.apply_chat_template(payload["messages"], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False))
                return {"model": model["served_model"], "choices": [{
                    "message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": count,
                              "completion_tokens": len(tokenizer.encode(content))}}

            budget = ExperimentBudget(tokenizer, transport, requests_max=4, tokens_max=32768)
            reader = build_assigned_reader("qwen", qwen_endpoint=
                "https://api.runpod.ai/v2/" + ENDPOINT, qwen_model=model["served_model"],
                qwen_checkpoint=model["checkpoint"], reader_guidance=PASSAGES_V2,
                max_steps=4, sandbox_image=IMAGE, policy_transport=budget)
            reader.model_identity.update(model)
            track_sandbox(reader, output)
            cache = {}
            server = MCPServer("Frozen QASPER evaluation adapter")

            @server.tool()
            def investigate_frozen_qasper(case_id: str) -> dict:
                """Inspect a frozen evaluation case; no discovery or ingestion."""
                if case_id != name:
                    raise ValueError("case not assigned")
                if cache:
                    return cache["result"]
                # Mark invocation before model dispatch; failed calls cannot retry inference.
                cache["result"] = {"status": "failed_attempt_no_retry"}
                packet = reader.investigate(case["question"], snapshot,
                                            passage_windows=case["windows"])
                save(folder / (name + ".packet.json"), packet)
                checked = verify_investigation_packet(packet, manifest, docs,
                    allowed_prompt_sha256s=frozenset({hashlib.sha256(
                        reader.system_prompt.encode()).hexdigest()}),
                    expected_reader_mode="assigned", expected_reader_guidance=PASSAGES_V2,
                    expected_passage_windows=case["windows"])
                save(folder / (name + ".verified.json"), checked)
                cache["result"] = checked
                return checked

            started = time.monotonic()
            async with Client(server) as client:
                reply = await client.call_tool("investigate_frozen_qasper", {"case_id": name})
                before = len(budget.records)
                retry = await client.call_tool("investigate_frozen_qasper", {"case_id": name})
                same = (not reply.is_error and not retry.is_error
                        and reply.structured_content == retry.structured_content
                        and len(budget.records) == before)
            save(folder / (name + ".mcp.json"), {"transport": "in_process",
                "tool": "investigate_frozen_qasper", "is_error": reply.is_error,
                "result": reply.structured_content, "same_retry_without_dispatch": same})
            raw = folder / (name + ".packet.json")
            packet = json.loads(raw.read_text()) if raw.exists() else {}
            report = {"case_id": name, "status": packet.get("status", "error"),
                "budget": budget.summary(), "seconds": time.monotonic() - started,
                "mcp_verification": "failed" if reply.is_error else "passed",
                "same_retry_without_dispatch": same, **action_metrics(packet),
                "semantic_support": "not_reviewed", "human_review": None}
            reports.append(report)
            summary = {"data_kind": "synthetic_responses" if scripted else "recorded",
                "cases": reports, "unattempted_cases": [c["case_id"] for c in cases[len(reports):]],
                "requests_reserved": sum(r["budget"]["requests_dispatched"] for r in reports),
                "human_review": None, "reward": "not_computed"}
            save(folder / "summary.json", summary)
            print(json.dumps(report), flush=True)
            if budget.transport_failed:
                break
        if scripted:
            save(output / "first-inputs.json", first_inputs)
        return summary
    finally:
        if queue:
            queue.audit.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run", "worker"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    marker = MARKER
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    tokenizer = load_tokenizer(BASE, BASE_REVISION)
    if args.phase == "prepare":
        prepare(args.output)
        summary = asyncio.run(run_cases(args.output, tokenizer, scripted=True))
        if (summary["unattempted_cases"] or not all(
                c["status"] == "no_evidence" and c["mcp_verification"] == "passed"
                for c in summary["cases"])):
            raise ValueError("scripted preflight did not complete")
        plan = check(args.output)
        plan["first_inputs_sha256"] = sha(args.output / "first-inputs.json")
        save(args.output / "plan.json", plan)
        print(json.dumps(summary))
        return
    plan = check(args.output)
    if plan["first_inputs_sha256"] != sha(args.output / "first-inputs.json"):
        raise ValueError("frozen model inputs changed")
    approval = json.loads((args.output / "authorization.json").read_text())
    if (approval.get("approved") is not True or approval.get("attempts") != 1
            or approval.get("plan_sha256") != sha(args.output / "plan.json")
            or approval.get("limits") != LIMITS or approval.get("budget_usd") != 0.5
            or type(approval.get("hourly_usd")) not in (int, float)
            or not 0 < approval["hourly_usd"] <= 0.69 or not approval.get("user_instruction")
            or not 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(
                approval["price_checked_at"])).total_seconds() <= 3600):
        raise ValueError("one current, plan-bound authorization required")
    key = credential()
    if args.phase == "worker":
        runtime = json.loads((args.output / "runtime.json").read_text())
        if (marker.read_text().strip() != str(args.output.resolve())
                or runtime["plan_sha256"] != sha(args.output / "plan.json")
                or time.time() >= runtime["deadline_unix"]):
            raise ValueError("worker is outside its single authorized session")
        with (args.output / "WORKER_STARTED").open("x"):
            pass
        print(json.dumps(asyncio.run(run_cases(args.output, tokenizer, key))))
        return
    if not PersistentREPL._docker_available(IMAGE):
        raise ValueError("Docker unavailable; no activation")
    class Endpoint:
        base, _key = f"https://api.runpod.io/v2/serverless/{ENDPOINT}", key
    before = RunpodQueueTransport._request(Endpoint(), "GET", "")
    if (before.get("workers", {}).get("min") != 0 or before.get("workers", {}).get("max") != 0
            or before.get("image") !=
            "registry.runpod.net/runpod-workers-worker-vllm-main-dockerfile:b7212acc6"
            or any(before.get("env", {}).get(k) != v for k, v in {
                "MAX_NUM_SEQS": "2", "MAX_CONCURRENCY": "2", "MAX_MODEL_LEN": "8192",
                "ENABLE_PREFIX_CACHING": "false"}.items())
            or before.get("entrypoint") != ["python3", "-u", "-c", (
                ROOT / "scripts/runpod_qwen_bootstrap.py").read_text()]):
        raise ValueError("endpoint must be parked with its original pinned bootstrap")
    with marker.open("x") as stream:
        stream.write(str(args.output.resolve()) + "\n")
    deadline = time.time() + 900
    save(args.output / "runtime.json", {"deadline_unix": deadline,
        "plan_sha256": sha(args.output / "plan.json"), "authorization": approval,
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--output",
               str(args.output.resolve())]
    try:
        supervise(ENDPOINT, args.output, command, key=key, seconds=900,
            deadline_unix=deadline, logged_running_ready=True,
            log_reader=WorkerLogReader(ENDPOINT, key))
    finally:
        after = RunpodQueueTransport._request(Endpoint(), "GET", "")
        save(args.output / "configuration-restoration.json", {"full_configuration_restored":
            before == after,
            "entrypoint_restored": before.get("entrypoint") == after.get("entrypoint")})
        if before != after:
            raise RuntimeError("full endpoint configuration restoration failed")


if __name__ == "__main__":
    main()
