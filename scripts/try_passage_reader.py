"""Prepare and run one bounded three-case development attempt through experiment MCP."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp import Client

from scripts.experiment_mode_mcp import create_server
from scripts.reader_comparison import credential
from scripts.runpod_qwen_bootstrap import BASE, BASE_REVISION
from src.env.repl import PersistentREPL
from src.eval.assigned_smoke_lifecycle import supervise
from src.eval.assigned_smoke_run import track_sandbox
from src.eval.worker_log_reader import WorkerLogReader
from src.policies.runpod_queue import RunpodQueueTransport
from src.product.arxiv_library import ArxivLibrary
from src.product.assigned_reader import build_assigned_reader
from src.product.experiment_budget import ExperimentBudget
from src.product.experiment_tokenizer import load_tokenizer
from src.product.ml_experiment_evidence import EvidenceResolver
from src.product.ml_experiment_store import ExperimentStore
from src.product.passage_reader_service import PassageReaderService
from src.product.passage_scope import PASSAGES

ENDPOINT = "q05pjrvguakieg"
IMAGE = "sha256:79bd8bf84bd4b7f2a32a0596cb1fd90164716c1cfca879f78829411193646382"
MARKER = ROOT / "out/passage-reader-three-case-LIVE_ATTEMPT"
LIMITS = {"model_requests": 12, "tokens_reserved": 98304, "seconds": 900, "gpus": 1}
FILES = ["scripts/try_passage_reader.py", "scripts/experiment_mode_mcp.py",
    "src/product/passage_reader_service.py", "src/product/passage_scope.py",
    "src/product/assigned_guidance.py", "src/product/assigned_actions.py",
    "src/product/assigned_reader.py", "src/product/qwen_investigator.py",
    "src/product/packet_verifier.py", "src/product/paper_assignment.py",
    "src/product/experiment_budget.py", "src/policies/runpod_queue.py",
    "src/eval/assigned_smoke_lifecycle.py", "src/eval/worker_log_reader.py",
    "src/env/repl.py", "src/env/tools.py", "src/env/tools_v2.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def hashes():
    return {name: sha(ROOT / name) for name in FILES}


def prepare(output):
    if not PersistentREPL._docker_available(IMAGE):
        raise ValueError("pinned Docker image unavailable")
    cases = json.loads((ROOT / "examples/base-v5-reader-comparison/cases.json").read_text())
    refs = json.loads((ROOT / "examples/base-v5-reader-comparison/references.json").read_text())
    selected = ("qlora-dtypes", "awq-protocol", "paged-target")
    by_id = {r["case_id"]: r for r in refs["cases"]}
    inputs = [{"case_id": name, "question": next(c["question"] for c in cases
                if c["case_id"] == name),
               "arxiv_id": next(c["paper_version"] for c in cases if c["case_id"] == name),
               "windows": [{k: w[k] for k in ("start", "end")}
                           for w in by_id[name]["source_contexts"]]} for name in selected]
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    old = ROOT / "out/base-v5-comparison-20261020/library/papers"
    for case in inputs:
        version = case["arxiv_id"]
        shutil.copytree(old / version, output / "library/papers" / version)
        for name in ("document.json", "notes.json"):
            relative = "library/papers/" + version + "/" + name
            sources[relative] = sha(output / relative)
        document = ArxivLibrary(output / "library").document(version)
        for context in by_id[case["case_id"]]["source_contexts"]:
            if document["text"][context["start"]:context["end"]] != context["text"]:
                raise ValueError("development source window differs")
    save(output / "cases.json", inputs)
    save(output / "review-input.json", {"data_kind": "exposed_development",
        "human_review": None, "sent_to_model": False,
        "cases": [by_id[name] for name in selected]})
    plan = {"schema_version": "three-case-passage-reader-v1", "limits": LIMITS,
        "endpoint": ENDPOINT, "sandbox_image": IMAGE, "implementation_sha256": hashes(),
        "source_sha256": sources, "cases_sha256": sha(output / "cases.json"),
        "review_input_sha256": sha(output / "review-input.json"),
        "tokenizer": BASE, "tokenizer_revision": BASE_REVISION,
        "authorization": "Preparation is not GPU authorization", "provider_dollar_cap": False}
    save(output / "plan.json", plan)
    return plan


def check(output):
    plan = json.loads((output / "plan.json").read_text())
    if (plan["limits"] != LIMITS or plan["endpoint"] != ENDPOINT
            or plan["sandbox_image"] != IMAGE or plan["implementation_sha256"] != hashes()
            or plan["cases_sha256"] != sha(output / "cases.json")
            or plan["review_input_sha256"] != sha(output / "review-input.json")
            or any(sha(output / p) != h for p, h in plan["source_sha256"].items())):
        raise ValueError("prepared implementation or sources changed")
    return plan


async def run_cases(output, tokenizer, key=None, *, scripted=False):
    check(output)
    cases = json.loads((output / "cases.json").read_text())
    review = json.loads((output / "review-input.json").read_text())
    refs = {r["case_id"]: r for r in review["cases"]}
    folder = output / ("scripted" if scripted else "live")
    folder.mkdir(exist_ok=False)
    library = ArxivLibrary(output / "library")
    store = ExperimentStore(folder / "records", evidence_resolver=EvidenceResolver(
        library_root=library.root))
    queue = None
    if not scripted:
        with (output / "session.requests.jsonl").open("x"):
            pass
        queue = RunpodQueueTransport(ENDPOINT, key, tokenizer,
                                    output / "session.requests.jsonl", requests_max=12,
                                    wait_seconds=120)
    first_inputs, reports = {}, []
    expected_inputs = ({} if scripted else json.loads((output / "first-inputs.json").read_text()))
    try:
        for case in cases:
            name, doc_id = case["case_id"], "arxiv_" + case["arxiv_id"].replace(".", "_")
            actions = [f'print(passage("{doc_id}", {w["start"]}, {w["end"] - w["start"]}))'
                       for w in case["windows"]]
            if name == "paged-target":
                actions += ['report_gap(missing="The target p95 latency is not supplied in '
                    'these passages.", follow_up_query="Llama A5000 p95 measured latency")',
                    'submit(answer="Unanswerable", citations=[], evidence=[])']
            else:
                evidence = [{"doc_id": doc_id, **w} for w in case["windows"]]
                actions += ["submit(answer=" + repr(" ".join(refs[name]["requirements"]))
                            + ", citations=" + repr([doc_id])
                            + ", evidence=" + repr(evidence) + ")"]
            scripted_actions, first = iter(actions), True
            def transport(url, payload, headers, timeout):
                nonlocal first
                if first:
                    if scripted:
                        first_inputs[name] = payload
                    elif payload != expected_inputs[name]:
                        raise ValueError("live first request differs from the frozen input")
                    first = False
                if not scripted:
                    return queue(url, payload, headers, timeout)
                content = next(scripted_actions)
                count = len(tokenizer.apply_chat_template(payload["messages"], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False))
                return {"model": "envoy", "choices": [{"message": {"content": content},
                    "finish_reason": "stop"}], "usage": {"prompt_tokens": count,
                    "completion_tokens": len(tokenizer.encode(content))}}
            def factory():
                budget = ExperimentBudget(tokenizer, transport, requests_max=4, tokens_max=32768)
                reader = build_assigned_reader("qwen", qwen_endpoint=
                    "https://api.runpod.ai/v2/" + ENDPOINT, qwen_model="envoy",
                    qwen_checkpoint="jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5@"
                    "27b912a863ff914ad45baa03624d8911dc1e17fb/artifacts/full/checkpoint-50",
                    reader_guidance=PASSAGES, max_steps=4, sandbox_image=IMAGE,
                    policy_transport=budget)
                track_sandbox(reader, output)
                return reader, budget
            service = PassageReaderService(store, library, factory, folder / "reports",
                                           data_kind="synthetic" if scripted else "recorded")
            server = create_server(store, library, None, passage_reader=service)
            async with Client(server) as client:
                saved = await client.call_tool("save_experiment_brief", {
                    "brief": {"question": case["question"]}})
                experiment_id = saved.structured_content["experiment_id"]
                arguments = {"experiment_id": experiment_id, "expected_revision": 1,
                    "note_id": "passage_read", "question": case["question"],
                    "arxiv_id": case["arxiv_id"], "windows": case["windows"]}
                result = await client.call_tool("read_experiment_passages", arguments)
                if result.is_error:
                    raise ValueError("passage MCP call failed: " + str(result.content)[:500])
                report = result.structured_content
                before = queue.remaining if queue else None
                repeated = await client.call_tool("read_experiment_passages", arguments)
                if repeated.structured_content != report or queue and queue.remaining != before:
                    raise ValueError("identical MCP retry changed the report or dispatched")
                resumed = await client.call_tool("resume_experiment", {
                    "experiment_id": experiment_id})
            save(folder / f"{name}.json", report)
            save(folder / f"{name}.resume.json", resumed.structured_content)
            reports.append({"case_id": name, "status": report["status"],
                "experiment_id": experiment_id, "note_attached": report["note_attached"],
                "budget": report.get("budget"), "same_note_retry_without_dispatch": True,
                "semantic_support": "not_reviewed"})
            summary = {"data_kind": "synthetic" if scripted else "recorded", "cases": reports,
                "unattempted_cases": [c["case_id"] for c in cases[len(reports):]],
                "requests_reserved": sum(r["budget"]["requests_dispatched"] for r in reports),
                "human_review": None, "reward": "not_computed"}
            save(folder / "summary.json", summary)
            if report.get("budget", {}).get("transport_failed") or report["status"] in {
                    "error", "unavailable"}:
                break
        if scripted:
            save(output / "first-inputs.json", first_inputs)
        return summary
    finally:
        if queue is not None:
            queue.audit.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run", "worker"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    tokenizer = load_tokenizer(BASE, BASE_REVISION)
    if args.phase == "prepare":
        prepare(args.output)
        summary = asyncio.run(run_cases(args.output, tokenizer, scripted=True))
        if summary["unattempted_cases"] or not all(c["note_attached"] for c in summary["cases"]):
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
        if (MARKER.read_text().strip() != str(args.output.resolve())
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
    with MARKER.open("x") as stream:
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
