"""One matched V5 attempt: frozen hybrid passages, unchanged reader and historical baseline."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import try_qasper_passages as previous
from scripts.try_qasper_passages import (
    BASE,
    BASE_REVISION,
    ENDPOINT,
    IMAGE,
    LIMITS,
    PersistentREPL,
    RunpodQueueTransport,
    WorkerLogReader,
    credential,
    load_tokenizer,
    save,
    sha,
    supervise,
)
from src.product.passage_scope import PASSAGES_V2, make_scope
from src.research.agent import load_snapshot

MARKER = ROOT / "out/qasper-hybrid-v5-LIVE_ATTEMPT"
BASELINE = ROOT / "examples/qasper-passage-v2"
RETRIEVAL = ROOT / "examples/hybrid-passage-retrieval"
ADDITIONAL_FILES = ["scripts/try_qasper_hybrid_v5.py", "src/research/hybrid_passages.py",
                    "scripts/compare_passage_retrieval.py"]


def additional_hashes():
    return {name: sha(ROOT / name) for name in ADDITIONAL_FILES}


def reference_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for root in (BASELINE, RETRIEVAL)
            for p in sorted(root.rglob("*")) if p.is_file()}


def selected_windows(case, assignment, document):
    """Accept only the exact precomputed, label-free assignment for this source/query."""
    if (assignment.get("method") != "hybrid" or assignment.get("split") != "test_candidate"
            or assignment.get("question_id") != case["case_id"]
            or assignment.get("doc_id") != case["doc_id"]
            or assignment.get("question") != case["question"]
            or assignment.get("status") != "selected"
            or assignment.get("source_sha256") != hashlib.sha256(
                document["text"].encode()).hexdigest()
            or assignment.get("counts_as_source_inspection") is not False):
        raise ValueError("hybrid assignment differs from the matched question/source")
    windows = assignment.get("windows")
    make_scope({case["doc_id"]: document}, windows, version=PASSAGES_V2)
    if len(windows) != 2 or any(w["end"] - w["start"] != 1600 for w in windows):
        raise ValueError("matched test requires two 1600-character windows")
    return copy.deepcopy(windows)


def prepare(output):
    plan = previous.prepare(output)
    baseline_cases = json.loads((BASELINE / "frozen/cases.json").read_text())
    cases = json.loads((output / "cases.json").read_text())
    if cases != baseline_cases:
        raise ValueError("historical case selection or source changed")
    assignments = json.loads((RETRIEVAL / "reader-assignments.json").read_text())
    rows = json.loads((RETRIEVAL / "test_candidate.json").read_text())["questions"]
    for case in cases:
        snapshot = output / case["snapshot"]
        _, docs = load_snapshot(snapshot)
        document = docs[case["doc_id"]]
        (assignment,) = [r for r in assignments if r["question_id"] == case["case_id"]
                        and r["method"] == "hybrid" and r["split"] == "test_candidate"]
        case["windows"] = selected_windows(case, assignment, document)
        (row,) = [r for r in rows if r["question_id"] == case["case_id"]
                 and r["method"] == "hybrid"]
        if row["windows"] != case["windows"]:
            raise ValueError("hybrid assignment and recorded ranking differ")
        for relative in ["manifest.json", "corpus/" + case["doc_id"] + ".json"]:
            if sha(snapshot / relative) != sha(BASELINE / "frozen" / case["snapshot"] / relative):
                raise ValueError("hybrid test changed the frozen paper snapshot")
        old = json.loads((snapshot / "retrieval.json").read_text())
        save(snapshot / "retrieval.json", {"method": "hybrid", "tuning": "none",
            "ranking": row["ranking"], "selected_windows": case["windows"],
            "original_document_sha256": old["original_document_sha256"],
            "source_report_sha256": sha(RETRIEVAL / "test_candidate.json")})
    if sha(output / "review-input.json") != sha(BASELINE / "frozen/review-input.json"):
        raise ValueError("scoring references changed")
    save(output / "cases.json", cases)
    plan.update(schema_version="qasper-hybrid-v5-matched-v1",
        cases_sha256=sha(output / "cases.json"),
        additional_implementation_sha256=additional_hashes(), reference_sha256=reference_hashes(),
        retrieval={"kind": "frozen_plain_hybrid_rrf", "rrf_k": 60, "reranker": False,
                   "characters": 3200, "historical_baseline": "examples/qasper-passage-v2"})
    plan["source_sha256"] = {name: sha(output / name) for name in plan["source_sha256"]}
    save(output / "plan.json", plan)
    return plan


def check(output):
    plan = previous.check(output)
    if (plan.get("schema_version") != "qasper-hybrid-v5-matched-v1"
            or plan.get("additional_implementation_sha256") != additional_hashes()
            or plan.get("reference_sha256") != reference_hashes()
            or "first_inputs_sha256" in plan and "first_input_comparison_sha256" not in plan
            or "first_input_comparison_sha256" in plan and plan[
                "first_input_comparison_sha256"] != sha(output / "first-input-comparison.json")):
        raise ValueError("hybrid implementation, historical baseline or comparison proof changed")
    return plan


def replace_scope_only(payload, old_scope, new_scope):
    unchanged = set(old_scope) - {"windows", "read_calls"}
    if (set(old_scope) != set(new_scope)
            or any(old_scope[k] != new_scope[k] for k in unchanged)):
        raise ValueError("matched first request may change only passage ranges/read examples")
    old_json = json.dumps(old_scope, ensure_ascii=False)
    new_json = json.dumps(new_scope, ensure_ascii=False)
    expected = copy.deepcopy(payload)
    content = expected["messages"][-1]["content"]
    if content.count(old_json) != 1:
        raise ValueError("historical first input must contain exactly one passage scope")
    expected["messages"][-1]["content"] = content.replace(old_json, new_json, 1)
    return expected


def compare_first_inputs(output):
    baseline_cases = {c["case_id"]: c for c in json.loads(
        (BASELINE / "frozen/cases.json").read_text())}
    old = json.loads((BASELINE / "frozen/first-inputs.json").read_text())
    new = json.loads((output / "first-inputs.json").read_text())
    cases = json.loads((output / "cases.json").read_text())
    if set(new) != set(old):
        raise ValueError("first-request case IDs changed")
    proof = []
    for case in cases:
        name = case["case_id"]
        _, docs = load_snapshot(output / case["snapshot"])
        old_scope = make_scope(docs, baseline_cases[name]["windows"], version=PASSAGES_V2)
        new_scope = make_scope(docs, case["windows"], version=PASSAGES_V2)
        expected = replace_scope_only(old[name], old_scope, new_scope)
        if new[name] != expected:
            raise ValueError("first request changed beyond passage metadata")
        proof.append({"case_id": name, "only_passage_scope_changed": True,
            "system_prompt_unchanged": True, "decoding_unchanged": True,
            "source_snapshot_unchanged": True, "old_windows": old_scope["windows"],
            "new_windows": new_scope["windows"]})
    save(output / "first-input-comparison.json", {"historical_reference": str(
        BASELINE.relative_to(ROOT)), "cases": proof, "remote_seed_guaranteed": False,
        "historical_control_limitations": "no contemporaneous/randomized control"})


async def run_cases(output, tokenizer, key=None, *, scripted=False):
    check(output)
    return await previous.run_cases(output, tokenizer, key, scripted=scripted)


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
        compare_first_inputs(args.output)
        plan = check(args.output)
        plan["first_input_comparison_sha256"] = sha(args.output / "first-input-comparison.json")
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
