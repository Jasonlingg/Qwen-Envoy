"""Prepare or run two bounded v5 trials on one pinned, already cached paper."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runpod_qwen_bootstrap import BASE, BASE_REVISION, ADAPTER, ADAPTER_REVISION, ADAPTER_SHA256
from src.env.tools import TOOL_PREAMBLE
from src.eval.native_pilot_client import Deadline
from src.eval.known_paper_diagnostic import run_trial, training_prompt
from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.policies.runpod_queue import RunpodQueueTransport
from src.product.arxiv_library import ArxivLibrary
from src.research.agent import load_snapshot

VERSION = "2211.17192v2"
DOC_ID = "arxiv_2211_17192v2"
TITLE = "Fast Inference from Transformers via Speculative Decoding"
QUESTION = f'Use the known paper "{TITLE}" (doc_id: "{DOC_ID}") to answer: What batch size was used for the walltime measurements?'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    library = ArxivLibrary(args.library_root)
    # Never perform a network fetch, expose a reference answer, or substitute paper versions.
    doc = library.document(VERSION)
    snapshot = library.view([VERSION])
    snapshot_manifest, _ = load_snapshot(snapshot)
    prompt = training_prompt(ROOT / "data/sft/qasper-v5/train.jsonl")
    source_files = ["src/eval/known_paper_diagnostic.py", "scripts/run_known_paper_diagnostic.py",
                    "src/policies/runpod_queue.py", "src/policies/openai_compatible.py",
                    "src/env/repl.py", "src/env/tools.py"]
    manifest = {"schema_version": "known-paper-diagnostic-v1", "question_id": "known-paper-batch-size-v1",
        "question": QUESTION, "paper_version": VERSION, "doc_id": DOC_ID,
        "source_sha256": hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest(),
        "snapshot": str(snapshot), "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "corpus_hash": snapshot_manifest["corpus_hash"],
        "tool_preamble_sha256": hashlib.sha256(TOOL_PREAMBLE.encode()).hexdigest(),
        "checkpoint": f"{ADAPTER}@{ADAPTER_REVISION}/artifacts/full/checkpoint-50",
        "adapter_sha256": ADAPTER_SHA256, "base": f"{BASE}@{BASE_REVISION}",
        "endpoint_id": args.endpoint_id, "sandbox_image": args.sandbox_image,
        "split": "assistant-authored development diagnostic; paper seen in prior debugging",
        "hypothesis": "v5 can answer a specific known-paper question under its original training prompt and Python/citation protocol.",
        "expected_signal": "Successful source inspection followed by the correct answer and the named paper citation.",
        "decision_rule": "Count a trial as supported only after answer/citation/inspection checks and explicit source review. Two greedy repeats are not a reliability estimate or causal comparison.",
        "trials": 2, "max_steps_per_trial": 6, "requests_max_total": 12,
        "input_token_reservation_max": 86016, "output_token_reservation_max": 12288,
        "temperature": 0.0, "top_p": 1.0, "seed": 42, "thinking": False,
        "process_watchdog_seconds": 900,
        "catalog_active_gpu_hourly_usd": 0.69, "actual_spend_usd": None,
        "reward": "not_computed", "human_review": "not_performed",
        "implementation_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files}}
    write(args.output / "manifest.json", manifest)
    if "batch size of 1" not in doc["text"][20895:21506]:
        raise ValueError("reference source window changed")
    write(args.output / "reference.json", {"answer": "1", "reviewer_kind": "assistant_authored",
        "human_review": "not_performed", "source_start": 20895, "source_end": 21506,
        "source_text": doc["text"][20895:21506], "sent_to_model": False})
    (args.output / "system-prompt.txt").write_text(prompt)
    print(json.dumps({"status": "prepared", "manifest": str(args.output / "manifest.json")}))


def run(args):
    manifest = json.loads((args.output / "manifest.json").read_text())
    for path, digest in manifest["implementation_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError("diagnostic implementation changed after prepare")
    if manifest["endpoint_id"] != args.endpoint_id or manifest["sandbox_image"] != args.sandbox_image:
        raise ValueError("runtime arguments differ from prepared manifest")
    prompt = (args.output / "system-prompt.txt").read_text()
    if hashlib.sha256(prompt.encode()).hexdigest() != manifest["system_prompt_sha256"]:
        raise ValueError("prepared prompt changed")
    snapshot_manifest, _ = load_snapshot(Path(manifest["snapshot"]))
    if snapshot_manifest["corpus_hash"] != manifest["corpus_hash"]:
        raise ValueError("prepared corpus changed")
    # Mark the attempt before initializing clients; no automatic repeats of a session.
    with (args.output / "RUN_STARTED").open("x") as stream:
        stream.write("The prepared session has been attempted.\n")
    key = tomllib.loads((Path.home() / ".runpod/config.toml").read_text()).get("apikey")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION, local_files_only=True)
    transport = RunpodQueueTransport(args.endpoint_id, key, tokenizer,
                                     args.output / "requests.jsonl")
    trials = []
    for trial in range(1, 3):
        policy = OpenAICompatiblePolicy(endpoint=f"https://api.runpod.ai/v2/{args.endpoint_id}",
            model="envoy", api_key="", system_prompt=prompt,
            max_tokens=1024, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_p": 1.0, "seed": 42},
            transport=transport)
        result = run_trial(policy, Path(manifest["snapshot"]), manifest["question"], manifest["doc_id"],
                           image=args.sandbox_image)
        result["trial"] = trial
        write(args.output / f"trial-{trial}.json", result)
        trials.append(result)
        print(json.dumps({"trial": trial, "status": result["status"],
                          "requests": result["model_requests_attempted"]}), flush=True)
        if result["status"] == "error":
            break  # Stop on an infrastructure/policy error rather than repeat paid failures.
    write(args.output / "results.json", {"trials": trials, "planned_trials": 2,
        "unattempted_trials": list(range(len(trials) + 1, 3)), "human_review": "not_performed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run"])
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--sandbox-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase == "prepare": prepare(args)
    elif args.worker:
        run(args)
    else:
        child = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--worker"],
                                  start_new_session=True)
        deadline = Deadline(child.pid, 900)
        try:
            code = child.wait()
        finally:
            deadline.close()
            write(args.output / "watchdog.json", {"fired": deadline.fired.is_set()})
        raise SystemExit(code)


if __name__ == "__main__":
    main()
