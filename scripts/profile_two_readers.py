"""Prepare locally; execute only inside a separately approved guarded GPU session."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reader_comparison import credential
from scripts.runpod_qwen_bootstrap import BASE, BASE_REVISION
from src.eval.assigned_smoke_control import cli
from src.eval.assigned_smoke_lifecycle import supervise
from src.eval.reader_profile import ProfileQueueClient, assess_profile, run_profile
from src.eval.reader_profile_worker import canonical_hash
from src.policies.runpod_queue import RunpodQueueTransport
from src.product.experiment_tokenizer import load_tokenizer

ENDPOINT = "q05pjrvguakieg"
INPUT_MANIFEST = ROOT / "examples/experiment-mode-walkthrough/profile-input-manifest.json"
IMPLEMENTATION = ("scripts/profile_two_readers.py", "src/eval/reader_profile.py",
                  "src/eval/reader_profile_worker.py", "scripts/runpod_qwen_bootstrap.py")
LIMITS = {"model_requests": 10, "telemetry_jobs": 9, "tokens_reserved": 81920,
          "session_seconds": 900, "gpus": 1}
CONFIG = {"MAX_NUM_SEQS": "2", "MAX_CONCURRENCY": "2", "MAX_MODEL_LEN": "8192",
          "ENABLE_PREFIX_CACHING": "false"}
MARKER = ROOT / "out/two-reader-profile-LIVE_ATTEMPT"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(output, tokenizer):
    manifest = json.loads(INPUT_MANIFEST.read_text())
    source = ROOT / manifest["source"]["path"]
    if sha(source) != manifest["source"]["sha256"]:
        raise ValueError("historical input ledger differs")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    payloads, counts = {}, {}
    for sample in manifest["samples"]:
        matches = [r["payload"] for r in rows if r.get("event") == "reserved"
                   and r.get("request") == sample["parent_request"]]
        if len(matches) != 1 or canonical_hash(matches[0]) != sample["payload_sha256"]:
            raise ValueError("historical input selection differs")
        payload = matches[0]
        if (payload.get("model") != "envoy" or payload.get("max_tokens") != 1024
                or payload.get("chat_template_kwargs") != {"enable_thinking": False}
                or payload.get("temperature") != 0 or payload.get("top_p") != 1
                or payload.get("seed") != 42 or payload.get("n", 1) != 1
                or payload.get("stream", False) is not False):
            raise ValueError("historical input decoding differs")
        tokens = tokenizer.apply_chat_template(payload["messages"], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if (not isinstance(tokens, list) or any(type(t) is not int for t in tokens)
                or not 1 <= len(tokens) <= 7168
                or len(tokens) != sample["historical_recomputed_prompt_tokens"]):
            raise ValueError("profile tokenizer/input count differs")
        payloads[sample["sample_id"]] = payload
        counts[sample["sample_id"]] = len(tokens)
    if len(payloads) != 2:
        raise ValueError("exactly two pinned inputs required")
    output.mkdir(parents=True, exist_ok=False)
    plan = {"schema_version": "reader-serving-profile-v1", "endpoint": ENDPOINT,
            "session": uuid4().hex, "source_manifest_sha256": sha(INPUT_MANIFEST),
            "payload_hashes": {name: canonical_hash(p) for name, p in payloads.items()},
            "prompt_tokens": counts, "tokenizer": BASE, "tokenizer_revision": BASE_REVISION,
            "limits": LIMITS, "config": CONFIG,
            "implementation_sha256": {p: sha(ROOT / p) for p in IMPLEMENTATION},
            "paid_authorization": "not_granted_by_preparation", "provider_dollar_cap": False}
    write(output / "payloads.json", payloads)
    write(output / "plan.json", plan)
    bootstrap = bootstrap_source(plan)
    (output / "entrypoint.py").write_text(bootstrap)
    return plan


def bootstrap_source(plan):
    """Inline operator code into the existing image; no remote code downloads."""
    base = (ROOT / "scripts/runpod_qwen_bootstrap.py").read_text()
    hook = (ROOT / "src/eval/reader_profile_worker.py").read_text()
    return ("import json, os, runpy, sys\n"
        "sys.path.insert(0, '/src')\n"
        f"base = {{'__name__': 'envoy_pins'}}\nexec({base!r}, base)\n"
        f"hook = {{'__name__': 'envoy_profile'}}\nexec({hook!r}, hook)\n"
        "from huggingface_hub import snapshot_download\nimport torch, vllm, runpod\n"
        "identity = base['configure'](snapshot_download)\n"
        "identity.update(torch_version=torch.__version__, vllm_version=vllm.__version__, "
        "cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0), "
        "gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)\n"
        "print('ENVOY_PINNED_IDENTITY ' + json.dumps(identity, sort_keys=True), flush=True)\n"
        f"monitor = hook['ProfileHook']({plan['session']!r}, {plan['payload_hashes']!r})\n"
        "original_start = runpod.serverless.start\n"
        "def monitored_start(config):\n"
        "    config = dict(config)\n"
        "    config['handler'] = monitor.wrap(config['handler'])\n"
        "    return original_start(config)\n"
        "runpod.serverless.start = monitored_start\n"
        "runpy.run_path('/src/main.py', run_name='__main__')\n")


def checked(output):
    plan = json.loads((output / "plan.json").read_text())
    payloads = json.loads((output / "payloads.json").read_text())
    if (plan["endpoint"] != ENDPOINT or plan["source_manifest_sha256"] != sha(INPUT_MANIFEST)
            or plan["limits"] != LIMITS or plan["config"] != CONFIG
            or plan["implementation_sha256"] != {p: sha(ROOT / p) for p in IMPLEMENTATION}
            or plan["payload_hashes"] != {name: canonical_hash(p) for name, p in payloads.items()}
            or (output / "entrypoint.py").read_text() != bootstrap_source(plan)):
        raise ValueError("prepared profile inputs, code or entrypoint changed")
    return plan, payloads


def authorization(output, plan):
    record = json.loads((output / "authorization.json").read_text())
    if (record.get("approved") is not True or record.get("limits") != LIMITS
            or record.get("plan_sha256") != sha(output / "plan.json")
            or type(record.get("budget_usd")) not in (int, float)
            or record["budget_usd"] not in (0.5, 1)
            or type(record.get("catalog_gpu_hourly_usd")) not in (int, float)
            or not 0 < record["catalog_gpu_hourly_usd"] <= 0.69
            or not record.get("user_authorization") or record.get("attempts") != 1):
        raise ValueError("profile needs plan-bound GPU authorization and a current price")
    checked_at = datetime.fromisoformat(record["price_checked_at"])
    if (checked_at.utcoffset() is None
            or not 0 <= (datetime.now(timezone.utc) - checked_at).total_seconds() <= 3600):
        raise ValueError("current GPU price read is stale")
    if record["catalog_gpu_hourly_usd"] * LIMITS["session_seconds"] / 3600 > record["budget_usd"]:
        raise ValueError("GPU time guard exceeds selected budget")
    return record


def guarded_run(output, plan, approved, *, control=cli, supervisor=supervise, key=None):
    key = key or credential()
    class Endpoint:
        base, _key = f"https://api.runpod.io/v2/serverless/{ENDPOINT}", key
    def api(method, body=None):
        return RunpodQueueTransport._request(Endpoint(), method, "", body)
    before = api("GET")
    expected_entrypoint = ["python3", "-u", "-c",
                           (ROOT / "scripts/runpod_qwen_bootstrap.py").read_text()]
    if (before.get("entrypoint") != expected_entrypoint
            or before.get("workers", {}).get("min") != 0
            or before.get("workers", {}).get("max") != 0
            or {k: before.get("env", {}).get(k) for k in CONFIG} != CONFIG
            or before.get("image") !=
            "registry.runpod.net/runpod-workers-worker-vllm-main-dockerfile:b7212acc6"):
        raise ValueError("parked endpoint image, entrypoint or serving configuration differs")
    with MARKER.open("x") as stream:
        stream.write(str(output.resolve()) + "\n")
    deadline = time.time() + LIMITS["session_seconds"]
    write(output / "runtime.json", {"approved_live": True,
        "plan_sha256": sha(output / "plan.json"), "deadline_unix": deadline,
        "authorization": approved,
        "original_entrypoint_sha256": canonical_hash(expected_entrypoint)})
    entrypoint = ["python3", "-u", "-c", (output / "entrypoint.py").read_text()]
    def stable(value):
        return {k: v for k, v in value.items() if k not in {"args", "entrypoint"}}
    changed = False
    try:
        # Mark uncertain mutations before dispatch so restoration is still attempted.
        changed = True
        api("PATCH", {"entrypoint": entrypoint})
        after = api("GET")
        if after.get("entrypoint") != entrypoint or stable(after) != stable(before):
            raise ValueError("instrumentation read-back changed unexpected endpoint settings")
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--output",
                   str(output.resolve())]
        supervisor(ENDPOINT, output, command, key=key, seconds=900,
                   logged_running_ready=True, deadline_unix=deadline)
    finally:
        if changed:
            control("update", ENDPOINT, "--workers-min", "0", "--workers-max", "0")
            api("PATCH", {"entrypoint": expected_entrypoint})
            restored = api("GET")
            ok = (restored.get("entrypoint") == expected_entrypoint
                  and stable(restored) == stable(before))
            write(output / "entrypoint-restoration.json", {"restored": ok,
                "entrypoint_sha256": canonical_hash(restored.get("entrypoint"))})
            if not ok:
                raise RuntimeError("endpoint restoration read-back failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "check", "run", "worker"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.phase == "prepare":
        tokenizer = load_tokenizer(BASE, BASE_REVISION)
        plan = prepare(args.output, tokenizer)
    else:
        plan, payloads = checked(args.output)
    if args.phase in {"run", "worker"}:
        approved = authorization(args.output, plan)
    if args.phase == "run":
        guarded_run(args.output, plan, approved)
    if args.phase == "worker":
        runtime = json.loads((args.output / "runtime.json").read_text())
        if (MARKER.read_text().strip() != str(args.output.resolve())
                or runtime.get("plan_sha256") != sha(args.output / "plan.json")
                or runtime.get("approved_live") is not True
                or runtime["deadline_unix"] <= time.time()):
            raise ValueError("worker is not bound to the live session and its absolute deadline")
        with (args.output / "WORKER_STARTED").open("x"):
            pass
        client = ProfileQueueClient(ENDPOINT, credential(), plan, payloads,
                                    args.output / "profile.requests.jsonl")
        try:
            result = run_profile(client, list(payloads),
                                 lambda r: write(args.output / "result.json", r))
            result["assessment"] = assess_profile(result)
            write(args.output / "result.json", result)
            if not result["assessment"]["comparable"]:
                raise RuntimeError("profile incomplete or incomparable; no retry")
        finally:
            client.audit.close()
    print(json.dumps({"status": args.phase, "limits": plan["limits"],
                      "model_calls_in_prepare_or_check": 0}))


if __name__ == "__main__":
    main()
