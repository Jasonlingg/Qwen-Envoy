"""Run the registered no-training Qwen3 v5 prompt and decoding diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "out/research/learning-loop-inference-diagnostic-20260926"
QUESTIONS = ROOT / "out/research/learning-loop-code-pilot-run-20260926/candidate-ids/retrieved_ids.json"
CORPUS = ROOT / "out/research/starter-2026-09-12/corpus"
BASE_REPO = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER_REPO = "jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5"
ADAPTER_REVISION = "27b912a863ff914ad45baa03624d8911dc1e17fb"
ADAPTER_SUBFOLDER = "artifacts/full/checkpoint-50"
ARMS = [
    ("checklist_greedy_42", "learning_loop_evidence_claim_check_v1.txt", 0.0, 1.0, 42),
    ("sampled_42", "learning_loop_evidence_v1.txt", 0.7, 0.8, 42),
    ("sampled_43", "learning_loop_evidence_v1.txt", 0.7, 0.8, 43),
    ("sampled_44", "learning_loop_evidence_v1.txt", 0.7, 0.8, 44),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in [QUESTIONS, ROOT / "data/research/learning_loop_code_pilot_v1.json"]:
        if not path.is_file():
            raise SystemExit(f"Missing frozen questions: {path}")
    if not CORPUS.is_dir():
        raise SystemExit(f"Missing frozen corpus: {CORPUS}")

    base = Path(snapshot_download(
        repo_id=BASE_REPO,
        revision=BASE_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt"],
    ))
    config = Path(hf_hub_download(
        ADAPTER_REPO, "adapter_config.json", subfolder=ADAPTER_SUBFOLDER,
        revision=ADAPTER_REVISION,
    ))
    weights = Path(hf_hub_download(
        ADAPTER_REPO, "adapter_model.safetensors", subfolder=ADAPTER_SUBFOLDER,
        revision=ADAPTER_REVISION,
    ))
    if config.parent != weights.parent:
        raise RuntimeError("Adapter config and weights are not together")
    identity = {
        "base_repo": BASE_REPO,
        "base_revision": BASE_REVISION,
        "adapter_repo": ADAPTER_REPO,
        "adapter_revision": ADAPTER_REVISION,
        "adapter_subfolder": ADAPTER_SUBFOLDER,
        "adapter_sha256": sha256(weights),
    }
    (OUTPUT / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")

    for label, prompt_name, temperature, top_p, seed in ARMS:
        transcript = OUTPUT / f"{label}.json"
        if transcript.exists():
            rows = json.loads(transcript.read_text())
            if len(rows) != 8 or any(row.get("status") == "error" for row in rows):
                raise SystemExit(f"Existing arm is incomplete: {transcript}")
            print(f"skip {label}", flush=True)
            continue
        prompt = ROOT / "data/prompts" / prompt_name
        if not prompt.is_file():
            raise SystemExit(f"Missing prompt: {prompt}")
        env = {**os.environ,
            "BASE_MODEL_PATH": str(base),
            "CHECKPOINT_PATH": str(weights.parent),
            "ENVOY_QWEN_TEMPERATURE": str(temperature),
            "ENVOY_QWEN_TOP_P": str(top_p),
        }
        cmd = [
            sys.executable, "scripts/run_eval.py", "--policy", "qwen_sft_policy",
            "--questions", str(QUESTIONS), "--corpus", str(CORPUS),
            "--max-steps", "15", "--seed", str(seed), "--require-evidence",
            "--system-prompt-suffix", str(prompt), "--no-vector-index",
            "--evidence-verifier", "--verifier-feedback-budget", "4",
            "--escalate-after-verifier-failure", "--run-label", label,
            "--output", str(transcript),
        ]
        print(f"run {label}", flush=True)
        with (OUTPUT / f"{label}.log").open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not transcript.is_file():
            raise SystemExit(f"Failed {label}; see {label}.log")
        rows = json.loads(transcript.read_text())
        if len(rows) != 8 or any(row.get("status") == "error" for row in rows):
            raise SystemExit(f"Incomplete {label}; see {label}.log")
        print(f"completed {label}", flush=True)


if __name__ == "__main__":
    main()
