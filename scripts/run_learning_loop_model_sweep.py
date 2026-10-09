"""Run the registered Hugging Face adapter sweep on the frozen learning pilot.

Run on one GPU pod after copying the source bundle and frozen candidate IDs.
Existing base-Qwen3 and targeted-SFT transcripts are reused for review.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "out/research/learning-loop-model-sweep-20260926"
QUESTIONS = ROOT / "out/research/learning-loop-code-pilot-run-20260926/candidate-ids/retrieved_ids.json"
CORPUS = ROOT / "out/research/starter-2026-09-12/corpus"
PROMPT_SUFFIX = ROOT / "data/prompts/learning_loop_evidence_v1.txt"
BASES = {
    "qwen3": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "qwen25": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
ARMS = [
    ("base3", "qwen3", None, None, None),
    ("early3", "qwen3", "jasonlingg/rlm-explorer-qwen3-8b-qasper-sft", "7cbe6289760d7718886fe57e3a6f4c45905a5704", "epoch-1"),
    ("aligned3", "qwen3", "jasonlingg/qwen-envoy-qwen3-8b-qasper-sft", "012d3c7aa13956702846892a1176ed2a170e923e", "final"),
    ("v5_3", "qwen3", "jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5", "27b912a863ff914ad45baa03624d8911dc1e17fb", "artifacts/full/checkpoint-50"),
    ("targeted3", "qwen3", "jasonlingg/qwen-envoy-qwen3-8b-qasper-targeted-sft-v1", "fc4bf6a47dd238dd1f3b745bc8322ae8250093a2", None),
    ("base25", "qwen25", None, None, None),
    ("sft25", "qwen25", "jasonlingg/doctracerrl-sft-qwen2.5-7b", "71ecc4c6748694c7f9bfac24e53b63c72fd1c5ff", None),
    ("grpo25", "qwen25", "jasonlingg/doctracerrl-grpo-qwen2.5-7b", "f84aa524a8f02b39e9bc0e461fe1454b994592bc", None),
    ("grpo25_50", "qwen25", "jasonlingg/doctracerrl-grpo-qwen2.5-7b-50steps", "3e28a2f44ed3085a8f7eab16586bab4befd5c0b8", None),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not QUESTIONS.is_file() or not CORPUS.is_dir() or not PROMPT_SUFFIX.is_file():
        raise SystemExit("Frozen candidate questions, corpus, or prompt suffix is missing")
    base_paths: dict[str, str] = {}
    for label, family, repo, revision, subfolder in ARMS:
        output = OUTPUT / f"{label}.json"
        if output.exists():
            if len(json.loads(output.read_text())) != 8:
                raise SystemExit(f"Incomplete existing output: {output}")
            print(f"skip completed {label}", flush=True)
            continue
        if family not in base_paths:
            base_id, base_revision = BASES[family]
            print(f"download base {base_id} @ {base_revision}", flush=True)
            base_paths[family] = snapshot_download(
                repo_id=base_id, revision=base_revision,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt"],
            )
        adapter_path = None
        weight_sha256 = None
        if repo is not None:
            config = Path(hf_hub_download(repo, "adapter_config.json", subfolder=subfolder, revision=revision))
            weight = Path(hf_hub_download(repo, "adapter_model.safetensors", subfolder=subfolder, revision=revision))
            if config.parent != weight.parent:
                raise RuntimeError(f"Adapter config and weight paths differ for {label}")
            adapter_path = str(weight.parent)
            weight_sha256 = sha256(weight)
        base_id, base_revision = BASES[family]
        identity = {
            "label": label, "base_model": base_id, "base_revision": base_revision,
            "adapter_repo": repo, "adapter_revision": revision,
            "adapter_subfolder": subfolder, "adapter_sha256": weight_sha256,
        }
        (OUTPUT / f"{label}.identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        env = {**os.environ, "BASE_MODEL_PATH": base_paths[family]}
        if adapter_path:
            env["CHECKPOINT_PATH"] = adapter_path
        else:
            env.pop("CHECKPOINT_PATH", None)
        cmd = [
            sys.executable, "scripts/run_eval.py", "--policy",
            "qwen_sft_policy" if adapter_path else "qwen_base_policy",
            "--questions", str(QUESTIONS), "--corpus", str(CORPUS),
            "--max-steps", "15", "--seed", "42", "--require-evidence",
            "--system-prompt-suffix", str(PROMPT_SUFFIX),
            "--no-vector-index", "--evidence-verifier",
            "--verifier-feedback-budget", "4", "--escalate-after-verifier-failure",
            "--run-label", label, "--output", str(output),
        ]
        print(f"run {label}", flush=True)
        with (OUTPUT / f"{label}.log").open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"completed {label}: exit={result.returncode}", flush=True)
        if result.returncode or not output.is_file() or len(json.loads(output.read_text())) != 8:
            raise SystemExit(f"Failed or incomplete {label}; see {label}.log")


if __name__ == "__main__":
    main()
