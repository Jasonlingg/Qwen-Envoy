"""Entrypoint for the pinned Runpod worker-vllm image (public weights only).

Resolve both revisions before starting vLLM. Serving a local immutable snapshot
also prevents the upstream worker's revision-recovery path from using HF HEAD.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

BASE = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER = "jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5"
ADAPTER_REVISION = "27b912a863ff914ad45baa03624d8911dc1e17fb"
SUBFOLDER = "artifacts/full/checkpoint-50"
ADAPTER_SHA256 = "7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3"


def configure(snapshot_download):
    base = Path(snapshot_download(repo_id=BASE, revision=BASE_REVISION,
                                  allow_patterns=["*.json", "*.safetensors", "*.jinja",
                                                  "*.txt", "*.model"]))
    adapter = Path(snapshot_download(repo_id=ADAPTER, revision=ADAPTER_REVISION,
                                     allow_patterns=[f"{SUBFOLDER}/*"])) / SUBFOLDER
    config = json.loads((adapter / "adapter_config.json").read_text())
    digest = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    if digest != ADAPTER_SHA256 or config.get("base_model_name_or_path") != BASE:
        raise ValueError("pinned adapter verification failed")
    if config.get("r") != 4:
        raise ValueError("unexpected adapter rank")
    os.environ.update(MODEL_NAME=str(base), TOKENIZER_NAME=str(base),
                      MODEL_REVISION="", TOKENIZER_REVISION="",
                      LORA_MODULES=json.dumps({"name": "envoy", "path": str(adapter),
                                               "base_model_name": str(base)}),
                      OPENAI_SERVED_MODEL_NAME_OVERRIDE="envoy-base-only")
    return {"base_repo": BASE, "base_revision": BASE_REVISION,
            "adapter_repo": ADAPTER, "adapter_revision": ADAPTER_REVISION,
            "adapter_subfolder": SUBFOLDER, "adapter_sha256": digest,
            "adapter_rank": 4, "served_adapter": "envoy"}


def main():
    from huggingface_hub import snapshot_download
    import torch
    import vllm
    identity = configure(snapshot_download)
    identity.update(torch_version=torch.__version__, vllm_version=vllm.__version__,
                    cuda_runtime=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0),
                    gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    print("ENVOY_PINNED_IDENTITY " + json.dumps(identity, sort_keys=True), flush=True)
    os.execv(sys.executable, [sys.executable, "/src/main.py"])


if __name__ == "__main__":
    main()
