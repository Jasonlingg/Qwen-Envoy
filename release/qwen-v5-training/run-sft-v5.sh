#!/usr/bin/env bash
set -uo pipefail

source /workspace/envoy-sft-venv/bin/activate
cd /workspace/envoy-sft-v5

export HF_HOME=/workspace/envoy-hf-cache
unset HF_HUB_ENABLE_HF_TRANSFER
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_root=/workspace/qwen3-qasper-v5-sft-20260919
smoke_out="$run_root/smoke"
full_out="$run_root/full"
mkdir -p "$run_root"

date -u +%Y-%m-%dT%H:%M:%SZ > "$run_root/started-at.txt"
python - <<'PY' > "$run_root/environment.txt"
import platform
import torch
import transformers
import trl
import peft
import bitsandbytes

print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name())
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
print("bitsandbytes", bitsandbytes.__version__)
PY

python scripts/train_sft.py train \
  --data data/sft/qasper-v5/train.jsonl \
  --val-data data/sft/qasper-v5/val.jsonl \
  --corpus-manifest data/sft/qasper-v5/manifest.json \
  --out "$smoke_out" \
  --epochs 1 \
  --max-steps 1 \
  --lr 2e-4 \
  --batch-size 1 \
  --grad-accum 4 \
  --max-seq-len 8192 \
  --save-steps 1 \
  --save-total-limit 2 \
  --seed 42 \
  > "$run_root/smoke.log" 2>&1
smoke_code=$?
printf '%s\n' "$smoke_code" > "$run_root/smoke.exit"
if [[ "$smoke_code" -ne 0 ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$run_root/failed-at.txt"
  exit "$smoke_code"
fi

python scripts/train_sft.py train \
  --data data/sft/qasper-v5/train.jsonl \
  --val-data data/sft/qasper-v5/val.jsonl \
  --corpus-manifest data/sft/qasper-v5/manifest.json \
  --out "$full_out" \
  --epochs 4 \
  --lr 2e-4 \
  --batch-size 1 \
  --grad-accum 4 \
  --max-seq-len 8192 \
  --save-steps 10 \
  --save-total-limit 12 \
  --seed 42 \
  > "$run_root/train.log" 2>&1
train_code=$?
printf '%s\n' "$train_code" > "$run_root/train.exit"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_root/finished-at.txt"
exit "$train_code"
