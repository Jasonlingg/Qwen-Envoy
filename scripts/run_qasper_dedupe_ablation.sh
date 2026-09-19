#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASE_MODEL_PATH:-}" ]]; then
  echo "Set BASE_MODEL_PATH to the pinned Qwen3-8B base model." >&2
  exit 2
fi
if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  echo "Set CHECKPOINT_PATH to the QASPER SFT adapter." >&2
  exit 2
fi

output_root="${1:-out/research/qasper-search-within-diverse-v1-gpu}"
mkdir -p "$output_root"

python scripts/run_eval.py \
  --policy qwen_sft_policy \
  --questions data/research/qasper_failure_ablation_v1.json \
  --corpus out/research/qasper-code-dev-v2/corpus \
  --max-steps 10 \
  --seed 42 \
  --no-vector-index \
  --question-only-observation \
  --search-within-top-k 6 \
  --search-within-mode ranked_diverse \
  --run-label ranked_diverse_top6 \
  --output "$output_root/ranked_diverse_top6.json"

echo "Completed QASPER ranked-diverse passage ablation: $output_root"
