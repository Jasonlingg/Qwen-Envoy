#!/usr/bin/env bash
set -euo pipefail

: "${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to the Qwen3-8B base model}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the QASPER v5 SFT adapter}"

OUT_DIR="${OUT_DIR:-out/research/qasper-query-planning-v1-gpu}"
mkdir -p "$OUT_DIR"

python scripts/run_eval.py \
  --policy qwen_sft_policy \
  --questions data/research/qasper_failure_ablation_v1.json \
  --corpus out/research/qasper-code-dev-v2/corpus \
  --max-steps 10 \
  --seed 42 \
  --require-evidence \
  --no-vector-index \
  --question-only-observation \
  --system-prompt-suffix data/prompts/qasper_fulltext_recovery_v1.txt \
  --search-within-top-k 3 \
  --search-within-mode raw \
  --run-label fulltext_recovery_top3 \
  --output "$OUT_DIR/fulltext_recovery_top3.json"
