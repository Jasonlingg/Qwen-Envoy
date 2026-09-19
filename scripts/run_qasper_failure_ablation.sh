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

output_root="${1:-out/research/qasper-failure-ablation-v1}"
questions="data/research/qasper_failure_ablation_v1.json"
corpus="out/research/qasper-code-dev-v2/corpus"
prompt_suffix="data/prompts/qasper_failure_recovery_v1.txt"

mkdir -p "$output_root"

run_condition() {
  local name="$1"
  local top_k="$2"
  shift 2
  python scripts/run_eval.py \
    --policy qwen_sft_policy \
    --questions "$questions" \
    --corpus "$corpus" \
    --max-steps 10 \
    --seed 42 \
    --no-vector-index \
    --question-only-observation \
    --search-within-top-k "$top_k" \
    --run-label "$name" \
    --output "$output_root/$name.json" \
    "$@"
}

run_condition control_prompt_top3 3
run_condition recovery_prompt_top3 3 --system-prompt-suffix "$prompt_suffix"
run_condition control_prompt_top8 8
run_condition recovery_prompt_top8 8 --system-prompt-suffix "$prompt_suffix"

echo "Completed QASPER failure ablation: $output_root"
