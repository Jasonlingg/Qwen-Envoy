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

diagnostic_root="${1:-out/research/qasper-oracle-evidence-v1}"
results_root="${2:-out/research/qasper-oracle-evidence-v1-results}"

if [[ ! -f "$diagnostic_root/benchmark.json" ]]; then
  python scripts/prepare_qasper_oracle_evidence.py --output "$diagnostic_root"
fi
mkdir -p "$results_root"

run_policy() {
  local policy="$1"
  local name="$2"
  python scripts/run_eval.py \
    --policy "$policy" \
    --questions "$diagnostic_root/benchmark.json" \
    --corpus "$diagnostic_root/corpus" \
    --max-steps 5 \
    --seed 42 \
    --no-vector-index \
    --question-only-observation \
    --force-known-paper-read \
    --run-label "$name" \
    --output "$results_root/$name.json"
}

run_policy qwen_base_policy oracle_base
run_policy qwen_sft_policy oracle_sft

echo "Completed QASPER oracle-evidence comparison: $results_root"
