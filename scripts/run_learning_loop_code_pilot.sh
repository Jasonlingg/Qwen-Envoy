#!/usr/bin/env bash
# Run the frozen learning-task development comparison on a GPU machine.
set -euo pipefail

cd "$(dirname "$0")/.."

pilot_python="${RESEARCH_PYTHON:-python3}"
pilot_snapshot="${RESEARCH_SNAPSHOT:-out/research/starter-2026-09-12}"
pilot_questions="data/research/learning_loop_code_pilot_v1.json"
pilot_output="${RESEARCH_OUTPUT:-out/research/learning-loop-code-pilot-$(date -u +%Y%m%dT%H%M%SZ)}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-8B}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-jasonlingg/qwen-envoy-qwen3-8b-qasper-targeted-sft-v1}"

if [[ -e "$pilot_output" ]]; then
  echo "Output already exists: $pilot_output" >&2
  exit 1
fi
if [[ ! -f "$pilot_snapshot/manifest.json" ]]; then
  echo "Frozen snapshot missing: $pilot_snapshot" >&2
  exit 1
fi

"$pilot_python" scripts/research_benchmark.py validate \
  --benchmark "$pilot_questions" --snapshot "$pilot_snapshot"
mkdir -p "$pilot_output"
"$pilot_python" scripts/validate_learning_loop_pilot.py \
  --benchmark "$pilot_questions" --snapshot "$pilot_snapshot" \
  --review-md "$pilot_output/source-check.md"
"$pilot_python" scripts/prepare_ai_paper_id_diagnostics.py \
  --questions "$pilot_questions" --corpus "$pilot_snapshot/corpus" \
  --output "$pilot_output/candidate-ids"

candidate_questions="$pilot_output/candidate-ids/retrieved_ids.json"
shared=(
  --questions "$candidate_questions"
  --corpus "$pilot_snapshot/corpus"
  --max-steps 15
  --seed 42
  --require-evidence
  --no-vector-index
  --evidence-verifier
  --verifier-feedback-budget 4
  --escalate-after-verifier-failure
)

"$pilot_python" scripts/run_eval.py \
  --policy qwen_base_policy "${shared[@]}" \
  --run-label base --output "$pilot_output/base.json"
"$pilot_python" scripts/run_eval.py \
  --policy qwen_sft_policy "${shared[@]}" \
  --run-label targeted-sft --output "$pilot_output/targeted-sft.json"

"$pilot_python" scripts/review_code_exec_pilot.py prepare \
  --questions "$pilot_questions" --corpus "$pilot_snapshot/corpus" \
  --run "$pilot_output/base.json" --run "$pilot_output/targeted-sft.json" \
  --output "$pilot_output/blind-review" --seed 42

echo "Comparison complete: $pilot_output"
echo "Review blind-review/review.md and source-check.md before scoring."
