#!/usr/bin/env bash
# Serve the QASPER SFT adapter over an OpenAI-compatible endpoint with vLLM.
#
# Run this ON the GPU box. The client (viewer.py / run_eval.py --policy
# openai_compatible) then talks to it over HTTP and needs no GPU itself.
#
# One vLLM process holds ONE copy of the base weights and serves many concurrent
# requests via continuous batching. That is the only path that scales to several
# Envoy agents at once -- loading qwen_sft_policy N times instead costs N copies
# of the weights in VRAM and still serialises on the GPU.
#
# The adapter is a rank-4 LoRA on Qwen3-8B, so the base model is downloaded too.
# vLLM applies the adapter by name at request time: the client must send
# ENVOY_MODEL_ID=<LORA_NAME>, not the base model id.
#
# Qwen3's stock chat template is correct here. The training-time patch in
# src/policies/qwen3_chat_template.py only adds {% generation %} masking markers
# and renders byte-identical text, so no --chat-template override is needed.
# The client chooses thinking mode per request via chat_template_kwargs. Set
# QWEN_REASONING_PARSER=qwen3 to expose generated reasoning separately in the
# OpenAI-compatible response; the default keeps the established server path.
set -euo pipefail

PINNED_BASE_MODEL="Qwen/Qwen3-8B"
PINNED_BASE_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
PINNED_ADAPTER_REPO="jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5"
PINNED_ADAPTER_SUBFOLDER="artifacts/full/checkpoint-50"
PINNED_ADAPTER_REVISION="27b912a863ff914ad45baa03624d8911dc1e17fb"
PINNED_ADAPTER_SHA256="7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3"
BASE_MODEL="${BASE_MODEL:-$PINNED_BASE_MODEL}"
BASE_REVISION="${BASE_REVISION:-$PINNED_BASE_REVISION}"
ADAPTER_REPO="${ADAPTER_REPO:-$PINNED_ADAPTER_REPO}"
ADAPTER_SUBFOLDER="${ADAPTER_SUBFOLDER:-$PINNED_ADAPTER_SUBFOLDER}"
ADAPTER_REVISION="${ADAPTER_REVISION:-$PINNED_ADAPTER_REVISION}"
ADAPTER_DIR="${ADAPTER_DIR:-/workspace/adapters/qasper-sft-v5-checkpoint-50}"
LORA_NAME="${LORA_NAME:-envoy}"
PORT="${PORT:-8000}"
# Adapter is r=4; vLLM's --max-lora-rank takes 8/16/32/64, so 8 is the floor.
MAX_LORA_RANK="${MAX_LORA_RANK:-8}"
# Cap concurrency so a wide agent fan-out queues instead of OOMing the KV cache.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Opt-in controlled inference experiment. Unset leaves the established launcher
# behavior intact. Both arms pin the same runtime and bind the dev API locally.
PREFIX_CACHING="${PREFIX_CACHING:-default}"
QWEN_REASONING_PARSER="${QWEN_REASONING_PARSER:-off}"
EXPERIMENT_ARGS=()
SERVE_HOST="0.0.0.0"
case "$PREFIX_CACHING" in
  default) ;;
  on|off)
    INSTALLED_VLLM=$(python3 -c 'from importlib.metadata import version; print(version("vllm"))')
    if [[ "$INSTALLED_VLLM" != "0.30.0" ]]; then
      echo "Cache comparison requires installed vLLM 0.30.0; no automatic upgrade performed." >&2
      exit 2
    fi
    if [[ "$MAX_MODEL_LEN" != "8192" || "$MAX_NUM_SEQS" != "1" ]]; then
      echo "Cache comparison requires MAX_MODEL_LEN=8192 and MAX_NUM_SEQS=1." >&2
      exit 2
    fi
    if [[ "$PREFIX_CACHING" == "on" ]]; then
      EXPERIMENT_ARGS+=(--enable-prefix-caching)
    else
      EXPERIMENT_ARGS+=(--no-enable-prefix-caching)
    fi
    EXPERIMENT_ARGS+=(--enable-prompt-tokens-details)
    # Enables local cache-reset administration for the declared warmup protocol.
    export VLLM_SERVER_DEV_MODE=1
    SERVE_HOST="127.0.0.1"
    ;;
  *) echo "PREFIX_CACHING must be default, on, or off." >&2; exit 2 ;;
esac

case "$QWEN_REASONING_PARSER" in
  off) ;;
  qwen3) EXPERIMENT_ARGS+=(--reasoning-parser qwen3) ;;
  *) echo "QWEN_REASONING_PARSER must be off or qwen3." >&2; exit 2 ;;
esac

if [[ "$BASE_MODEL" != "$PINNED_BASE_MODEL" || "$BASE_REVISION" != "$PINNED_BASE_REVISION" ||
      "$ADAPTER_REPO" != "$PINNED_ADAPTER_REPO" ||
      "$ADAPTER_SUBFOLDER" != "$PINNED_ADAPTER_SUBFOLDER" ||
      "$ADAPTER_REVISION" != "$PINNED_ADAPTER_REVISION" ]]; then
  echo "This script serves only the pinned Qwen3-8B v5 checkpoint." >&2
  exit 1
fi

if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ||
      ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
  echo "Downloading $ADAPTER_REPO@$ADAPTER_REVISION/$ADAPTER_SUBFOLDER -> $ADAPTER_DIR" >&2
  mkdir -p "$ADAPTER_DIR"
  # --include keeps this to the ~30MB adapter instead of every checkpoint in the repo.
  python3 - "$ADAPTER_REPO" "$ADAPTER_REVISION" "$ADAPTER_SUBFOLDER" "$ADAPTER_DIR" <<'PY'
import shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo, revision, subfolder, target = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
local = snapshot_download(repo_id=repo, revision=revision,
                          allow_patterns=[f"{subfolder}/*"])
source = Path(local) / subfolder
if not (source / "adapter_config.json").is_file() or not (
        source / "adapter_model.safetensors").is_file():
    raise SystemExit(f"{repo}@{revision}/{subfolder} is missing adapter files")
for item in source.iterdir():
    if item.is_file():
        shutil.copy2(item, target / item.name)
print(f"adapter files: {sorted(p.name for p in target.iterdir())}")
PY
fi

# Check local overrides and cached files against the published v5 weight hash.
# The revision pins downloads; the hash pins the bytes actually passed to vLLM.
python3 "$SCRIPT_DIR/verify_v5_adapter.py" \
  --adapter-dir "$ADAPTER_DIR" \
  --base-model "$BASE_MODEL" \
  --expected-sha256 "$PINNED_ADAPTER_SHA256"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not found; installing" >&2
  if [[ "$QWEN_REASONING_PARSER" == "qwen3" ]]; then
    pip install --quiet "vllm==0.30.0"
  else
    pip install --quiet "vllm>=0.8"
  fi
fi

if [[ "$QWEN_REASONING_PARSER" == "qwen3" ]]; then
  INSTALLED_VLLM=$(python3 -c 'from importlib.metadata import version; print(version("vllm"))')
  if [[ "$INSTALLED_VLLM" != "0.30.0" ]]; then
    echo "Qwen3 reasoning comparison requires installed vLLM 0.30.0; no automatic upgrade performed." >&2
    exit 2
  fi
fi

echo "Serving $BASE_MODEL@$BASE_REVISION + $ADAPTER_REPO@$ADAPTER_REVISION/$ADAPTER_SUBFOLDER (sha256=$PINNED_ADAPTER_SHA256) as LoRA '$LORA_NAME' on port $PORT" >&2
echo "Client: ENVOY_MODEL_ENDPOINT=http://<host>:$PORT/v1 ENVOY_MODEL_ID=$LORA_NAME" >&2
echo "Prefix caching: $PREFIX_CACHING; host: $SERVE_HOST; record startup logs for comparison." >&2
echo "Qwen reasoning parser: $QWEN_REASONING_PARSER" >&2

exec vllm serve "$BASE_MODEL" \
  --revision "$BASE_REVISION" \
  --served-model-name "$BASE_MODEL" \
  --enable-lora \
  --lora-modules "$LORA_NAME=$ADAPTER_DIR" \
  --max-lora-rank "$MAX_LORA_RANK" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --host "$SERVE_HOST" \
  "${EXPERIMENT_ARGS[@]}" \
  --port "$PORT"
