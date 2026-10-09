# Qwen prefix-cache experiment

**Prepared, not measured.** This is a small inference-systems exercise using the
existing v5 checkpoint and multi-step EnvoyBench environment. It changes no
weights, rewards, tools, prompts, or quality grades. It is not another training
run or a held-out model-improvement result.

## What we are changing

Each tool turn sends the system prompt and growing conversation back to vLLM.
Automatic prefix caching can reuse KV entries for identical token prefixes from
earlier requests. Ordinary KV caching during autoregressive generation remains
enabled in both arms. This experiment concerns reuse **between requests**.

Qwen3-8B already uses grouped-query attention: 32 query heads share 8 KV heads.
Changing those head counts is a weight/architecture change, not an inference
switch for our adapter. We start with a serving change that preserves the model.

- **Hypothesis:** prefix reuse reduces elapsed inference-request time on tool turns.
- **Expected signal:** cached prompt tokens in the ON arm, lower paired request
  time, and unchanged actions, answers, evidence, and generated-token counts.
- **Decision rule:** at least a 15% median paired reduction in summed request
  time, with complete telemetry, matching workloads, and no observed failures.
  A failure to meet this rule is not proof caching never helps. Three development
  questions and two repetitions are an initial diagnostic, not a general speed claim.

Full HTTP request time includes transport and generation. It is **not** time to
first token or GPU kernel time. Episode time additionally includes the tool
sandbox. Neither timing metric establishes answer quality.

## Frozen workload

The plan selects the first three IDs in the frozen development split without
selecting on their outputs. It records the dataset, corpus, question IDs,
checkpoint revisions and adapter hash, source-run hash, decoding, and relevant
implementation hashes. Source traces supply configuration only; these are new
live trajectories, not exact historical request replays.

The four phases run sequentially: **OFF-1 → ON-1 → ON-2 → OFF-2**. Each phase
uses a fresh server process, the same GPU/runtime, one worker, an 8192-token
context limit, and the same question order. Each question permits at most 15
program rounds and 1024 output tokens per request. There are 12 measured episodes
and four small warmup requests. No concurrent clients may use the server.

The local harness seed is 42. Remote `send_seed=false` is preserved from the
source configuration. Greedy decoding does not guarantee bitwise determinism;
any observed output divergence blocks the clean paired-speed claim.

## Prepare offline

```bash
python -m benchmarks.envoybench.inference_ablation --prepare \
  --source-run out/envoybench/provisional-full-20260930 \
  --serving-hardware 'NVIDIA A40 48GB (single GPU; prospective run)' \
  --serving-runtime 'vLLM 0.30.0; PyTorch 2.13.0+cu130' \
  --output out/envoybench/cache-ablation-prepared-20261001
```

Preparation does not read endpoint credentials, start compute, or send model
requests. Existing output directories are never overwritten. If implementation,
dataset, or runtime changes before execution, create a new plan before making
measured requests. Actual runtime and hardware must match the declaration.

## Execute on an existing GPU server

This section incurs compute cost. Provisioning is separate. Use the same GPU
allocation for all phases, record its ID, `nvidia-smi` output, installed versions,
and server startup logs alongside the results. The launcher requires installed
vLLM 0.30.0 for this experiment and verifies the pinned adapter bytes; it does
not automatically upgrade an experiment server.

Start a fresh process for each phase, substituting `on` for the two ON phases:

```bash
PREFIX_CACHING=off MAX_NUM_SEQS=1 MAX_MODEL_LEN=8192 \
  bash scripts/serve_vllm.sh
```

Experiment mode binds to loopback and enables the local cache-reset API and
cached-token usage details. Reach it through an SSH tunnel when the harness is
local. Wait for server readiness; do not start timing model downloads or startup
compilation as inference. Keep these setup costs in the total compute accounting.

On the GPU server, perform the **same disjoint warmup once** and then require a
successful prefix-cache reset before each phase. The request is stored in
`plan.json` under `warmup.request`. For example, with the plan copied locally:

```python
import json
from pathlib import Path
from urllib.request import Request, urlopen

plan = json.loads(Path("out/envoybench/cache-ablation-prepared-20261001/plan.json").read_text())
origin = "http://127.0.0.1:8000"
warmup = Request(origin + "/v1/chat/completions",
                 data=json.dumps(plan["warmup"]["request"]).encode(),
                 headers={"Content-Type": "application/json"})
with urlopen(warmup, timeout=180) as response:
    result = json.load(response)
    assert result.get("choices"), result
with urlopen(Request(origin + "/reset_prefix_cache", method="POST"), timeout=30) as response:
    assert json.load(response).get("success") is True
```

An HTTP 200 by itself is insufficient: reset can return `success:false`. Do not
restart between warmup and measurement, since that also loses kernel warmup.
Prefix entries may accumulate across the three questions within each phase;
this protocol measures a sequential workload, not isolated cold requests.

Use a single-model config for the pinned v5 identity/decoding in the plan, with
the same hardware/runtime and `serving_context_window_tokens: 8192`. The existing
runner resolves `endpoint_env`; credentials do not belong in the plan or results.
After completing the setup declarations, run one phase:

```bash
export ENVOY_QWEN_ENDPOINT=http://127.0.0.1:8000/v1
python -m benchmarks.envoybench.inference_ablation --execute \
  --prepared out/envoybench/cache-ablation-prepared-20261001 \
  --source-run out/envoybench/provisional-full-20260930 \
  --models out/envoybench/cache-ablation-prepared-20261001/models.json \
  --phase off-1 --server-prefix-caching off \
  --server-process-id '<pod-id>/<process-pid>/<start-time>' \
  --fresh-server-process --warmup-cache-reset-complete \
  --output out/envoybench/cache-ablation-20261001/off-1
```

Repeat with a fresh server and verified warmup/reset for `on-1`, `on-2`, and
`off-2`, in that order. Substitute the actual unique process identifier, phase,
cache flag, and output path. The cache setting and preparation declarations are
**operator supplied, not endpoint attestation**. Preserve logs as corroboration.

## Compare and interpret

```bash
python -m benchmarks.envoybench.inference_ablation --compare \
  --prepared out/envoybench/cache-ablation-prepared-20261001 \
  --phase-dir out/envoybench/cache-ablation-20261001/off-1 \
  --phase-dir out/envoybench/cache-ablation-20261001/on-1 \
  --phase-dir out/envoybench/cache-ablation-20261001/on-2 \
  --phase-dir out/envoybench/cache-ablation-20261001/off-2 \
  --output out/envoybench/cache-ablation-20261001/comparison.json
```

The comparator checks the complete paired matrix and matching configuration,
reports missing telemetry and failures, and withholds a clean timing claim when
workloads differ. Inspect per-question results and cache use, not just the median.
The report always sets `quality_promotion: false`. A useful result would establish
an inference improvement for this workload; it would not establish a better model.

Stop the experiment allocation after copying its outputs. Include startup,
warmup, and idle time in spending; request durations alone are not a cloud bill.

## Learning objective and references

The engineering contribution is a controlled experiment, useful instrumentation,
and analysis of where runtime goes. vLLM supplies the caching implementation.
To practice the interview-level math separately, implement tiny full-sequence
and cached GQA reference functions on CPU and check their numerical equivalence
before trying any custom attention implementation with pretrained weights.

- [vLLM automatic prefix caching](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/automatic_prefix_caching.md)
- [vLLM 0.30.0 server flags](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/entrypoints/launchers/cli_args.py)
- [vLLM 0.30.0 cache reset API](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/entrypoints/serve/dev/cache/api_router.py)
- [Pinned Qwen3-8B configuration](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json)
