# Pinned Qwen reader on Runpod

This is the initial A6000 smoke snapshot. The API key is now configured and a
subsequent 24 GB A5000 run completed. See the
[October 20 follow-up](2026-10-20-qwen-24gb-live.md) for the actual worker
failure, host fallback, saved brief and final parked state.

The v5 adapter ran on an RTX A6000 through a new Runpod Serverless endpoint,
`q05pjrvguakieg`. The full native research-to-experiment check stopped during
MCP setup because the local server has no Runpod API key. No host answering
turn or paper-reader request ran in that attempt. This is a deployment smoke,
not a successful paper-research workflow or a model-quality comparison.

## Observations

- Official `runpod-workers/worker-vllm` release `v2.29.0`, Hub release
  `cmuxgi0a3000407jv5igk1ezr`; observed image digest
  `sha256:a218acc3f5032d2bc1b9104b613b5ed0bf34bcd6b7c561396d17a60aa4b7962c`.
- Base `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`.
- Adapter `jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5` at
  `27b912a863ff914ad45baa03624d8911dc1e17fb`, subfolder
  `artifacts/full/checkpoint-50`. Its weights matched SHA-256
  `7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3`.
- Bootstrap attested RTX A6000, PyTorch `2.13.0+cu130`, CUDA runtime `13.0`,
  vLLM `0.31.0`. The worker logged loading the `envoy` LoRA; `/v1/models`
  identified its pinned adapter directory separately from `envoy-base-only`.
- The model-listing job spent **296.300 s queued** during startup and **0.214 s**
  executing. Queue delay includes scheduling and image/model preparation; it
  is not solely weight-load time.
- One real chat request to `envoy` returned `print("qwen_gpu_ready")`, with
  **36 input / 8 output tokens**, **22 ms queue delay** and **5.747 s job
  execution time**. This was a warm, nonstreaming smoke, not a latency benchmark.
- That exact generated line ran in Docker with networking disabled and printed
  `qwen_gpu_ready`. It did not investigate a paper.

The [observed artifact](../../examples/qwen-runpod-v5-smoke/observed.json)
contains the request, response, identity, Docker digest and limitations. Raw
operator receipts and the unsuccessful native setup attempt remain under
ignored `out/qwen-live-20261019/`. Human review is unperformed. There are no
paper claims to grade in this smoke; claim-level support review remains
required when the actual paper workflow runs.

## Serving and budget boundaries

`scripts/runpod_qwen_bootstrap.py` downloads public, pinned snapshots, verifies
the adapter config and weights, then launches the official worker against local
paths. The worker's upstream revision-recovery behavior cannot switch a local
snapshot to HF HEAD. The adapter request name is `envoy`; the separate base name
helps catch accidental base-model calls. No network volume was created.

`src/policies/runpod_queue.py` bridges OpenAI-style chat bodies to Runpod's
asynchronous `/run` and `/status/{id}` API. It rejects redirects, bounds HTTP
bodies, makes one submission, and polls only the returned job ID. An uncertain
submission consumes its reservation. A durable, file-locked session ledger
shares at most 12 requests across native MCP processes and restarts, reserving
7,168 input and 1,024 output tokens each;
the pinned tokenizer checks prompts against the 8,192-token model context.
The server also enforces the 8,192-token context and 120-second job execution
timeout. This is a request/token bound, **not a provider dollar cap**.

`scripts/arxiv_qwen_runpod_mcp.py` explicitly opts into this transport, Qwen v5,
V2 and the full six-tool research surface. Each reader has eight steps and uses
Docker; the optional reasoner is off. Existing source-only registrations and
user edits were preserved. Model inference uses PyTorch on the remote GPU;
the local MCP coordinates requests and Docker execution.

The live catalog listed the AMPERE_48 pool at **$1.22 per active GPU hour**.
No billing records were available at the observation time; actual spend remains
unknown. The endpoint was created with min 0/max 1 and a 60-second idle timeout.
The worker inventory briefly listed three worker records despite max 1, while
health reported one running worker. Do not infer a hard billing/concurrency cap
from that setting. While credentials are missing, the endpoint is explicitly
**parked at min 0/max 0**. It must be re-enabled at max 1 before new jobs.
Read-back confirmed **zero workers**, zero queued jobs and zero in-progress jobs
after parking. The pre-existing stopped pods and network volume were untouched.

Verification: 19 focused tests passed, followed by the full suite with **1,500
tests passing** (three existing SWIG deprecation warnings). All ten recorded
protected-file hashes still match. A subsequent native-runner change tightened
the MCP inventory and host-instruction checks; compilation, 16 focused tests
and a final full-suite run (1,500 passed in 53.91 seconds) passed afterward.

## Finish the native check

The Runpod connector's OAuth login is available to the assistant's Runpod
tools, not to the Python research server. Configure a Runpod API key locally in
`~/.runpod/config.toml` (`apikey`), or provide `RUNPOD_API_KEY` to the research
MCP process. Do not commit or paste the key. Re-enable this endpoint's max
workers to 1, leaving min 0. Retain the endpoint and its template for future
on-demand use; no GPU launch is needed merely to configure the local server.

Then run a new, explicitly budgeted session (the output directory must be new):

```sh
python scripts/run_qwen_mcp_live.py \
  --endpoint-id q05pjrvguakieg \
  --sandbox-image sha256:79bd8bf84bd4b7f2a32a0596cb1fd90164716c1cfca879f78829411193646382 \
  --output out/qwen-live-native-retry
```

This fresh native Codex session must open `2211.17192v2`, call `investigate`
once, inspect context, save an experiment brief and retrieve it. It explicitly
tests required worker use; it is separate from the earlier free-choice A/B
pilot. A completed host answer alone is insufficient: inspect the native tool
trace, Qwen audit, paper evidence and saved record. Review whether major claims
follow from passages; exact span checks establish provenance only. Report
reader failures or missing evidence without turning them into supported claims.

GPU memory holds one shared model plus KV cache for active sequences. Max
sequences is two, but this smoke did not measure concurrent readers, throughput,
or memory under a realistic workload. Base-vs-trained and host-only-vs-MCP quality
comparisons still require separate, matched evaluations.
