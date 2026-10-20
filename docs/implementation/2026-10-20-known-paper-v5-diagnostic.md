# v5 known-paper extraction diagnostic

V5 answered one narrow factual question correctly in both fresh attempts under
its original training prompt and citation-only protocol. Both attempts inspected
the source before submitting; neither recorded an execution error. Each also
executed the identical search twice. This establishes a working narrow example,
not reliable production tool use or a fine-tuning improvement.

The question was: **What batch size was used for the walltime measurements?**
The initial message supplied the title *Fast Inference from Transformers via
Speculative Decoding*, paper version `2211.17192v2`, and document ID
`arxiv_2211_17192v2`. The correct answer was **1**.

| Attempt | Model requests | Source actions | Execution errors | Answer | Elapsed |
|---|---:|---:|---:|---|---:|
| 1 | 3 | 2 | 0 | Batch size 1, correct paper cited | 111.50 s |
| 2 | 3 | 2 | 0 | Batch size 1, correct paper cited | 14.00 s |

Both trajectories were identical:

```python
print(search_within("arxiv_2211_17192v2", "batch size walltime measurements"))
# The next action repeats this exact call.
```

The third action submitted the answer and paper citation. The repeated search
was executed under the legacy diagnostic protocol, not rejected by the product
V2 controller. The first request queued for 89.80 seconds during cold startup;
the two elapsed times should not be treated as comparable warm latency samples.

## What was fixed before the run

All 35 saved v5 training conversations share one system prompt, and none of
their final targets contains the newer `EVIDENCE` suffix. This diagnostic pins
that exact prompt, SHA-256
`7b5ae820827750b2dd93f5a2d1c67bacf2c1d13c3def62b17aec6d0e433ddc8a`,
and the original V1 Python preamble, SHA-256
`233adfef3cd72c8b8221e45dcaa58c76809f36bd717d1d342514a7c332c3dd71`.
It supplies the paper ID and accepts the original citation-only submission.
The adapter is the same pinned v5 checkpoint used in the earlier live test.

The prepared manifest recorded the hypothesis, expected signal, decision rule,
source/corpus hashes, decoding, runtime limits and implementation hashes before
activation. A separate frozen reference answer was never sent to the model.
The source was already cached; the model still had to write Python, inspect
tool output and submit an answer. Python ran only in the pinned Docker sandbox,
with no host execution fallback. No training reward was computed.

This is an isolated diagnostic. Product V2's discovery, exact-span evidence and
duplicate-code guards remain unchanged. It is not an MCP end-to-end pass.

## Evidence and interpretation

The [prepared manifest](../../examples/qwen-known-paper-v5/prepared-manifest.json)
preserves the live implementation at `5923d30`. The
[recorded trials](../../examples/qwen-known-paper-v5/trials.json) retain every
action and observation, including the redundant searches and empty
model-supplied evidence lists. The
[observed artifact](../../examples/qwen-known-paper-v5/observed.json) records
usage, timings, job IDs, hardware identity and final infrastructure state.
The raw request ledger remains local at
`out/qwen-known-paper-20261020/requests.jsonl`.

A separate small [assistant claim review](../../examples/qwen-known-paper-v5/claim-review.json)
checks the answer against the retrieved passage and frozen reference context.
The passage explicitly says batch size 1 on one TPU-v4. The review reconstructs
the source window from the recorded observation; these are not model-produced
`EVIDENCE` spans. Exact matching checks provenance; the assistant's contextual
review assesses support. Independent human review was not performed. These
measurements do not establish an optimal batch size for another workload.

The [earlier broad MCP test](2026-10-20-qwen-24gb-live.md) failed to produce
evidence. This diagnostic changed the question, prompt, controller and output
requirements together. It therefore cannot identify which change explains the
success. Two same-seed greedy repeats on one familiar development question are
not independent samples or a reliability estimate. There was no base-Qwen
comparison, Nemotron run, fixed-action replay or additional training. The work
does not prove fine-tuned Qwen improved.

## Limits, cost and verification

The shared durable budget allowed at most 12 requests, reserving at most 86,016
input and 12,288 output tokens; each attempt allowed six actions and each request
at most 1,024 output tokens. Actual usage was **6 requests, 5,022 input and 190
output tokens**. Temperature was 0, top-p 1, seed 42, thinking disabled. The
900-second local watchdog did not fire. These are request/token/time limits,
not a provider dollar cap. No Nebius requests were made.

Runpod startup verified the pinned adapter hash on an NVIDIA RTX A5000, 24 GB,
with PyTorch 2.13.0+cu130 and vLLM 0.31.0. The catalog rate was $0.69 per active
GPU hour, including loading and idle time. The final read confirmed endpoint
min 0/max 0, zero active workers and zero queued/in-progress jobs. Runpod's
UTC-day endpoint aggregate was $0.1124037, including earlier runs; this is not
the diagnostic's isolated bill. Actual diagnostic-only spend remains unknown.

Independent code/evidence review found one interruption cleanup defect: Ctrl-C
could leave the separately started worker running after its parent's watchdog
was canceled. `689114c` now kills and reaps the worker before disarming the
watchdog. A regression exercises a real separate-session process. This fix was
made after the completed live run; the original hashes remain intact. Killing
the local client alone does not guarantee cancellation of an already submitted
remote job; endpoint parking was separately verified. No paid rerun was made
for the cleanup change.

Focused verification passed 22 tests; the full suite passed **1,511 tests in
53.26 seconds**, with three existing SWIG deprecation warnings. All ten
protected-file hashes remain unchanged. Independent follow-up review found the
cleanup defect resolved with no further findings.

The next useful experiment would keep this same question and paper fixed and
add discovery, span formatting and recovery requirements one at a time. That
would test where transfer breaks. No further paid experiment is included here.
