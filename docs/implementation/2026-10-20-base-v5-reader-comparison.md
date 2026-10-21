# Base versus v5: matched assigned-paper development comparison

Status: prepared locally; **no new model inference, GPU activation or training**.
This compares the existing base and adapter, rather than treating the preceding
prompt experiment as evidence that training helped. The preceding live result
was one pass and one fail, including a temperature/speedup reversal.

## Question and decision

Hypothesis: v5 improves useful supported answering over base Qwen3-8B under the
same assigned-paper MCP harness, beyond increasing refusals.

Expected signal: more completely supported answers on eight answerable cases,
without more condition swaps, fabricated numbers, or execution-failure episodes.
Four explicitly cued target-setup gaps are reported separately. An always-refuse
reader can earn zero of the eight useful-answer passes.

Decision rule, frozen before inference: at least two additional supported-answer
passes out of eight, no additional condition-swap/fabricated-number cases or
execution-failure episodes, no more than 25% higher total reported tokens, and a
median paired v5/base episode-time ratio at most 1.25. Every request must report
complete token usage. Require all 24 cells and attributed, source-visible claim
review. Missing cells, usage or reviews means inconclusive. A favorable signal
only justifies a separate untouched evaluation; it proves no general improvement
and does not authorize training. Report gap performance even if the primary
signal is favorable. Latency includes per-cell Docker/MCP work, excludes shared
GPU cold start; lifecycle timing is separate. GPU billing is not token billing.

## Frozen questions

The source is four live arXiv downloads, then immutable cached HTML extraction:

| Paper | Version | Cases |
| --- | --- | --- |
| PagedAttention/vLLM | 2309.06180v1 | Block mapping; ordinary versus OPT-175B trace duration; target p95 latency gap |
| QLoRA | 2305.14314v1 | Storage/computation/gradients; double-quantization overhead; target peak-VRAM gap |
| FlashAttention-2 | 2307.08691v1 | Forward warp partition; Triton forward/backward speedups; target end-to-end gap |
| AWQ | 2306.00978v6 | Profile task/token/time mapping; Figure 9 benchmark protocol; target speedup gap |

Exact questions: [cases.json](../../examples/base-v5-reader-comparison/cases.json).
The references contain exact cached context offsets and clause-level requirements.
An independent agent reviewed all 12 questions and every cited context against
the four cached sources. It found no factual mapping errors; the AWQ device
question was narrowed to its protocol paragraph. That review is **model-assisted,
not human validation**. Extracted HTML may omit figures, tables and equations;
absence claims concern inspected extracted evidence, not the full literature.
The FlashAttention-2 end-to-end paragraph has an apparent self-comparison typo;
it is explicitly unsuitable for silently inferring a corrected target result.

These are development prompts, fresh relative to the preceding two-case smoke.
They are not certified disjoint from training or pretraining. The gap cases
mention later model/setup combinations and explicitly invite qualification;
they test extrapolation restraint, not neutral open-world question difficulty.

## Matched execution

Both arms use the same pinned Qwen3-8B tokenizer, effective focused-v1 assigned
V2 prompt, tools, six-action limit, temperature 0, top-p 1, seed 42, thinking off,
1,024 output tokens and 7,168 input tokens per request. Each cell has a fresh
reader, conversation and Docker sandbox. The model only sees the question,
controller assignment metadata and subsequent tool observations. References,
case-kind labels and grading requirements are excluded from model inputs.

The base alias is `envoy-base-only`; v5 is `envoy`. The existing pinned bootstrap
already configures those aliases. Request selection and response model identity
must match; wrong-model responses fail closed. Existing transport callers remain
adapter-only by default. Base and adapter ledgers cannot be mixed or reset.
The worker still loads the pinned adapter while serving base requests with it
unapplied: a base-model comparison does not mean an adapter-free process.
[vLLM documents shared base/LoRA serving](https://docs.vllm.ai/en/stable/features/lora/).
Actual live alias behavior still needs verification in the bounded session.

Pairs are adjacent, questions shuffled with seed 42, with six base-first and six
v5-first pairs. Shared GPU/cache effects remain possible. The experiment measures
known-paper reading and the model-plus-harness combination, not discovery,
open-web research, free-running RAG, or performance versus Codex browsing.
Mechanical span checks do not grade semantic support. Model-authored Python runs
only in the network-disabled Docker REPL. There is no reward computation.

## Bounded live proposal

One new session, 12 questions × 2 models = 24 cells, one attempt per cell:

- At most 6 model requests per cell, 144 total, including format retries.
- Reserved input/output ceilings: 1,032,192 / 147,456 tokens total.
- At most one A5000 worker; a 900-second watchdog starts before activation.
- No inference resubmissions after uncertain dispatch. Stop on infrastructure
  failure; retain previous packets, per-action traces and partial results.
- Cleanup cancels owned jobs, removes owned Docker containers, parks endpoint
  min/max at zero, and verifies no workers or queued/running jobs remain.

The earlier two-case live allowance was consumed. A **new plan-bound activation
record** must explicitly approve these larger limits and record a freshly checked
GPU rate before running. No approval record has been created. This requirement
comes from the existing task-by-task approval workflow and the earlier bounded
allowance, not a new training or infrastructure requirement.
At the previous $0.69/hour quote, 15 minutes is about $0.17; that is conditional
arithmetic, not a current price or guaranteed charge. Cold start is within the
watchdog; provider metering and shutdown overhang prevent a hard dollar ceiling.
Actual spend remains unknown until billing evidence is available. No Nebius calls.

## Reproduce preparation and review

The library is at `out/base-v5-comparison-20261020/library`; the prepared bundle
is `examples/base-v5-reader-comparison/prepared`. Snapshots remain local cache
artifacts, with pinned versions and document/corpus hashes. A fresh clone needs
the same extracted snapshots; refetching a version alone does not guarantee
byte-identical extraction. The commit plus source hashes identify the working
tree, including preserved uncommitted configuration; the commit alone does not.

```sh
python scripts/reader_comparison.py check \
  --prepared examples/base-v5-reader-comparison/prepared
```

Later, after explicit bounded activation approval, `reader_comparison.py run`
requires a new output directory and `--authorization-file`. A global exclusive
attempt marker prevents automatic reruns under a different output path. The
run is still not semantically scored when its mechanical checks pass.

For a completed run, use `scripts/review_reader_comparison.py export` with
`--prepared`, `--run-dir` and a new `--review-dir`. Give a reviewer only
`review-packet.json` and the pinned sources; withhold `model-key.json` and logs.
Fill a separate copy of the blank template, attributing human or model-assisted
review accurately. `report` requires that authored `--reviews` file; it rejects
blank grades, inconsistent pass/error labels, changed references, packets,
results or plan identity. Record partial/fail for unsupported or misbound clauses,
even when every quoted character exists. A model-assisted result remains
provisional; independent human review remains unperformed.

## Completed local verification

Transport tests: 19 passed. Final focused tests: **40 passed**, including both
aliases through real MCP and real Docker with scripted model responses, uncertain
dispatch without retry, prior-result retention, incomplete token coverage and
reference/runtime/schedule tampering. Full offline repository suite: **1,701
passed, zero skipped**, with three existing SWIG deprecation warnings. Ruff and
`git diff --check` passed. All ten protected file hashes remain unchanged; the
V1 preamble retains SHA-256
`233adfef3cd72c8b8221e45dcaa58c76809f36bd717d1d342514a7c332c3dd71`.

The independent implementation reviewer reported no remaining blockers after
fixes. Initial prompts range from 2,472 to 2,516 tokens, identical across the
paired arms. The preparation pins 150 implementation files. Full evidence is in
[validation.json](../../examples/base-v5-reader-comparison/validation.json) and
[independent-review.json](../../examples/base-v5-reader-comparison/independent-review.json).
There are **no new live comparison results**. Scripted responses do not show
whether base or v5 is better. The live-attempt marker has not been created.
