# Base versus v5: completed assigned-paper development comparison

**The run did not meet the predeclared useful-answer improvement threshold.**
On eight answerable questions, independent model-assisted review awarded base
2 full passes and v5 1. On four explicitly cued target-setup gaps, base passed
2 and v5 3. Both therefore have four full passes overall, with different mixes.
Do not treat extra gap handling as improved supported answering.

All 24 live episodes completed through real MCP with model-authored Python in
Docker. There were zero execution-error episodes in either arm. The A5000 was
shut down; fresh independent provider reads confirm min/max 0/0, no workers and
no queued or running jobs. No training, paid grader or Nebius calls were made.

## Frozen comparison and interpretation

The [predeclared protocol](2026-10-20-base-v5-reader-comparison.md) fixed the same
focused-v1 assigned V2 prompt, source snapshots, tools, six-action/six-request
cell limits, tokenizer, temperature 0, top-p 1, seed 42 and thinking disabled.
Pairs ran adjacently with balanced order, on one shared A5000. Base used
`envoy-base-only`; the existing adapter used `envoy`. The provider echoed each
requested alias correctly; startup attestation verified the pinned checkpoints.

Base: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`.
V5: `jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5@27b912a863ff914ad45baa03624d8911dc1e17fb`,
`artifacts/full/checkpoint-50`, adapter SHA-256
`7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3`.
Serving was BF16 on NVIDIA RTX A5000, PyTorch 2.13.0+cu130 and vLLM 0.31.0.
The shared worker still had the adapter loaded during base requests; model
selection controlled whether it was applied. This is not an adapter-free process.

The frozen decision required at least two additional supported-answer passes,
no additional condition-error/unsupported-number/execution-failure episodes,
and bounded token and latency regressions. The supported-answer requirement
was not met. No promotion, new training or further paid run follows automatically.

## Claim-level results

| Measurement | Base | V5 |
| --- | ---: | ---: |
| Fully supported, complete answer passes | 2/8 | 1/8 |
| Answerable partials | 5/8 | 7/8 |
| Answerable failures | 1/8 | 0/8 |
| Target-gap passes | 2/4 | 3/4 |
| Target-gap partials / failures | 0 / 2 | 1 / 0 |
| Condition-error episodes | 2 | 0 |
| Unsupported numeric flags | 3 | 1 |
| Execution-error episodes | 0 | 0 |

A fresh reviewer saw all 24 answers as shuffled A/B slots, their cited evidence,
the frozen requirements and cached sources. The key and raw model logs were
withheld. Its completed judgments were committed before decoding the model key.
This is **provisional model-assisted review**, not independent human validation.
No semantic score was inferred from exact-quote validation or packet acceptance.

The strict rubric requires every frozen clause in the answer and support from
its own submitted evidence. Several partials contain correct core facts but omit
setup details, or cite a passage clipped before the supporting sentence. Some
requirements repeat conditions already in the question; the QLoRA rubric also
expects dequantization. Thus 1/8 and 2/8 are strict complete-answer counts, not
claims that every other answer is factually wrong. These completeness choices
limit how broadly the low pass rates can be interpreted.

Two clear source-condition errors occurred in base: it attributed the original
FlashAttention split-K arrangement to FlashAttention-2, and gave the RTX4090
2.7–3.9x TinyChat result as the requested A5000/batch8/8192-token result without
qualification. V5 avoided those condition errors, but sometimes answered the
wrong measurement or omitted a requested condition. For example, its Triton
speedup answer instead discussed throughput utilization.

The QLoRA double-quantization numeric flags in both arms concern correct values
whose submitted excerpts stop before the final result. A possible unstated
arithmetic derivation makes those flags debatable; both answers remain partial
for other frozen omissions. Do not call every unsupported-number flag a
fabricated value, or treat the flag difference as a validated hallucination rate.

| Question | Base verdict (requests) | V5 verdict (requests) |
| --- | --- | --- |
| PagedAttention block mapping | partial (3) | partial (3) |
| vLLM evaluation trace durations | pass (3) | pass (3) |
| vLLM target p95 gap | fail (6) | pass (5) |
| QLoRA dtypes and gradients | partial (3) | partial (3) |
| QLoRA double quantization | partial (4) | partial (3) |
| QLoRA target VRAM gap | pass (5) | pass (5) |
| FlashAttention warp partition | fail (4) | partial (3) |
| FlashAttention Triton comparison | partial (3) | partial (3) |
| FlashAttention target end-to-end gap | pass (5) | partial (5) |
| AWQ latency profile | partial (3) | partial (3) |
| AWQ Figure 9 protocol | pass (3) | partial (3) |
| AWQ target speedup gap | fail (3) | pass (5) |

[Per-question results](../../examples/base-v5-reader-comparison/per-question-results.json)
include action/request counts, timings, rejected submissions, exact quote spans
and every review rationale. Raw packets retain the full per-action observations.

## Mechanics, latency and budget

| Measurement | Base | V5 |
| --- | ---: | ---: |
| Requests | 45 | 44 |
| Reported input tokens | 161,478 | 158,324 |
| Reported output tokens | 2,191 | 2,149 |
| Reported total tokens | 163,669 | 160,473 |
| Median episode seconds | 19.06 | 20.77 |
| Rejected submissions | 3 | 4 |
| Rejected actions / repeated calls | 0 / 0 | 0 / 0 |

Total: **89 unique completed jobs and 324,142 reported tokens**, with usage on
every request. Maximum prompt/completion lengths were 4,729/136 tokens, below
7,168/1,024 caps. No inference request was resubmitted. Six of seven rejected
abstentions were followed by a mechanically valid gap report; base exhausted its
budget on the remaining one. These rejections are protocol feedback, not Python
execution errors, and all additional calls counted against the limits.

V5 used 1.95% fewer reported tokens. The median of paired v5/base episode-time
ratios was 1.181 (18.1% higher). This is whole reader/MCP/Docker episode time,
not a pure model-speed benchmark or billing measurement. The first base episode
also included residual serving-engine startup after the checkpoint identity was
available. Shared startup/order effects and one run limit latency conclusions.

The session guard recorded 628.34 seconds before cleanup, below its 900-second
watchdog. Runtime-start to final lifecycle-file write was about 661.23 seconds,
including cleanup; that is controller wall time, not metered GPU time. There
were no watchdog or cleanup errors. All 89 owned jobs were read back COMPLETED;
provider completed count increased from 29 to 118, with failed/retried/pending
counts zero. The current worker was `r2eanrcirpr4ld` in EU-SE-1.

The fresh Runpod catalog quoted $0.69/hour. The scoped hourly billing read
returned no records for 2026-10-21 01:00–02:00 UTC at query time. That does not
establish a free run or settled charge: **actual per-run spend remains unknown**.
The bounded session was not represented as a hard dollar ceiling.

## Evidence and checks

Approval: `e679c62`. Raw run and sealed blind judgments: `2e134af`.
The original frozen preparation was not rewritten; its readiness validation is
a historical pre-run record. Its plan SHA-256 remains
`99e4ff20a37323b92ff86dc5e4d3e40ca61dd30fce7a70456ad693b9dcd90b88`.
[Live validation](../../examples/base-v5-reader-comparison/live-validation.json),
[derived summary](../../examples/base-v5-reader-comparison/live-summary.json),
[claim review](../../examples/base-v5-reader-comparison/review/claim-review.json), and
[independent execution audit](../../examples/base-v5-reader-comparison/live-execution-review.json)
keep mechanical and semantic findings separate.

The full offline repository suite passed again after shutdown: **1,701 passed,
zero skipped**, with three existing SWIG deprecation warnings (104.62 seconds).
The 150 pinned implementation hashes, all ten protected file hashes and the old
smoke-attempt marker remain unchanged. A separate execution reviewer audited all
24 cells, payloads, aliases, jobs, containers, source spans and shutdown reads,
with no findings. It did not perform the semantic grading.

These are 12 assistant-authored development questions across four papers, with
no established training/pretraining disjointness. The four negative setups are
overtly cued and mention later model/setup combinations. HTML extraction may
omit figures, tables and equations. This measures assigned-paper reading under
one harness; it does not evaluate paper discovery, multi-paper research,
end-to-end ML experiments, a Codex web-search baseline, or general fine-tuning
improvement. No reward was computed and human review remains unperformed.

The MCP user can ask a larger assistant to assign a paper to a small Qwen reader
and inspect the resulting claims, exact quotes, gaps and failure trace. This run
shows that path works within bounds. It also shows why the larger assistant
still needs to check whether each quote supports the complete claimed result.
