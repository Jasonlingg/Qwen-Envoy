# October 24 native Qwen two-update gate: passed

**Two real Qwen gradient updates completed and both full checkpoints passed fresh
native reload verification.** The optimizer counter was **0 → 1 → 2** across all
504 trainable parameter states. Both sampled-token scoring gates, adapter export
comparisons and trainer-startup comparisons passed. This is a training-system
integration result, not evidence of improved answers or a benchmark score.

The user authorized this run with “ok go” after requesting removal of the separate
30-minute work cutoff. The hypothesis and success rule remained two consecutive
native updates with exact state carryover and complete reload verification.
The [prospective timing override](../examples/verifiers-port/prime-next-integration-policy.json)
was prepared before launch. Native APIs retain a 14,400-second finite ceiling,
later than the independent rental guard's maximum 7,200 seconds. The existing
estimated $5 rental guard, 256-request cap and 2,097,152-token reservation cap
remained active. The old numerical protocol and earlier results were not rewritten.

## Actual results

| Cycle | Selected group rewards | Native scoring: tokens / mean delta / max delta | Gradient update | Fresh checkpoint reload |
| --- | --- | --- | --- | --- |
| 1 | [1, 1, 1, 0] | 567 / 0.001937 / 0.118456 | 24.9 seconds | Passed, counter 1 |
| 2 | [1, 0, 0, 0] | 528 / 0.004693 / 0.175669 | 22.0 seconds | Passed, counter 2 |

Scoring thresholds stayed mean ≤0.03 and max ≤0.25. Loss and gradient norm were
finite at both updates; peak GPU memory was **35.34 GiB**. Cycle 2 preparation
preserved the exact optimizer fingerprint, including moments, from the verified
step-1 checkpoint. The fresh reloads matched each exported adapter exactly and
confirmed that each trainer started with the policy used to collect its batch.

The native stopped-result receipt reports `verified_updates: 2` and
`unverified_attempts_may_have_updated_weights: false`. A separate read-only DCP
inspection retained the committed-checkpoint counter evidence: **504 states each
at 0, 1 and 2**. This uses the real Qwen checkpoints, not a tiny-model substitute.

Verified portable adapter SHA-256 values:

- Step 1: `a1984d390d333b64ddcb5753ce6647d578a4b7c1564bf3d8f7975ac48f9f3f03`
- Step 2: `af5db09f6391e35f2ab8782ebb8d5ca5abb75537a50c291fa3af7d0b97479995`

Both adapters were copied locally and rehashed before cloud cleanup. They remain
diagnostic artifacts, with no promotion or Hugging Face upload. The temporary full
optimizer checkpoints were deleted with the volume; a portable adapter alone does
not retain optimizer state for exact training resumption.

## Collection and interpretation

The shared ledger recorded **139 requests / 572,022 reserved tokens**, all returned.
Reservations are not actual generated-token counts or API billing. All **44
original episodes remained untrainable** and infrastructure-healthy. The collector
visited 11 training questions in fixed seed-42 order. Cycle 2 skipped nine healthy
uniform-zero groups before finding its mixed group; only the two complete mixed
groups were copied into native training batches after admission checks.

[Per-question results](../examples/verifiers-port/prime-two-update-20261024/question-group-results.json)
include source question/paper IDs and every group's scores. Rescoring these saved
training traces with the unchanged reward yielded four reviewed answer/support
matches, 22 answers outside accepted wording variants, four missing required
reviewed support spans, one additional citation without reviewed support and 13
without a verified supported terminal packet. These are mechanical reward
categories, **not semantic answer grades**. Unknown wording can receive zero
without proving an answer false. No validation or locked evaluation question was
sampled, and no locked reference contents were reviewed.

The task is the existing training-only adaptation of QASPER, not an original
question benchmark or an official QASPER leaderboard run. Native scorer agreement
validates sampled-probability alignment, not answer correctness. Quote provenance,
reward matches, tensor comparisons and successful updates do not establish better
research answers. No independent human review or model-quality comparison was run.

## Reproducibility, timing and cleanup

The run started from original pinned V5, not an adapter from an earlier attempt.
Base, V5 and native source revisions, tokenizer, prompt, reward, corpus/split
identities and the seed remain pinned in the
[summary](../examples/verifiers-port/prime-two-update-20261024/summary.json) and
[GPU preflight](../examples/verifiers-port/prime-two-update-20261024/gpu-preflight.json).
The setup uses Prime-RL 0.10.0, PyTorch 2.13.0+cu130, vLLM 0.31.0, Verifiers 0.4.0,
Transformers 5.6.2 and the previously pinned renderer/image revisions on one
A100-SXM4-80GB. Four trajectories per group, temperature 1, top-p 1, 1,024 action
tokens, four investigator steps and assigned source windows were unchanged.
Model-authored Python ran only inside the frozen local Docker sandbox.
Engine load RPC/path checks remain distinct from tensor attestation.

Native work took **1,783 seconds (29 minutes 43 seconds)**; owned-resource lifetime,
including setup and read-only evidence export, was **36.7 minutes**. This particular
attempt finished narrowly within 30 minutes of native work; the prior attempt did
not. The timing override prevented reliance on that narrow margin, but this is
not a controlled demonstration that changing the timer caused faster completion.
Only about 47 seconds were gradient updates; cold setup, collection and checkpoint
I/O dominate the workflow and must inform any larger-run projection.

No SSH-read retries were needed in this attempt. The independent watchdog deleted
live Pod `x40apwdixaelxi` and temporary 600GB volume `ie1httce8x`; fresh MCP reads
returned 404 for both. All 44 owned local Docker containers were confirmed absent.
Other user resources were untouched. At the conservative quoted $1.90/hour combined
bound, this attempt cost **$1.16 estimated**, before tax. Filtered provider billing
had no posted rows yet: actual billed spend is pending, not zero.

Validation: **139 focused native tests** and **eight deployment transport/guard
tests passed**. The latest full suite passed **2,204 tests with 16 optional skips**
before this retry on unchanged production sources; documentation and receipts did
not trigger a repeat of that suite. Source-bound plans, frozen source/reference
hashes, exported adapter hashes and user-file preservation were checked. No native
trainer, loss, optimizer, reader or reward implementation was changed for this run.
No commits or uploads were made.

The two-update integration gate is now satisfied. The ten-update pilot and paired
quality evaluation were not started. Their timing/budget feasibility and the sparse
mixed-group signal still need review before spending on a larger run. A successful
integration smoke test does not establish that RL or fine-tuning improved Qwen.

Operational plans, raw traces, full logs, receipt archive and portable adapters:
`out/reader-prime-no-work-cutoff-20261024/`. Small public receipts:
[native result](../examples/verifiers-port/prime-two-update-20261024/native-result.json),
[counter evidence](../examples/verifiers-port/prime-two-update-20261024/optimizer-counters.json),
[summary](../examples/verifiers-port/prime-two-update-20261024/summary.json).
