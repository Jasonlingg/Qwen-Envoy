# Bounded native integration attempt — October 24, 2026

**The two-update gate did not pass.** One real native gradient update ran, then
fresh-checkpoint verification rejected its optimizer counter. No second update,
ten-update pilot, held-out reference review or model-quality comparison ran.
The failed run was not restarted or promoted. No Git commit or Hub upload.

[Machine-readable receipts](../examples/verifiers-port/prime-rl-integration-20261024/summary.json)
are separate from the unchanged [pilot protocol](../examples/reader-rl-pilot-v1/protocol.json).
The user authorized continuing after the two-update / conditional-ten-update
explanation, within the existing $5 estimated / two-hour lifetime proposal.

## Actual run

| Measurement | Observed |
| --- | --- |
| GPU / image | One A100 SXM4 80GB; same pinned image and native stack as the earlier smoke |
| Training questions / completed episodes | 1 / 4 |
| Group rewards | `[0, 1, 1, 0]` |
| Generation requests / reserved tokens | 15 / 62,002 |
| Sampled loss tokens | 697 |
| Sampler/scorer difference | Mean 0.00362124, max 0.20048976; preset 0.03 / 0.25 gates passed |
| Real gradient updates / verified updates | 1 / 0 |
| Native update time / peak GPU memory | 26.7 seconds / 35.3 GiB |
| Native loss / gradient norm | 0.0145 / 0.2478; finite |
| Initial checkpoint AdamW counters | All 504 parameter states at 1 |
| Counters after one actual gradient update | All 504 at 2; expected 1, so verification stopped |
| Paired quality comparison | Not run |

Question: `qasper_train_2b78052314cb730824836ea69bc968df7964b4e4--rl-short-fact`.
Only the 81-question training selection was available to the collector. Four
original episodes remained untrainable; native admission compiled a separate
complete mixed group. Model-authored Python ran in the unchanged local Docker
sandbox through a loopback SSH inference forward. GPU host ran trusted native
inference, scoring, optimization and checkpointing. No LocalREPL fallback.
Rewards are the existing strict finite-answer/provenance contract, not a 50%
semantic-accuracy result. Independent human review remains null.

## Failure mechanism and local repair

PyTorch's `get_state_dict()` initializes an empty AdamW optimizer by taking an
internal zero-gradient step. The warm-start path saved the original V5 model to
native DCP *before* the first real training update. That materialization advanced
its counter from zero to one. The native trainer then performed a real update,
producing counter two at progress step one. The strict verifier correctly refused
to certify that state. The earlier smoke did not establish a zero initial counter.

This was reproduced on the actual GPU stack with a tiny CPU parameter and read
directly from the actual initial/post-update checkpoint metadata. It is a
checkpoint-bootstrap issue, not evidence that Qwen cannot perform the reader task.
It also changes first-step AdamW bias correction, so simply accepting an offset
would hide a meaningful initialization difference.

The local repair materializes only a fresh zero-decay AdamW state, checks that
weights stayed identical and every moment is zero, then normalizes the dummy
counter to zero before saving. It refuses an existing optimizer state. Native
trainer, optimizer, loss, scheduler and checkpoint implementation are unchanged.
The optimizer-step verifier remains strict. A real PyTorch DCP round trip on a
tiny CPU model now preserves step zero; its first real update exactly matches
fresh AdamW and has counter one. This repaired path has **not** been rerun on GPU.
The frozen 29-file reader runtime, prompt, source/references, reward and numerical
experiment protocol were not modified.

## Cleanup and cost

Created Pod `dgnq9xllg9jecw` and 600GB standard volume `sjnuc6xqu3` in US-WA-1
were both deleted, with fresh MCP reads returning 404 for each. Older account
resources were untouched. No owned test resource remains billing.

The independent watchdog refused its early cleanup request because runpodctl 2.14
returns creation time as `2026-10-24 19:58:31.04 +0000 UTC`, while MCP returns
`2026-10-24T19:58:31.04Z`. Its identity check compared strings. Cleanup used an
explicit MCP fallback, not a fabricated automatic-cleanup success. The local
repair normalizes both timestamp formats and has regression coverage while still
rejecting a different creation instant. Live automatic deletion after this repair
has not been established by the local fixture.

The current catalog read quoted $1.79/hour GPU and standard storage $0.07/GB/month.
The conservative time/rate projection through cleanup confirmation is **$0.95**,
below the $5 allowance. Filtered provider billing reads returned no rows yet;
**actual billed spend is pending**, not zero. This was an estimate/time watchdog,
not a provider-enforced dollar cap. Setup included a non-root SSH initialization
repair and a transfer ownership fix; no generation occurred during those repairs.
A dropped SSH launch connection was reattached to the exact owned native parent
without restarting its deadline, resampling or resetting the budget.

Raw logs, ledger, rollout packets, native configuration/scorer/checkpoint receipts,
unverified adapter and cleanup evidence are in
`out/reader-prime-integration-20261024/`. The native step-one adapter is retained
only for diagnosis; it was not approved for evaluation or uploaded. Large full
optimizer checkpoints were deleted with temporary storage after the scalar-counter
proof and receipts were copied. An inference adapter is not an optimizer resume.

## Next gate

A fresh bounded two-update check with the repaired initializer must pass before
any ten-update training or locked evaluation. Do not label the local regression a
successful GPU run. Checkpoint I/O took minutes per phase, much longer than the
26.7-second gradient update; a full-run time/cost projection must include that
before allocating more compute. The attempt stopped under the predeclared failure
rule; no additional paid retry or new update was run during the repair.

## Final local validation

After the initialization and timestamp repairs: **139 native tests passed**
(three expected single-process CPU DCP warnings), and **2,204 full-suite tests
passed / 16 optional skips / three existing SWIG warnings**. Ruff and whitespace
checks pass. The source/review artifact manifest explicitly records that this is
post-failure local repair, links the earlier pre-run manifest SHA and retains the
original numerical protocol hash. Old prepared plans reject the source change;
a future trial needs fresh preparation. No current failed run was rewritten.

The repaired watchdog was also invoked after both resources were already absent:
its real CLI absence reads passed idempotently. This is not proof of repaired
live automatic deletion; the actual deletion above used the MCP fallback.
