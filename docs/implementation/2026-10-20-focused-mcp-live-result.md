# Focused MCP live result — October 20, 2026

The real candidate end-to-end test completed, and the A5000 endpoint is verified
parked again. The MCP/reader/Docker/verifier path worked. Source-visible answer
review found **one pass and one fail**, so the frozen two-question quality gate
was not met. There was no automatic replacement or additional paid run.

## Authorized attempt and actual execution

The user said “ok lets go test it go” after reviewing the concrete preparation
and limits. The [plan-bound approval](../../examples/assigned-reader-focused-live-approval.json)
was committed as `1222b36` before activation. It authorizes one attempt against
plan SHA-256
`e8690bda6fc7ddbfbe48458205af724800e0ddf4017d12ba53e1032ccf398b15`.
The previous attempt marker was archived byte-for-byte; all earlier archives and
published artifacts remain intact. The new allowance is now exhausted and its
exclusive marker remains in place.

The frozen hypothesis was that `focused-v1` guidance would help v5 interpret
displayed evidence and report an unresolved target-setup gap. Both cases had to
meet their source/claim requirements and execution limits; quote matching alone
could not pass a case. The same paper, questions, model/checkpoint, decoding,
Docker image and request/token/watchdog limits were retained.

Fresh preflight verified the 32 implementation hashes, initial prompt tokens
2,462/2,488, a parked/empty endpoint and the $0.69/hour A5000 serverless quote.
The pinned identity log confirmed Qwen3-8B base revision
`b968826d9c46dd6066d109eabc6255188de91218` and v5 adapter revision
`27b912a863ff914ad45baa03624d8911dc1e17fb`, subfolder
`artifacts/full/checkpoint-50`, adapter SHA-256
`7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3`.
It reports RTX A5000, PyTorch 2.13.0+cu130 and vLLM 0.31.0. Temperature 0,
top-p 1, seed 42 and disabled thinking were retained. Startup log timeouts were
handled within the existing bounded reads; inference began only after identity
verification. This worker was in EU-SE-1; the previous saved run was in CA-MTL-1.

Both questions went through the actual in-process MCP `investigate` tool and
bounded remote Qwen transport. Model Python ran only in the pinned Docker
sandbox. Host-visible answers, quotes, valid gap reports, identity and request
counts were checked against the raw packets. This covers a cached-paper MCP
workflow; desktop/stdio process launch, fresh discovery and reasoner synthesis
were excluded as frozen in the preparation.

## Results and source review

| Question | Model requests | Reader time | Saved baseline review | Candidate review |
| --- | ---: | ---: | --- | --- |
| Batch size used for walltime measurements | 3 | 69.44 s | Fail | **Pass** |
| Speedup established for A5000 at batch 8 | 3 | 23.49 s | Partial, not pass | **Fail** |

The first case searched for `batch size walltime`, inspected source characters
21,000–22,000 and correctly answered **batch size 1 on a single TPU-v4**. The cited
text supports both conditions and the MCP host received the same answer/quote.
This corrects the observed false missing-information claim on this familiar
development question.

The second case searched for the target setup, inspected characters
21,400–22,400 and submitted a related TPU result. The source reports **2.6× at
temperature 1 (standard sampling)** and **3.4× at temperature 0 (argmax)** for
translation. Qwen reversed that association, assigning 2.6× to argmax and 3.4× to
standard sampling. Its quote was exact, but its interpretation was wrong.

The answer retained TPU-v4/batch-1 context and did not invent a numeric speedup
for A5000/batch 8. Its “no evidence is provided” sentence is only defensible as
not established in the inspected extracted context; it cannot prove absence
from the full paper or literature. It also omitted the required explicit
`report_gap` and useful next query. The candidate's mechanical gap requirement
applies to literal abstention; this supported-looking submission did not trigger
that gate. The limitation is retained rather than hidden by a successful span
check. The MCP host faithfully received the erroneous candidate answer.

There were **zero Python errors, action/submission/gap rejections, repeated-call
rejections or budget breaches**, and both MCP/packet checks passed. Each case
used a search, a focused passage and a submission. Neither emitted a gap report.
The separate [claim/action review](../../examples/assigned-reader-focused-live/claim-review.json)
records each action and source-supported/contradicted subclaim. Original raw
packets, MCP responses, request ledgers and mechanical results are unchanged.

Root and independent reviews agree on one pass/one fail. These are
**model-assisted source-visible judgments, not human review**. The human-review
template remains blank. No reward was computed.

## Usage, shutdown and interpretation

Actual use was **6 completed model requests**, 21,349 reported input tokens and
355 output tokens, within twelve requests total and six per case. No request
failed or was resubmitted. The 900-second watchdog did not fire. The lifecycle
records ~169.470 seconds before cleanup; that is not invoiced GPU time. Actual
billed spend is unknown. Startup/cleanup/idle/storage can bill, and these bounds
are not a provider dollar ceiling.

All six jobs are recorded completed. Cleanup and independent subsequent MCP
reads agree: min/max 0/0, no active workers, all health worker counts zero and
no queued/in-progress jobs. The [final state](../../examples/assigned-reader-focused-live/final-runpod-state.json)
records those reads. No further activation occurred.

The result is mixed on two familiar assistant-authored development questions:
the factual answer improved, while the target-setup case acquired a source
condition-mapping error and still lacks a gap/query. The same trained checkpoint
was used throughout; there was no training, base-model control, held-out set or
larger-assistant comparison. Prompt/controller guidance and MCP invocation path
changed, and cold-start/host conditions differ. Neither general model improvement,
fine-tuning benefit nor latency/cost advantage is established. The candidate is
not promoted by this result; host semantic review remains necessary.

The unchanged implementation passed **1,677 tests** during preparation, with no
skips and three existing SWIG warnings. No source code changed during this paid
attempt. Fresh frozen-source checks, live MCP fidelity/provenance checks, budget
audits and shutdown checks passed. The [validation record](../../examples/assigned-reader-focused-live/validation.json)
contains exact artifact hashes, test provenance, preservation checks and
independent review. All protected files, earlier artifacts and frozen preparations
remain unchanged.
