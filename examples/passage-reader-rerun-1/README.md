# Repaired-harness Qwen passage-reader rerun

One newly authorized attempt on the same three exposed development cases and
host-selected source windows as the [first trial](../passage-reader-three-case/README.md).
The checkpoint and limits are unchanged. The harness now normalizes bounded
read-length arithmetic and supplies exact literal read examples. This is a live
model rerun, not fixed-action replay; local scripted rehearsal is synthetic.

The [predeclared scope](../../docs/implementation/2026-10-21-passage-reader-rerun.md)
requires supported answers on both answerable cases and a completed, scoped
missing-evidence procedure on the third. Semantic review is separate from quote
provenance and mechanical acceptance. Independent human review remains blank.
No training, reward changes, held-out comparison or productivity claim is included.

## Actual result

The overall rule was **not met**. Two answerable cases now have supported answers
under implementing-assistant source review, compared with one in the first run.
The third case still did not finish the gap procedure. Independent human review
remains blank; model weights did not change.

| Case | First attempt | Repaired rerun | Calls |
| --- | --- | --- | ---: |
| QLoRA dtypes and gradients | Incomplete | Requested facts supported by its two citations | 3 |
| AWQ Figure 9 protocol | Supported answer | All six requested fields supported | 2 |
| PagedAttention target p95 | Incomplete | Two reads; premature abstention; gap recorded too late | 4 |

All five passage reads succeeded. The gap case did not invent a target latency,
but it returned no substantive partial findings and exhausted its four actions
before a valid final abstention. There were zero Python execution errors and one
rejected submission. Retain that distinction: incomplete procedure is not a
factually incorrect answer, and a recorded gap is not proof of absence.

- [Per-question/action results and usage](outcome.json)
- [Source-based claim review](claim-review.json), separate from mechanical checks
- [Full real MCP reports and persisted experiment records](live/summary.json)
- [Actual model request/response ledger](session.requests.jsonl)
- [Frozen implementation and source hashes](plan.json), [checkpoint/decoding](experiment-config.json), [effective prompt](system-prompt.txt)
- [Lifecycle](lifecycle.json), [full configuration restoration](configuration-restoration.json), [fresh shutdown reads](endpoint-final.json), [owned Docker cleanup](docker-proof.json)
- [Validation and artifact hashes](validation.json)

Nine actual requests used 25,784 input + 518 output = **26,302 reported tokens**.
They reserved 73,728 tokens within the 98,304 cap; no inference resubmissions ran.
Native identity/readiness verification used thirteen received startup snapshots.
Supervisor time before cleanup was 185.907 seconds; the 900-second watchdog did
not fire. Fresh reads at 23:19:41 UTC verified workers 0/0, no workers or pending
jobs, and the full original endpoint configuration restored. All six owned
rehearsal/live Docker containers were absent. Actual invoice cost is unknown.
The local $0.50 allowance is not a provider-enforced dollar ceiling.

Scope commit `2ffb7d9`; frozen preparation/live implementation `df8f243`. Focused
checks: 25 passed. Full suite: 1925 passed in 160.61 seconds, no skips, three
existing SWIG warnings. Eleven protected files and 381 prior artifacts/markers
were unchanged. The new attempt marker is consumed; no further paid attempt ran.

Both arithmetic normalization and exact read examples changed between attempts.
This rerun chose literal lengths, so it did not exercise arithmetic normalization
or isolate which intervention mattered. These familiar source-selected examples
do not prove held-out generalization, fine-tuning improvement, autonomous paper
discovery, or quality/cost/effort benefit over ordinary assistant research. The
combined MCP path was a real in-process client/server with remote Qwen and local
Docker; it was not a new native-host installation or optional stdio deployment
test. Default experiment-mode stdio behavior remains covered by the full suite.
