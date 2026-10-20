# Focused MCP end-to-end preparation — October 20, 2026

The candidate test is frozen and locally verified. **No GPU was activated and no
paid inference was performed.** The prepared plan still has `approved: false`.
It is ready for review and one separately approved live attempt.

## Concrete test

The two questions, paper version `2211.17192v2`, cached source/corpus, v5 adapter,
base revision, Docker image, temperature 0, top-p 1, seed 42 and disabled thinking
are unchanged from the [saved live baseline](2026-10-20-assigned-reader-smoke-v2logs.md).
The [candidate prompt](../../examples/assigned-reader-focused-prepared/system-prompt.txt)
uses `focused-v1`; its SHA-256 remains
`c0f14e1dca80c88d477f6a96b1676c621b54ef93c8f27de94d701617ea67d7ad`.

| Question | Required source-visible result |
| --- | --- |
| What batch size was used for the walltime measurements? | Correct batch-size answer, supporting inspected text and exact source spans |
| What speedup is established specifically for A5000 at batch 8? | Qualified target-setup gap and useful follow-up query; no invented or transferred measurement |

Both must satisfy the frozen claim-review requirements and MCP/packet checks,
with zero Python errors, controller/submission/gap rejections, repeated calls or
budget breaches. Recovered errors will be reported rather than hidden. Exact
quotes alone cannot pass a question. Human review remains unperformed and its
template remains blank. These are familiar development regressions, not held-out
questions or a fine-tuning comparison.

Limits remain **two cases, six actions/requests per case, twelve model requests
total, 7,168 input and 1,024 output tokens per request, one A5000 worker and a
900-second watchdog armed before activation**. No inference resubmissions or
automatic replacement attempts are allowed. Initial candidate inputs contain
2,462/2,488 tokens. References and correct-answer criteria are separate files and
are not included in model requests.

## Actual end-to-end boundary

The earlier live smoke called the reader directly. This candidate executor now
uses the real local MCP client/protocol and `investigate` tool, followed by the
bounded remote Qwen transport, Docker document actions, packet verifier and
host-visible MCP result. The existing supervised activation, pinned worker
identity check and shutdown remain in force for the later live attempt.

This test uses an in-process MCP transport and a one-paper frozen cache adapter.
It does **not** test an external desktop client, stdio process launch, fresh web
discovery or an optional reasoner. The cache adapter rejects other versions and
discovery calls instead of fetching sources. There are no extra model calls for
MCP or synthesis.

The executor saves raw reader evidence before MCP verification so a failure
cannot erase it. It also saves the host-visible MCP response, then checks that
the candidate answer, quotes, valid gap reports, prompt/model/guidance/paper
identity, corpus, request count and status survive the boundary. Separate
metrics record action, submission and gap-report rejections.

Independent review identified a missing host-answer equality check; this is
fixed and corruption regressions cover dropped/changed answers and identities.
Integration testing then caught an incorrectly expected top-level corpus field;
commit `67769c2` checks the actual paper-assignment field. The final corrected
suite passes. Implementation is in `2e91e66` and `67769c2`.

## Verification and preserved evidence

The focused suite passed **101 tests** in 8.44 seconds. The full suite passed
**1,677 tests**, no skips, with three existing SWIG warnings in 58.27 seconds.
Ruff and whitespace checks passed. Tests include actual MCP/Docker execution
with scripted responses, unchanged request limits, scope tampering, host-response
corruption, raw-evidence retention, startup identity and shutdown behavior.

A separate [scripted rehearsal](../../examples/assigned-reader-focused-prepared/scripted-rehearsal.json)
used the real frozen paper, cached tokenizer, MCP tool and Docker, with six
assistant-scripted responses and **zero model requests**. It returned both
mechanically verified outcomes and a host-visible gap. Those scripted answers
are not Qwen output and provide no answer-quality or training-benefit evidence.
The rehearsal procedure is saved alongside its summary.

All prior attempt artifacts, original preparation and protected uncommitted files
are unchanged. The old exclusive attempt marker remains in place. No approval
record was created and no marker was reset or archived in this preparation.
The new plan pins the guidance module, MCP adapter and execution/control files
along with the model-facing implementation.

The [read-only readiness check](../../examples/assigned-reader-focused-prepared/readiness.json)
found worker limits 0/0, no active workers or pending jobs, and a current A5000
serverless quote of **$0.69/hour**. Availability/pricing must be rechecked before
activation. Request/token/watchdog bounds are not a provider dollar ceiling;
startup, cleanup, idle or storage can bill. Actual future billed spend is unknown.

The [frozen plan](../../examples/assigned-reader-focused-prepared/plan.json) and
[validation record](../../examples/assigned-reader-focused-prepared/validation.json)
provide the exact hashes and reviewable limits. A live run requires a new
plan-bound approval for one attempt; no previous exhausted allowance is reused.
