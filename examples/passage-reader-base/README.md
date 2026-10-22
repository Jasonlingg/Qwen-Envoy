# Base Qwen passage-reader attempt

One newly authorized base Qwen3-8B attempt on the same three exposed development
cases as the [saved V5 repaired rerun](../passage-reader-rerun-1/README.md).
The [predeclared scope](../../docs/implementation/2026-10-21-base-passage-reader.md)
keeps prompt, source windows, tools, four-action limit and decoding unchanged.
Only the model route changes to `envoy-base-only`, without applying V5.

First requests were checked against V5's frozen requests: only `model` differs.
The effective prompt SHA and product implementation hashes match. Source,
question and review-input hashes match; references stay outside model inputs.
Scripted local rehearsal is synthetic and does not establish model quality.

## Actual comparison

Base returned one supported answer under implementing-assistant source review;
the saved V5 repaired rerun returned two. Both failed the required gap procedure.
The overall rule was not met for either arm. Independent human review stays blank.

| Case | Base | Saved V5 | Base / V5 calls |
| --- | --- | --- | ---: |
| QLoRA dtypes/gradients | All requested facts supported | All requested facts supported | 4 / 3 |
| AWQ Figure 9 protocol | Incomplete; source reads never executed | Six requested fields supported | 4 / 2 |
| Missing target p95 | Incomplete; missing gap report | Incomplete; gap report too late | 4 / 4 |

Base first tried two reads in one action on QLoRA and the gap case, then recovered
with separate reads after rejection. On AWQ it mixed read/search code with an
Unanswerable submission in every response. The existing policy cleaner selected
the submission line, so the read code never executed. All four abstentions were
rejected for missing source inspection. This does not measure base's semantic
ability to interpret the AWQ passage. No target p95 was invented in the gap case.

Base had two action rejections and five submission rejections, versus one
submission rejection for V5. Neither run had a Docker Python execution error.
V5 followed the current tool workflow better in this small diagnostic; the
comparison does not establish a general fine-tuning improvement or stronger
technical knowledge. Training may target repeatable protocol errors, but another
training job needs reviewed demonstrations, stable interfaces and separate
evaluation. These exposed cases must not become a held-out test by relabeling.

- [Every question, raw model output, policy action and observation](outcome.json)
- [Claim-level source review](claim-review.json)
- [Real MCP reports and durable experiment records](live/summary.json)
- [Actual model request/response ledger](session.requests.jsonl)
- [Checkpoint and decoding](experiment-config.json), [effective prompt](system-prompt.txt), [implementation/source hashes](plan.json)
- [Lifecycle and owned-job states](lifecycle.json), [full restoration](configuration-restoration.json), [fresh shutdown reads](endpoint-final.json), [Docker absence proof](docker-proof.json)
- [Validation and artifact hashes](validation.json)

Twelve base requests used 33,974 input + 672 output = **34,646 reported tokens**,
reserving the full 98,304-token cap. V5 used nine requests and 26,302 reported
tokens. This is observed usage in two separate sessions, not a general cost or
latency comparison; invoices are unknown. No new V5 inference or resubmissions ran.
All twelve jobs completed with no new failed/retried jobs. Automatic startup
used thirteen received native log snapshots; supervisor time before cleanup was
203.884 seconds and the watchdog did not fire. Fresh reads at 00:54:54 UTC on
October 22 verified zero workers/jobs and full original configuration restored.
All six owned local rehearsal/live Docker containers were independently absent.

Implementation/scope: `6c84969`; frozen preparation/live commit: `4ad24e3`.
Focused tests: 46 passed. The first full run had one Docker startup failure before
any scripted model request, while a separate rehearsal also ran (1928 passed).
The exact cause is unproven. No code/timeouts changed: four standalone Docker
tests then passed, and the standalone full suite passed **1929 tests in 152.10
seconds**, no skips, three existing SWIG warnings. All test logs are retained.
Eleven protected files and 507 prior artifact/marker hashes remained unchanged.

The new base allowance is consumed; original attempts remain frozen. These are
three familiar, host-selected passage cases, with an implementing-assistant
review and no independent human labels. They do not establish held-out
generalization, autonomous discovery, productivity benefit, training causality
or statistical superiority. The real MCP path was an in-process client/server
with remote Qwen and local Docker, not a new native-host installation.
