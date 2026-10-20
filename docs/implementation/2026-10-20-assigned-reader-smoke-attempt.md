# Assigned-reader smoke attempt — October 20, 2026

The approved live attempt stopped during setup, before either question reached Qwen.
This is a supervisor startup failure, not a model-quality result. The endpoint is
verified parked again: min/max workers 0/0, zero worker records and no queued or
in-progress jobs. No automatic replacement attempt was performed.

## What ran

The user approved activation after a live Runpod catalog check quoted an RTX A5000
24 GB serverless worker at $0.69/hour with high availability. The attempted runner
was committed as `d6b9532`; its exact file hashes and user-approval record are in
[runtime.json](../../examples/assigned-reader-smoke-startup-aborted/runtime.json).
The [frozen preparation](../../examples/assigned-reader-smoke-prepared/plan.json)
remained unchanged: one batch-size question and one A5000/batch-8 evidence-gap
question, six model requests per case, twelve total, 7,168 input/1,024 output tokens
per request, one attempt, and a 900-second session watchdog.

The supervisor armed the watchdog before changing this named endpoint from 0/0
workers to 1/1. A fresh configuration read confirmed that worker band and the
unchanged A5000 selection. Startup code would have checked the pinned v5 identity
before launching inference; model-authored Python would run through the existing
Docker-only assigned reader. The child inference process never launched.

| Case | Actual result | Model requests |
| --- | --- | ---: |
| Reported walltime batch size | Not attempted; startup aborted | 0 |
| A5000/batch-8 measured speedup or qualified gap | Not attempted; startup aborted | 0 |

There is no answer, quote, execution-error rate, claim-level answer verdict, or
latency comparison to score. The hypothesis and decision rule are still untested.
No semantic-support pass, trained-versus-base improvement, or comparison with a
larger assistant is established. Human answer review was not performed.

## Failure and cleanup

The runner fetched container logs as soon as a worker record appeared. The latest
saved health response still reported ready=0. The CLI log call failed, causing the
supervisor to abort. The original wrapper discarded the CLI's stderr, so the exact
provider error is unknown; a 404 or a particular provider failure is not established.

The supervisor's `finally` path restored min/max workers to 0/0 and verified empty
worker records, zero health worker counts and no pending jobs. Its recorded
~2.996-second interval ends before cleanup; it is neither billed worker duration
nor an invoice. Actual spend is unknown. Zero inference does not establish zero
startup/storage cost. The request/token/watchdog limits are not a provider dollar cap.

The raw [lifecycle record](../../examples/assigned-reader-smoke-startup-aborted/lifecycle.json)
reports shutdown_verified=true, cleanup_errors=[], watchdog_fired=false and no child
exit code. Independent fresh MCP reads agree with the
[final state](../../examples/assigned-reader-smoke-startup-aborted/final-runpod-state.json).
No inference ledger or WORKER_STARTED marker was created. The original global
LIVE_ATTEMPT marker remains in place to prevent an automatic second activation.

## Local repair after the attempt

Commit `518e52c` waits until the current worker reports ready before fetching its
identity logs. Identity still requires the currently active worker ID and a timestamp
after activation. Known transient log-read errors receive at most three read-only
attempts; authentication/unknown errors fail immediately. This does not retry
inference. CLI failures now retain only safe command/code/status diagnostics.

The earlier independent review also found and closed stale identity acceptance and
successful shell exits after failed child/cleanup/packet verification. Shutdown now
has a bounded read-back check, and incomplete execution due to infrastructure or
verification failure exits unsuccessfully while retaining saved case evidence.

The repair has been tested locally and independently reviewed; it has not been
live-tested. A replacement paid attempt requires a new explicit approval/amendment
to the frozen one-attempt rule, while preserving this attempt's records and limits.

## Verification

- Final focused suite: 48 passed (startup handling, smoke runner/preparation and queue budgets).
- Final full suite: **1,620 passed**, no skips, three existing SWIG warnings, 55.48 seconds.
- Ruff, CLI help and diff whitespace checks passed.
- All ten protected file SHA-256s match the pre-existing values.
- Independent review closed the pre-run findings and found no actionable issue in
  the local startup repair or the failed-attempt interpretation.

Commands, original configuration, summaries and hashes are in the
[validation artifact](../../examples/assigned-reader-smoke-startup-aborted/validation.json).
The final suite used the cached tokenizer and the real pinned Docker image; scripted
model responses in tests are not live v5 results.
