# Prepared two-reader serving diagnostic — no paid run yet

This prepares the next measurement proposed by the
[experiment-mode walkthrough](../experiment-mode-walkthrough/README.md).
It compares the same two historical next-action inputs sent serially versus
concurrently to one Qwen3-8B v5 engine. No model-generated action is executed.
It is not a complete agent task, an answer-quality test, or evidence that MCP
outperforms ordinary research.

## Prepared scope

- Same pinned base/adapter and exact two input hashes, locally recounted as
  1400 and 2043 input tokens using the cached matching tokenizer.
- Two warmups, then serial/concurrent and concurrent/serial rounds: ten total
  inference requests, each reserving 7168 input plus 1024 output tokens.
- Nine separate telemetry jobs: preflight, plus begin/end for each batch. They
  generate no model tokens but consume worker time within the session window.
- One A5000, one engine, absolute 900-second window including loading; stop
  after uncertain inference, worker change, missing telemetry or invalid usage.
- Capture actual engine phase counters and sampled concurrency/KV usage, client
  pair time, per-request timings and before/after GPU memory snapshots.

Prefix caching is already disabled on the endpoint, so cache reset calls are
unnecessary. Both arms retain that same setting. Memory snapshots are not true
peak device memory, sampled gauges can miss short overlaps, and instrumentation
and 200 ms queue polling add overhead. A difference within a 400 ms polling
uncertainty margin is reported as inconclusive. Two rounds cannot estimate tails
or establish a general speedup. Changed outputs/token counts also make the
comparison inconclusive; temperature zero is not a determinism guarantee.

## Current state and authorization

[preflight.json](preflight.json) records read-only Runpod observations: endpoint
`q05pjrvguakieg` is parked at min/max 0/0, with zero workers and zero pending jobs.
Its image/entrypoint and serving configuration were left unchanged. The current
A5000 catalog rate is $0.69/GPU-hour; 900 seconds at that rate is $0.1725 of GPU
compute. That estimate is not an invoice or a provider dollar cap.

The user was asked to choose a total session allowance of $0.50 or $1, or local
preparation only. **No answer has arrived, no authorization file was created,
and no endpoint mutation, inference submission or GPU activation occurred.**
The old research/reader test budgets and consumed attempt markers are untouched.

The prepared plan is [plan.json](plan.json). Its task-owned local directory is
`out/two-reader-profile-preparation-20261021/ready/`. Exact input payloads and the
generated entrypoint stay in ignored local files. Preparation itself downloads
no tokenizer/model files and makes no inference/control-plane requests.

```sh
python scripts/profile_two_readers.py check \
  --output out/two-reader-profile-preparation-20261021/ready
```

To prepare a separate local copy, run `prepare --output <new-directory>` with
the cached tokenizer and original saved input ledger available. A fresh copy
has a fresh session/plan digest; it does not grant another paid attempt.

After a budget answer, the operator must record the actual instruction and a
fresh price read in a plan-bound authorization before running. The `run` phase
consumes one experiment-wide attempt marker, installs only the telemetry
entrypoint on the parked endpoint, verifies its read-back, and uses the existing
absolute-deadline supervisor for startup, identity checks, execution and shutdown.
It then restores/read-checks the original entrypoint. Do not bypass the supervisor,
delete the marker, or create new directories to repeat paid work.

## Implementation and local checks

The opt-in hook wraps the pinned image's original chat handler. It accepts only
the two exact input hashes, caps model requests at ten, and obtains fixed-path
loopback metrics instead of sending Prometheus text through the image's JSON
proxy. It adds no arbitrary code-execution endpoint. Returned errors and partial
responses remain in the durable local request log; uncertain POSTs are not retried.
This instrumentation has not been verified on a live GPU yet.

Twelve new local tests exercise parser/state boundaries, the real hook/client
schedule with scripted responses, hard request counts, immutable inputs,
uncertain response handling, partial concurrent results, telemetry failure and
restoration after an uncertain endpoint PATCH. The later focused run with the
existing Docker image passed 23 checks before the final returned-error regression
was added; the new twelve-check file then passed. No fixture timing is published
as GPU performance.

The final full suite passed **1867 tests in 109.08 seconds**, with no skips and
three existing SWIG deprecation warnings. Ruff and `git diff --check` passed.
The final prepared-plan check reproduced its pinned hashes and made zero model
calls. [validation.json](validation.json) records the checks and protected-file
hashes.

Actual serving results and any ordinary-assistant versus experiment-mode
benchmark remain unmeasured. Human review is absent. No fine-tuning improvement
or cost advantage is claimed.
