# Approved replacement smoke — October 20, 2026

The replacement reached a ready A5000 worker, then stopped while retrieving its
startup identity logs. Neither question reached Qwen. The endpoint is verified
parked again, with worker min/max 0/0, zero worker records and no pending jobs.
There was no further activation after this replacement.

## Authorization and unchanged experiment

The user explicitly approved one replacement after the
[first aborted attempt](2026-10-20-assigned-reader-smoke-attempt.md). The
[approval amendment](../../examples/assigned-reader-smoke-replacement-approval.json)
was saved before activation and binds the original plan hash and unchanged limits.
The original marker was archived without changing its bytes; the replacement has
its own exclusive attempt marker. Original attempt artifacts remain untouched.

Commit `ebbd0d4` adds an optional authorization file and rejects a changed plan hash,
expanded limits or a replacement count other than one. The replacement's
[runtime record](../../examples/assigned-reader-smoke-replacement/runtime.json)
embeds the exact approval record and its hash, plus the implementation file hashes.

A fresh catalog read returned RTX A5000 24 GB serverless at $0.69/hour with high
availability. The same pinned v5 adapter, source paper, effective prompt, questions,
Docker sandbox and decoding were retained. The hypothesis remains whether v5 can
inspect the assigned paper, answer the batch-size question with supporting evidence,
and report a qualified A5000/batch-8 gap without inventing a target-setup speedup.
The frozen decision rule requires both claim-reviewed cases and no execution,
repeated-call or budget failures; this setup attempt cannot pass that rule.

Limits remained six model requests per case, twelve total, 7,168 input and 1,024
output tokens per request, one active GPU worker, and a 900-second watchdog armed
before activation. Original plus replacement attempts used zero model requests.

| Case | Actual outcome | Model requests |
| --- | --- | ---: |
| Reported walltime batch size | Not attempted; startup aborted | 0 |
| A5000/batch-8 speedup or qualified gap | Not attempted; startup aborted | 0 |

There are no model answers, spans or claim-level answer verdicts to review. Human
answer review was not performed. This provides no model execution-error rate,
answer-quality result, fine-tuning improvement or comparison with a larger assistant.

## Observed failure

The saved latest health response reports ready=1. The next identity-log read raised
`ControlError` with recorded code `unknown` and status null. The inference child
never launched: no WORKER_STARTED marker or inference ledger exists. The
[lifecycle record](../../examples/assigned-reader-smoke-replacement/lifecycle.json)
retains ~30.425 seconds before cleanup; that interval is not billed worker time.

A read-only reproduction after shutdown exposed this CLI stderr shape:

```text
--tail is ignored when --since is set
{"error":"worker not found","code":"not_found","status":404}
```

The attempted wrapper parsed all stderr as one JSON object, so a warning before
the JSON discarded the structured error code and prevented classification for
bounded transient reads. This is a confirmed parser weakness. The reproduction's
404 was observed after/overlapping shutdown and does not prove the discarded
active-time error was also 404, that retries would succeed, or that the provider's
live log-access path works. That path remains unverified.

The [Runpod CLI README](https://github.com/runpod/runpodctl#reading-logs) documents
that log reads can fail on a filter timeout. Our reproduction also shows a warning
before the JSON error. A worker reporting ready is insufficient to establish the pinned runtime
identity; this attempt kept the identity check rather than dispatching inference
without it.

## Repair and final state

Commit `32e18c8` scans stderr for its structured error record while retaining only
safe command/code/status diagnostics, and stops passing --tail together with --since.
A regression uses the observed warning-prefixed error shape. This repair is locally
tested, independently reviewed, and has not been live-tested. It is not a demonstrated
fix for provider log access. No further paid attempt is proposed until that access
path can be validated.

Shutdown read-back and later independent MCP reads agree with the
[final state](../../examples/assigned-reader-smoke-replacement/final-runpod-state.json):
0/0 worker limits, zero active worker records, all health worker counts zero and
no queued/in-progress jobs. There was a transient disagreement between the first
post-cleanup health rollup and worker list; the subsequent final reads both report
zero. The final state is based on these later reads.

Actual billed spend is unknown. GPU startup and storage can bill despite zero
inference, and the request/token/watchdog limits are not a provider dollar ceiling.
The user-approved replacement is exhausted and its marker remains in place.

## Verification

Focused tests: **49 passed** for approval binding, startup parsing/readiness,
runner shutdown, preparation and queue budgets. Ruff and whitespace checks passed.
The final full suite passed **1,621 tests**, no skips, with three existing SWIG
warnings in 63.03 seconds. Commands and protected-file hashes are in the
[validation artifact](../../examples/assigned-reader-smoke-replacement/validation.json).
Independent review found no actionable issue in the authorization amendment or
parser repair, and confirmed the failed-attempt interpretation. All ten protected
file hashes remain unchanged.
