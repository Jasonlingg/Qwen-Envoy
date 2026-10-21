# Automatic profile live validation

The user said “ok continue” after the automatic local repair passed and live
validation remained outstanding. This authorizes one fresh attempt under the
same local $0.50 allowance: one A5000, 900 seconds including loading, ten model
requests, nine telemetry jobs and 81,920 reserved tokens. No further attempt is
included. Both prior attempts and their consumed markers remain unchanged.

Hypothesis: native bounded startup-log reads and transport-normalized input
hashes allow the fixed-input profile to complete automatically. Expected signal:
verified current-worker identity/readiness, both warmups and four timing batches,
with no manual handoff. Decision rule: compare timings only if outputs, token
counts and finish reasons match, telemetry confirms engine overlap, and all
budget and shutdown checks pass. Differences within 0.4 seconds are inconclusive.
Any conclusive failure stops inference without a retry.

The serverless catalog read at 2026-10-21 20:55:57 UTC reports $0.69 per GPU-hour.
The maximum 900-second compute-window estimate is $0.1725. This local time guard
is not a provider dollar cap or an invoice. The endpoint was parked at 0/0,
with zero workers and zero pending jobs. Only its temporary entrypoint and
worker activation will change; the original settings must be restored.

A fixed new marker binds this allowance to the consumed replacement's outcome,
plan and marker hashes. Changing output directories cannot grant another run.
Prepare using the pinned cached tokenizer, run focused and full checks before
activation, and record results separately. This is a serving diagnostic with
two short historical inputs, not answer-quality or fine-tuning evaluation.

## Outcome

Implementation `10607e1`, preparation record `85bd051`. Focused checks: 38
passed in 23.01 seconds. Full suite: 1904 passed in 98.79 seconds, no skips,
three existing SWIG warnings. Ruff and whitespace checks passed.

The automatic live path completed all ten model and nine telemetry jobs. Both
inputs produced matching outputs, usage and finish reasons on all five calls;
both concurrent windows sampled two engine requests running. All fifteen native
startup reads returned records and no manual handoff was needed. Returned usage
was 17,350 total tokens. No model-generated code was executed.

Performance remains inconclusive: round 1 was 3.423 seconds serial versus 3.355
concurrent, within the 0.4-second margin; round 2 was 3.829 versus 2.850. The
runner deliberately exited 1 rather than accept an incomparable result. No
retry ran. Device-memory snapshots were 21,045 / 24,564 MiB in each batch.

Shutdown and full original configuration restoration were verified. Fresh reads
at 21:03:54 UTC show workers 0/0, zero active workers and zero pending jobs.
The allowance is consumed; actual invoice cost is unknown. See the
[saved live evidence](../../examples/two-reader-profile/automatic-live-1/README.md).
These results verify this bounded serving path, not research usefulness or
fine-tuning improvement.
