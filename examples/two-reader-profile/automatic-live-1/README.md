# Automatic live profile — one attempt

Prepared after the user's “ok continue” instruction. Earlier original and
replacement attempts remain consumed and unchanged. This run tests the native
HTTP startup reader and transport-normalized payload guard without a manual
handoff. It uses the same two saved inputs, pinned model and adapter, A5000,
two-sequence configuration and cache-disabled decoding as the earlier profile.

Maximum allowance: ten model jobs, nine telemetry jobs, 81,920 reserved tokens,
one GPU and 900 seconds including loading. At the fresh serverless catalog
rate of $0.69/hour, that time window estimates $0.1725 compute, below the $0.50
local allowance. There is no provider dollar cap; actual invoice cost is unknown.

Pre-activation checks: 38 focused tests passed in 23.01 seconds; full suite
1904 passed in 98.79 seconds, no skips, three existing SWIG warnings. Ruff and
whitespace checks passed. Protected files and prior artifacts/markers retain
their recorded hashes. Implementation commit: `10607e1`.

Only compare serial/concurrent timings if all outputs, token counts and finish
reasons agree and telemetry confirms overlap. Differences within 0.4 seconds
are inconclusive. These two short inputs cannot establish tail latency, useful
answer quality, adaptive agent behavior or a fine-tuning benefit.

See `plan.json`, `authorization.json` and `preflight.json` for the fixed scope.
Results will be recorded separately after this attempt and verified shutdown.
