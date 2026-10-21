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

The automatic execution completed: 10/10 model jobs and 9/9 telemetry jobs,
zero new failed jobs, no manual handoff. All 15 native startup-log reads
returned records. The live identity matched the pinned base and v5 adapter.
Returned usage totals 17,215 input and 135 output tokens; reservations stayed
at 81,920. Each input completed five times with identical output, token counts
and finish reason across warmup, serial and concurrent runs.

| Round | Serial pair | Concurrent pair | Observed difference |
| --- | ---: | ---: | ---: |
| 1 | 3.423 s | 3.355 s | 0.068 s |
| 2 | 3.829 s | 2.850 s | 0.979 s |

**Performance is inconclusive.** Round 1 is within the predeclared 0.4-second
polling margin. The runner deliberately exited 1 for this rejected comparison;
all jobs had completed. No retry ran. Both concurrent windows sampled two
requests running in the same engine. Device-memory snapshots around all four
batches were 21,045 / 24,564 MiB; these are snapshots, not peak-memory estimates.
This establishes that two reader requests can overlap on one loaded model in
this configuration. It does not establish a reliable speedup.

The outputs were a search action and an unanswerable submission, preserved in
`outcome.json`; they were not executed or graded for research usefulness. This
is not an end-to-end research MCP answer test. The original prior rejection's
received-input transformation remains unknown, although both repaired input
guards accepted this run.

The watchdog did not fire. Supervisor time before cleanup was 257.465 seconds.
Runpod briefly rescheduled/throttled workers before usable startup. Final reads
at 21:03:54 UTC confirmed workers 0/0, zero active workers or pending jobs,
and the full original endpoint configuration restored. No further paid attempt
is authorized by this record. Actual invoice cost remains unknown.

See `plan.json`, `authorization.json` and `preflight.json` for the fixed scope;
`result.json` and `profile.requests.jsonl` preserve all responses and telemetry.
`outcome.json` summarizes per-input/per-batch results, and `validation.json`
records checksums and verification. To reproduce the saved assessment locally:

```bash
python - <<'PY'
import json
from pathlib import Path
from src.eval.reader_profile import assess_profile
path = Path('examples/two-reader-profile/automatic-live-1/result.json')
result = json.loads(path.read_text())
assert assess_profile(result) == result['assessment']
print(result['assessment'])
PY
```
