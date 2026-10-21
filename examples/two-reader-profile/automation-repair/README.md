# Automatic profile plumbing — locally verified

The timing runner now verifies startup through a direct, bounded HTTP/SSE worker
log reader. It no longer calls the CLI for startup logs. The existing supervisor
still owns activation, its original absolute deadline, process cleanup and
shutdown; model identity and readiness checks remain mandatory.

Commit `7596653` adds this reader and wires it into the real profile guard. It
uses only the fixed HTTPS worker-log route, with no proxy, redirect or internal
retry. Each snapshot has a five-second ceiling, a quiet-read timeout, a 1 MB
response limit and a record limit. It binds records to the requested worker and
rejects malformed, truncated or foreign records. The supervisor retains the
strict source/time checks and bounded private diagnostics. The new dependency
is included in preparation hashes. Other smoke runners retain their existing
startup transport.

Commit `45a815a` adds a complete loopback integration check using the actual
reader, supervisor, child process, queue client and worker hook. A local JSON
intermediary normalizes integral floating-point values; a scripted model
supplies responses. It verifies:

- Recovery from delayed headers and automatic identity/readiness verification,
  without CLI log reads or a manual handoff.
- Two warmups, four batches, ten model requests and nine telemetry jobs.
- Matched request content across JSON serialization and blocking request eleven.
- Rejection of altered content before model dispatch, retained terminal-job
  errors, no inference retry, and cleanup on the failed child path.
- HTTP error classification, no redirected credentials, bounded streams,
  malformed records and continued enforcement of startup deadlines.

Run the public loopback checks without a GPU:

```sh
python -m pytest -q tests/test_profile_wire.py tests/test_worker_log_reader.py
```

## Exact saved-input component check

[saved-input-check.json](saved-input-check.json) records an additional local
check using the exact two saved input payloads from the failed live attempt.
Their historical canonical hashes remain unchanged. The pinned cached tokenizer
recounted 1,400 and 2,043 prompt tokens. The real queue client and hook accepted
numeric normalization, preserved messages, completed ten scripted requests and
nine telemetry jobs, and blocked an additional model request. The new reader
also parsed the selected saved startup records and passed the existing identity
and readiness functions.

This is a local component replay, not fixed-action model replay or live model
inference. Raw payloads, fake model timings and request traces remain under
ignored `out/profile-automation-repair-20261021/`. Repeating this exact saved-input
check requires those local payloads and tokenizer cache; the public tests use
controlled fixture inputs and dummy credentials. No model-authored code executes.

## Verification and limits

Focused checks: 51 passed in 13.33 seconds, then all three full-wire checks
passed in 11.43 seconds. The full suite passed **1903 tests in 107.59 seconds**,
with no skips and three existing SWIG deprecation warnings. Ruff and whitespace
checks passed. [validation.json](validation.json) binds code, evidence and
protected-file hashes.

All eleven protected files and all 33 prior artifacts/consumed-marker records
match their pre-task hashes. No earlier failed-run interpretation was rewritten.
[endpoint-after.json](endpoint-after.json) reports min/max 0/0, zero workers and
no pending jobs. Only three final read-only cloud checks were made; this repair
made no GPU activation or inference submission and grants no new paid allowance.

The original live received input remains unavailable, so numeric normalization
is still a reproduced candidate cause, not a proven provider transformation.
The updated automatic path has not been tested live. Scripted response times,
phase counters and token usage cannot measure GPU speed or answer quality.
There is no new evidence for MCP benefit or fine-tuned Qwen improvement.
