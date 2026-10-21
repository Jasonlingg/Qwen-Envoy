# Startup-log timeout investigation and repair

Scope: diagnose the October 21 aborted reader profile without activating a GPU
or submitting inference. Preserve all consumed allowances and saved failed-run
artifacts. Read the installed CLI source at `dd55bcf` and reproduce its actual
HTTP/SSE behavior locally with dummy credentials.

Hypothesis: a quiet or delayed log response can yield the same structured
`timeout` as a transport failure. The supervisor incorrectly turns three such
reads into terminal startup failure before its absolute deadline, and loses
identity/readiness lines between reads.

Expected signal: the actual installed CLI returns `timeout` from a local server
that withholds headers, while later log output is readable; the repaired
supervisor survives multiple pending reads, retains same-worker evidence across
reads and dispatches only after current pinned identity and readiness.

Decision rule: accept only with local regressions for quiet/delayed responses,
partial output retention, worker replacement/staleness, identity mismatches,
permanent errors and deadline shutdown. The original live cause is established
only if actual retained provider logs identify it; local reproduction alone
cannot prove that cause. No new paid attempt is part of this repair.

## Findings and repair

The pinned CLI transport reproduces the timeout on a local server withholding
response headers, without a GPU. The old supervisor aborts after three such
errors. Its per-poll identity/readiness observations were discarded, and its
private diagnostics did not retain stdout. The repair is committed in `066d0c5`:
transient observations remain pending under the original watchdog, both log
sources are read, and bounded private logs and same-worker evidence survive
polls. Wrong identities, authentication errors, stale state and deadlines still
prevent dispatch. Updated preparation pins include lifecycle/control code.

A fresh read of the old worker's system logs returned 404. Endpoint reads
reported 0/0, zero workers and no pending jobs. No cloud settings were changed;
no GPU activation or model request was made in this investigation. The actual
provider-side reason for the original missing response headers cannot be
recovered from the available records. Do not claim a hardware diagnosis or a
live-verified repair.

The [local evidence](../../examples/two-reader-profile/log-repair/README.md)
records the CLI reproduction, regression coverage, unchanged failed-attempt
artifacts and protected-file checks. The focused suite passed 67 tests in 8.54
seconds; the full suite passed 1878 tests in 77.40 seconds, with no skips and
three existing SWIG deprecation warnings. Ruff and whitespace checks passed.
