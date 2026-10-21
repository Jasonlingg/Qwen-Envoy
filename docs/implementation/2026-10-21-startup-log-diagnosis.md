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
