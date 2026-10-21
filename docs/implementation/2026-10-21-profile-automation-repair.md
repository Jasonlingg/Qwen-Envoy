# Automatic startup and serialized-request repair

The user asked to fix the remaining failures. Repair locally; no new GPU
activation, inference allowance, attempt-marker reset or manual handoff.
Preserve both live attempts and protected user changes.

Hypothesis: the profile can verify startup automatically through a bounded
direct HTTP/SSE tail reader, then complete its fixed-input request protocol
despite integral-number normalization in a JSON intermediary. Existing worker,
source and activation-time checks must remain mandatory.

Expected signal: one local integration run using the real log reader,
supervisor, queue client and profile hook. It should traverse delayed startup,
the JSON wire boundary, two warmups, four timing batches, telemetry and cleanup
without manual intervention. A scripted model supplies responses; this is a
component integration check, never GPU performance or answer-quality evidence.

Decision rule: accept only if both normal and failed requests retain diagnostics,
no foreign/stale/invalid log can authorize dispatch, redirects and oversized
streams fail, requests remain capped at ten plus nine telemetry jobs, and the
original deadline and shutdown checks remain effective. Run focused checks
after each change and the full relevant suite before finishing. Live behavior
after these repairs remains unverified until a separately authorized run.

## Implemented and verified

`7596653` replaces the profile's CLI startup-log dependency with a fixed-host,
bounded native HTTP/SSE reader. The existing source/time/worker checks and
supervisor deadline remain in force. `45a815a` exercises the complete automatic
local flow through a real child process and JSON queue, including numeric
normalization, all 19 jobs, an extra-request block, rejection before dispatch,
failure retention and cleanup. These are scripted component checks, not a model
evaluation or GPU benchmark.

An additional loopback run used the exact saved payloads and pinned cached
tokenizer. Raw hashes and message content stayed unchanged; input counts were
1400 and 2043, and all ten scripted requests/nine telemetry jobs completed.
Saved startup logs also passed the new reader and existing verification logic.

Focused results: 51 passed (13.33 seconds), then 3 full-wire checks passed
(11.43 seconds). Full suite: 1903 passed, no skips, three existing SWIG warnings,
107.59 seconds. Ruff and whitespace checks passed. All protected files and
prior live-attempt artifacts/markers are unchanged. No activation or inference
occurred; final read-only endpoint checks show 0/0 and zero workers/pending jobs.

The [repair evidence](../../examples/two-reader-profile/automation-repair/README.md)
records scope, source hashes and limitations. Live behavior after this repair
and the original rejected-input transformation remain unverified.
