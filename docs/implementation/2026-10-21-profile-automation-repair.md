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
