# Local identity-log repair — October 20, 2026

Commit `9580aad` repairs two startup hazards without activating a GPU or sending
model requests. The installed Runpod CLI passed local HTTP/SSE integration tests.
Live identity-log access remains unverified; there are still no Qwen answers from
the frozen two-case assigned-reader smoke.

## Findings and changes

The supervisor previously obtained worker IDs from the legacy endpoint response.
The installed CLI's pinned [v2 API source](https://github.com/runpod/runpodctl/blob/dd55bcf/internal/api/restv2.go)
warns that this legacy worker expansion is unreliable for active worker identity.
Its [log command](https://github.com/runpod/runpodctl/blob/dd55bcf/cmd/serverless/logs.go)
uses the v2 worker list. The supervisor now uses that same canonical list during
preflight, startup and shutdown, validates its shape/counts, excludes stale
workers from readiness, and retains sanitized worker IDs for diagnosis.

Worker discovery uses a fixed HTTPS host, the existing bounded HTTP routine,
and no redirects, proxies, queue submission or inference retries. Environment
fields from endpoint and worker responses are excluded from persisted records.
The existing pinned model/adapter/GPU identity check is retained before inference.

The old one-second log read can expire before a valid SSE response arrives. A
local server delaying its first response by 1.3 seconds reproduced a structured
`timeout` with the actual installed `runpodctl 2.14.0-dd55bcf`; the same response
succeeded with five seconds. Reads now allow five seconds and at most three
attempts for the named transient error codes, including timeout. Authentication
and unknown errors still fail immediately. Each attempt records only safe worker,
time and command/code/status fields. The original watchdog, exclusive attempt
markers, request/token limits and shutdown checks remain in force.

These are confirmed integration hazards, not established causes of either prior
live failure. The earlier attempts did not persist their active worker IDs and
the second attempt discarded its structured active-time CLI error. Their saved
records and interpretations remain unchanged:
[first attempt](2026-10-20-assigned-reader-smoke-attempt.md),
[replacement](2026-10-20-assigned-reader-smoke-replacement.md).

## Verification and actual results

The **65 focused tests passed** in 6.10 seconds. They cover canonical worker
binding, rejection of stale/duplicate/malformed records, secret exclusion,
shutdown, existing approval and request budgets, and the real CLI against a local
HTTP/SSE fixture. The local fixture uses a dummy key and loopback API overrides;
it does not access a GPU or submit inference. The full offline suite passed
**1,637 tests**, no skips, with three existing SWIG deprecation warnings in
92.68 seconds. Ruff and whitespace checks passed. The cached preparation check
still reports the same two cases, initial token counts 1,938/1,964 and frozen
limits, with zero model calls.

A read-only HTTPS request to the actual v2 worker-list endpoint succeeded and
returned zero workers. Subsequent independent MCP reads returned worker min/max
0/0, zero active workers, all health worker counts zero and no queued/in-progress
jobs. This verifies the parked state and worker-list access, not live log access.
No activation, inference, live answer evaluation or benchmark was performed in
this repair. The two prior one-attempt allowances remain exhausted and their
markers are intact. Actual prior billed spend remains unknown.

Independent code review found no actionable issue in worker binding, credential
destination, identity checks, retries or cleanup. All ten protected file hashes
and both prior published attempt artifact sets are unchanged. Commands, source
hashes and sanitized read-back records are in the
[validation artifact](../../examples/assigned-reader-log-access-repair/validation.json).

This improves startup diagnostics and local lifecycle reliability. It provides no
measurement of model answer quality, execution-error rate, fine-tuning benefit,
or advantage over a larger assistant. No human answer review was performed.
