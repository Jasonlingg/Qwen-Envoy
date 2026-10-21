# Startup log repair — local verification

The October 21 attempt failed because the supervisor promoted three log-read
timeouts into a terminal startup error before its overall deadline. It never
submitted model inference. The implementation also discarded earlier log
observations and did not save returned log contents. These are confirmed
supervisor defects; no GPU hardware error was recorded.

The installed CLI's [pinned transport implementation](https://github.com/runpod/runpodctl/blob/dd55bcf/internal/api/logs.go)
classifies expiration before response headers as `timeout`. Runpod documents
that its log API can withhold headers until a matching log exists. The same
error can therefore reflect no matching output, slow startup, or a transport
failure; it does not identify a GPU timeout.

[cli-reproduction.json](cli-reproduction.json) records the actual installed CLI
against a local HTTP/SSE server with dummy credentials. Three requests that
receive no headers return `timeout`; a fourth receiving an identity frame
succeeds. The committed integration regression also runs the real CLI through
the supervisor: after three quiet reads, identity and two readiness events
arrive in three separate responses. The supervisor waits for all of them before
launching its scripted child. All fixture requests are loopback GETs; there is
no GPU, model or provider inference in this reproduction.

Commit `066d0c5` repairs the handling:

- A transient log error becomes a pending observation. The original absolute
  startup deadline owns retries; authentication and unknown errors still fail.
- Read both platform and container logs. Only current container records may
  establish model identity or engine readiness.
- Save bounded returned/partial output privately with mode 0600. Keep safe read
  metadata separate; incomplete JSON records cannot establish identity.
- Accumulate bounded, deduplicated evidence across polls for the same worker.
  Clear it across worker changes; stale/regressing state also invalidates older
  evidence. Inspect every identity record for a mismatch.
- Pin the changed lifecycle/control dependencies in newly prepared profile plans.
  Existing failed plans, artifacts and consumed attempt markers remain intact.

The [provider read](provider-read.json) found the old worker's platform logs
unavailable (`404 worker not found`). Consequently the original provider-side
reason for missing timely headers remains unknown. Neither this reproduction
nor the repair establishes that the original model loaded successfully or that
a future live attempt will succeed. The endpoint remains at 0/0, with zero
workers and no pending jobs. No paid retry was made.

The focused suite passed 67 tests in 8.54 seconds. The full suite passed 1878
tests in 77.40 seconds, with no skips and three existing SWIG deprecation
warnings. Ruff and whitespace checks passed. Commands and artifact hashes are
recorded in [validation.json](validation.json). No model
quality, concurrency speedup, MCP advantage or fine-tuning improvement was
measured. Human answer review is absent.
