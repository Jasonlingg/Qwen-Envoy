# One authorized two-reader retry

The user approved one fresh live attempt after the startup-log repair. This
replaces only the October 21 profile attempt; preserve its plan, outcome and
consumed marker. Bind the new authorization to their hashes and consume a
separate fixed replacement marker. No additional replacement is authorized.

Hypothesis, expected signal and decision rule remain those in the original
two-reader diagnostic: same two fixed inputs, two rounds per mode, matched
outputs/usage and engine overlap. Missing telemetry or changed outputs make
the comparison inconclusive; differences within 400 ms cannot establish speedup.
This does not measure reader quality, reactions to feedback or model improvement.

One A5000, absolute 900-second activation window including loading, ten model
requests and nine telemetry jobs, 81920 reserved tokens, $0.50 local allowance.
Fresh catalog read: $0.69/GPU-hour, maximum window compute estimate $0.1725;
this is not a provider dollar cap or invoice. Keep serving settings unchanged.
Use repaired log polling, private bounded diagnostics, original watchdog and
shutdown/restoration verification. Do not retry uncertain inference requests.

## Live observation and subsequent offline hypothesis

The worker loaded the pinned model, became vLLM-healthy and passed fitness
checks. MCP tail reads retained these records while supervised CLI reads
continued to time out. Direct CLI tail and since comparisons sometimes succeeded;
this does not establish a deterministic timestamp-format defect. An explicit
operator handoff verified the same worker with the existing identity/readiness
functions and ran the existing worker phase within the original activation,
absolute deadline and request allowance. The original supervisor retained
shutdown/cleanup ownership. Startup automation did not pass.

One telemetry preflight completed. The first warmup request failed with a worker
input-pin rejection, before forwarding to vLLM. No request was retried and no
timing batch completed. Shutdown and entrypoint restoration were verified.

Hypothesis for offline repair: cross-language JSON serialization may turn the
saved `temperature=0.0`, `top_p=1.0` into integral numbers. The original hash
distinguishes these equivalent values. Expected signal: a local numeric
round-trip reproduces rejection, while a separate transport hash accepts safe
integral-number normalization and still rejects changed strings, booleans or
decoding values. Preserve original byte/canonical input pins. Decision rule:
accept this narrow repair only with those regressions; the received live input
was not captured, so do not claim normalization is its proven cause.

Also preserve terminal FAILED job responses before raising, and use bounded
tail retrieval with the existing strict client-side worker/time filters.
These repairs do not authorize another paid request or activation.

## Final verification

Live code `971932c` passed 33 focused checks and 1879 full-suite tests before
activation. Offline repairs `43097e1` and `ae8a874` passed 45 focused checks and
1882 full-suite tests in 80.37 seconds, with no skips and three existing SWIG
warnings. Ruff and whitespace checks passed. The original attempt artifacts,
consumed marker and all eleven protected file hashes remain unchanged.

The [retry record](../../examples/two-reader-profile/retry-1/README.md) includes
selected startup evidence, the explicit manual handoff, failed job, zero timing
rounds and independent endpoint restoration/shutdown reads. No further paid
requests were made, and no serving-speed or model-quality improvement is claimed.
