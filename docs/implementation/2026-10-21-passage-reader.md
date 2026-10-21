# Passage-focused Qwen reader

User instruction: “ok lets try then”, after the proposal to give Qwen relevant
passages and test three exposed development examples. Implement the optional
MCP connection and verify locally before a bounded live attempt. No RL, reward
changes, reserved evaluation access or new benchmark framework.

Hypothesis: removing discovery and within-paper navigation lets the existing
v5 checkpoint return useful facts with sufficient evidence and correct conditions.
The host selects the paper and windows; Qwen must inspect them through document
tools, then submit supported facts or a scoped gap. Code stays in Docker.

Expected signal: three existing development cases (QLoRA dtype facts, AWQ
benchmark conditions, PagedAttention target-measurement gap), at most four
actions each. Save each result or failure in its experiment through the combined
MCP, with source provenance and semantic review clearly separate.

Decision rule: report every case. A useful demonstration needs both answerable
cases to supply the explicitly requested facts with adequate citations and the
gap case to avoid inventing the target measurement. An implementation-assistant
claim review checks meaning and conditions; human review remains blank. Three
familiar examples cannot demonstrate generalization, training improvement or
lower cost than host-only research. A host re-reading these examples for review
is not measured effort savings.

Keep baseline/V1 prompt bytes unchanged. The opt-in passage prompt must be
identical in policy, investigator and verification hashes. Reject reads outside
the assigned windows before Docker execution; reject citations outside actually
inspected windows. Metadata assignment is not source inspection. Retry of the
same MCP note ID must not dispatch another model call; uncertain attempts stay
consumed, and a stale result must remain recoverable without automatic inference.

Live scope, if safely run: one existing A5000, 900 seconds including loading,
twelve aggregate model requests, 98,304 reserved input/output tokens, no
inference resubmissions, one exclusive attempt marker and verified shutdown.
Refresh price and parked state before activation. Use the native bounded startup
reader; no manual readiness handoff. A $0.50 local allowance is a time/request
guard, not a provider invoice cap. No other live attempt is included.

## Actual result

`fe3a5a5` implements the opt-in passage scope, Docker-only reads, inspected-span
citation gate, durable single-attempt research notes, optional combined MCP tool
and bounded runner. `0b6bdda` records the prepared inputs before activation.
Before activation, 30 focused tests passed in 8.70 seconds and the full suite
passed 1918 tests in 130.82 seconds, no skips and three existing SWIG warnings.

The live attempt completed ten requests with 26,505 input and 421 output tokens.
AWQ produced a complete answer supported by its own citation: prompt 4 tokens,
generation 200 tokens, batch 1, median latency, RTX 4090 and Jetson Orin. QLoRA
and the target-gap case remained incomplete after four actions each. All reports
were retained and attached as notes; identical retries made no new model calls.

The three-case acceptance rule was not met. Qwen's valid end-minus-start passage
arguments hit the literal-only parser, consuming recovery turns. One replacement
also exceeded the allowed window and was correctly rejected. The gap case then
attempted premature abstention. There is no accepted gap and no answer to grade
for either incomplete case; these observations do not isolate reading ability.
The [source review](../../examples/passage-reader-three-case/claim-review.json)
is implementation-assistant checking, not independent human review or a reward.

Native startup succeeded without manual intervention. The watchdog did not fire;
supervisor time before cleanup was 184.782 seconds. Fresh reads at 22:14:21 UTC
verified workers 0/0, no workers/pending jobs and the full configuration restored.
Six owned rehearsal/live containers were independently checked absent. Invoice
cost is unknown. The allowance is consumed; no paid retry ran.

## Local repair after the attempt

`9161aab` parses bounded constant integer addition/subtraction as AST data and
normalizes the read into literal Python before Docker execution. It executes
no host Python. Exact read-call examples now include lengths. Authored and
normalized actions are retained; source bounds and inspected-span checks remain.
Tests reject calls, booleans, exponentiation, shifts and excessive integers.

Post-fix focused checks: 37 passed in 10.12 seconds, including real Docker with
the recorded arithmetic form. Final full suite: 1925 passed in 129.95 seconds,
no skips, three existing SWIG warnings. A new local scripted rehearsal completed
the same three cases; all such output is marked synthetic. The correction was
not tested with live Qwen, and scripted/fixed-action tests cannot predict its
responses to new feedback. The old plan, prompt and outputs remain frozen.

See the [passage-reader guide](../PASSAGE_READER.md) for explicit configuration
and [actual trial](../../examples/passage-reader-three-case/README.md) for every
case and limitation. One narrow extraction succeeded; reliable research quality,
reading-effort/cost benefit, training improvement and the broader autonomous
research workflow remain unproven. No RL, reward change or reserved evaluation
access occurred.
