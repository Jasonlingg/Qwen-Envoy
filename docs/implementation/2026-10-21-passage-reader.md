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
