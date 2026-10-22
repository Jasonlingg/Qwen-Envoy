# Atomic passage-reader gap completion

User approved fixing the gap workflow after both base and V5 failed the separate
report-gap / abstain procedure. Scope: a local controller/MCP change, not another
paid run, model training or reward change.

Hypothesis: after two successful distinct passage inspections, one terminal
`submit_gap(missing=..., follow_up_query=...)` removes the unnecessary ordering
trap while retaining source bounds, inspection requirements and an explicit gap.
Expected signal: deterministic read/read/gap completion in three actions and
recovery from the recorded premature-abstention pattern within four actions.
Decision rule: pass positive and adversarial controller/verifier checks plus a
real MCP/Docker scripted path with saved notes and no redispatch on identical
retry. Early, malformed, mixed-code, uninspected or forged terminal gaps must
fail. These checks establish behavior of the harness, not live Qwen response
quality or semantic absence. Independent human review remains blank.

Add `passages-v2` as an explicit protocol version. Preserve the existing V1 tool
preamble and all legacy baseline/focused/passages-v1 prompt bytes. The optional
experiment MCP uses v2; historical comparison runners retain v1. All authored
document Python stays in Docker; terminal control calls are parsed only as
bounded literal AST data. No evaluation of model-authored Python on the host.

A terminal gap records an unreviewed missing-evidence statement and proposed
next query, then ends with no accepted answer. It launches no search and makes
no verified claim of absence; the scope remains selected inspected passages. Retain
the inspected source observations and all earlier independent evidence. Do not
claim this adds substantive partial-findings synthesis or improves model weights.

Keep the four-request/32,768-reserved-token per-reader budget and all consumed
GPU attempt markers unchanged. No Runpod activation or paid calls are included.

## Implemented and verified locally

Scope commit `802ccec`; implementation `ed75991`. The optional experiment MCP
selects `passages-v2` while historical comparison runners retain v1. The old
two-read / separate-abstention pattern now receives a one-action recovery hint;
a valid final `submit_gap` atomically records the gap and ends the investigation.
Separate `report_gap` actions in v2 receive the same recovery direction. The
new effective prompt and Nemotron stop reminders remove the conflicting legacy
abstention instruction. Earlier baseline, focused and passage-v1 prompt bytes
are preserved: all twelve saved legacy prompt hashes match, as does the existing
V1 tool-preamble SHA-256.

Terminal calls require bounded literal strings, two distinct successful passage
inspections, an in-scope inspected ledger and a verified final trace. Rejected
calls cannot add gap records or inspection credit. Malformed ancillary gaps
still cannot discard independent verified quotes. MCP notes retain the missing
statement and the proposed next query explicitly marked unreviewed/not executed.
Same-request retry and service restart return the saved result without inference.

Focused checks: 90 passed in 63.67 seconds. Standalone full relevant suite:
1963 passed in 214.82 seconds, no skips, three existing SWIG warnings. Ruff and
diff checks passed. The real MCP/Docker integration test reproduces read/read/
premature-abstention/terminal-gap recovery in four actions; the ordinary path
completes in three. Negative checks reject early, duplicate/failed-read, nonliteral,
mixed-code, oversized, forged and nonterminal completion attempts. Qwen and
Nemotron policy/investigator prompt identities match; CLI setup makes no inference.

[Inspectable scripted report, persisted note and validation](../../examples/atomic-passage-gap/README.md).
Both the source paper and model responses in that example are synthetic test
fixtures. No post-change live Qwen test, paid model request, GPU activation,
training or reward change occurred. Eleven protected files and 637 prior
artifact/marker hashes remained unchanged. This verifies the harness contract,
not semantic absence, successful Qwen adoption or model-quality improvement.
