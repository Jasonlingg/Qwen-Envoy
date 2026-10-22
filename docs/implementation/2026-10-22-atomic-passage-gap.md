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
next query, then ends with no accepted answer. It launches no search and proves
absence only nowhere: the scope remains selected inspected passages. Retain
the inspected source observations and all earlier independent evidence. Do not
claim this adds substantive partial-findings synthesis or improves model weights.

Keep the four-request/32,768-reserved-token per-reader budget and all consumed
GPU attempt markers unchanged. No Runpod activation or paid calls are included.
