# Three-case Qwen passage reader attempt

One explicitly requested development attempt: give v5 relevant source windows
and inspect whether it can return useful facts and conditions through MCP.
The host selects windows from already exposed source reviews. This is an easier
source-selection task than paper discovery or full-paper navigation, not held-out
evaluation or evidence that the fine-tune improves Qwen.

Cases: QLoRA storage/computation/gradients, AWQ benchmark protocol and hardware,
and the missing PagedAttention target latency measurement. Expected review facts
are separate from model prompts. Qwen receives the question, paper metadata and
window coordinates, then must inspect text in network-disabled Docker. The
opt-in `passages-v1` mode permits only contained passage reads and citations to
actually inspected source text. Quotes establish provenance, not semantic support.

One existing A5000, 900 seconds including loading, at most twelve model calls
and 98,304 reserved tokens. The serverless catalog at 22:01:47 UTC on October 21
reports $0.69/hour, giving a maximum-window compute estimate of $0.1725 under
the $0.50 local allowance. Actual invoice cost is unknown; no provider dollar
cap exists. Earlier consumed attempts remain unchanged. No paid retry is included.

Implementation `fe3a5a5`. Before activation, 30 focused tests passed in 8.70
seconds and the full suite passed 1918 in 130.82 seconds, with no skips and three
existing SWIG warnings. Ruff and whitespace checks passed. A scripted rehearsal
used the real MCP, Docker and exact source windows, completing all three cases
with nine synthetic model responses. That rehearsal is not a Qwen quality result.

The live attempt's case outputs, evidence, semantic source review and verified
shutdown will be recorded after completion. Success needs both answerable cases
to cover requested facts with adequate support and the gap case to avoid inventing
the target measurement. Host source review is model-assisted; human review stays
blank. No RL, reward change or research-productivity comparison is included.
