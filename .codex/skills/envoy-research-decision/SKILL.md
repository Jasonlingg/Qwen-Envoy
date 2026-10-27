---
name: envoy-research-decision
description: Use for an ML experiment decision that needs paper evidence, a prior result, or a durable decision record in Qwen Envoy. Skip ordinary coding and one-off factual lookups.
---

# Research-backed experiment decisions

Use the connected `envoy-research` MCP when a decision about trying, changing,
or revisiting an ML experiment should survive this conversation. The skill
guides the assistant; the MCP owns the source and record checks. Do not call it
on every turn.

1. Check for a relevant prior attempt with `find_related_experiments` or
   `find_decision_threads`; inspect promising IDs before starting new research.
2. State the current workload, candidate change, simpler baseline, constraints,
   and what remains unknown. Search and open primary sources as needed. Use
   `get_passage` for the exact text behind material paper claims, and look for a
   relevant limitation or alternative rather than treating papers as votes.
3. Save the question and source-linked findings through the experiment tools
   when the user is making a material decision. Keep reported paper results,
   local measurements, and proposed transfer to this workload distinct. A
   verified quote proves provenance, not semantic support.
4. Propose the cheapest matched test with an acceptance rule and cost/error
   ceilings. Freeze a plan only when a concrete test has been selected. Attach
   measurements only from an actual supplied run or artifact. Mark an
   assistant-authored adopt/reject/defer decision as `agent_proposed`; never
   present it as user-approved or overwrite an earlier result.
5. Return a short recommendation with source IDs, the simpler baseline, the
   decisive uncertainty, the next test, and the experiment/decision IDs saved.
   If evidence is insufficient, say what is missing and defer the claim.

The base workflow needs no model endpoint. Do not call optional Qwen or
Nebius/Nemotron tools merely because they are available; a model request needs
the user's task to warrant it and its own resource limits. Exact source checks
and citation IDs never substitute for claim-level review.

If the `envoy-research` MCP is not connected, use ordinary research tools as
appropriate and state that no Envoy record was saved. For connection commands
and tool details, read [the MCP guide](../../../docs/RESEARCH_JUSTIFICATIONS.md)
and [experiment workflow](../../../docs/EXPERIMENT_MODE.md).
