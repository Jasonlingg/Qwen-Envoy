# Five practical project decisions

This is a small development walkthrough, not another automatic judging system.
The user approved checking practical usefulness before RL. Use the five
project-derived questions in questions.json; hardware/load numbers are prospective
constraints, not measured local results. Questions were offered to the user for
substitution; absent a substitute, proceed with the proposed five. Existing
reader evaluation references and reserved papers stay outside this work.

Hypothesis: using the research MCP gives the main assistant useful, checkable
paper evidence for these decisions. The expected signal is a concrete decision
brief backed by applicable sources, with material unsupported claims identified.
Review all five pairs; retain problems and missing results. Continue product work
only if the walkthrough reveals useful evidence/recommendations; report any
advantage as qualitative development observations pending user review. No training
promotion, new benchmark score, scalar reward or statistical superiority follows.

A uses ordinary web research. B uses the same native host/settings and ordinary
web access, but is explicitly instructed to use the source MCP for paper discovery
and passage inspection. This is a **prescribed source-tool comparison**, not a
measurement of natural MCP adoption. Qwen workers and the server reasoner are off.
It cannot measure Qwen reader cost/quality or base-versus-fine-tuned improvement.
B may use web for official implementation/model docs and failed discovery fallback;
record failures and whether source MCP actually returned relevant evidence.

Reuse the existing isolated native-session collector: same host/model/reasoning,
new workspace and empty MCP cache per answer, one attempt, alternating arm order.
Keep answers isolated, with a500-word target and600-second local deadline. Research
calls have the previously disclosed monitored target20; record overshoot. Native
host token totals are reported where available; actual host request counts/dollar
charges are unknown, not zero. No extra paid model endpoint, GPU launch or proposed
ML experiment is authorized. Full local event traces remain in ignored out/; export
verbatim answers, tool/usage/time receipts and a short paired summary.

Review is an ordinary assistant source audit, clearly attributed and unblinded,
plus a simple blank user preference sheet. It is not a separate automatic judge
or independent human judgment. For each question, inspect the recommendation and
its central cited claims; note unsupported assertions, applicability, unknowns,
latency, tool use and cost coverage. Leave user usefulness/preference unknown until
supplied. Do not ask a model to fill human review fields.

The earlier October19/20 five-question A/B run had zero MCP use in B; it is
preserved and is not rerun or relabeled. This new set includes concurrency memory
and actual project failure observations. The source cutoff stays2026-10-19 to
reuse matched collector instructions without changing historical frozen code.
Native protocol guidance: https://learn.chatgpt.com/docs/app-server .
