# Qwen evidence-card probe — 2026-10-26

This is an exposed, three-paper development probe for a possible Qwen role in
the personal research MCP. It does not evaluate the full MCP workflow, a held-out
set, or a model improvement over base Qwen. The frozen inputs and raw outputs
are saved in `examples/qwen-evidence-card-probe-20261026/`.

## Hypothesis and decision rule

The hypothesis was that either the unchanged V5 SFT adapter or the saved RL
step-9 adapter could turn a supplied arXiv passage into a useful evidence card
for the question, “What evidence supports adding domain mixing to
quality-filtered web data for language-model pretraining?” Before inference,
`examples/qwen-evidence-card-probe-20261026/protocol.json` set the following
gate: all three cards for an arm must have a
valid schema and exact quote, at least two must pass claim-level support, and
both comparator papers must say that they do not directly test the proposed
addition. Passing would permit a **draft-only reader**, not autonomous research
judgment.

The three exposed papers were WebOrganizer (`2502.10341v3`), DCLM
(`2406.11794v4`), and FineWeb (`2406.17557v2`). Their source-file SHA-256 values,
exact supplied passages, and prompt hash are in `protocol.json` and
`bundle/cases.json`. The input SHA-256 is
`135f346a30b31d84f9a1f7a4beaef81e14f8e779ca99d7d8b867ce2259082505`.
The prompt asked for JSON fields `method`, `comparator`, `measured_outcome`,
`limitation`, and an exact contiguous `quote`, with missing measurements
explicitly marked.

## Execution

`scripts/eval_qwen_evidence_cards.py` ran the same prompt against Qwen3-8B at
base revision `b968826d9c46dd6066d109eabc6255188de91218`. The V5 adapter
SHA-256 was `7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3`;
the locally saved RL step-9 adapter SHA-256 was
`a2a773caf222aa34becbeb2f886ce553c0aefa0871783fa17efe8da0557ca810`.
The run used an NVIDIA A40, PyTorch `2.8.0+cu128`, Transformers `4.57.6`,
PEFT `0.21.2`, Accelerate `1.15.0`, seed 42, greedy decoding,
`enable_thinking=False`, and a limit of 320 generated tokens per card. Both
adapters shared one loaded base model. The temporary Runpod pod
`iswv664nk62a91` was deleted after the raw outputs were copied; a subsequent
get returned 404. The listed rate was $0.59/hour. Billing had not posted when
queried, so an actual charge is unavailable.

| Paper | V5 schema / exact quote | RL9 schema / exact quote | Claim review |
| --- | --- | --- | --- |
| WebOrganizer | pass / pass | pass / pass | The cited passage directly reports FineWeb-Edu average accuracy increasing from 54.2% to 56.2% with domain mixing. Method, comparator, and outcome are supported. The limitation field is vague. |
| DCLM | pass / pass | pass / pass | The passage supports model-based filtering and 64% five-shot MMLU, and the card correctly says it is not a direct domain-mixing test. Its selected quote only says the benchmark *permits* experiments on data mixing; it does not support the numerical result in the card. Quote-to-claim alignment fails. |
| FineWeb | pass / pass | pass / pass | The card correctly disclaims a direct domain-mixing experiment, but leaves method and outcome as “not reported” even though the passage describes FineWeb-Edu filtering and qualitatively better MMLU/ARC performance. It omits useful evidence. |

All six raw card bodies were identical between V5 and RL9, including the
omissions. The rule's direct-comparison check passed for both comparator papers,
but only the WebOrganizer card passes the stricter claim-level review as a
useful cited card. **Neither adapter passes the predeclared draft-reader gate.**
Exact substring matching proves only that a quote came from the supplied text;
it does not prove that the quote supports every field in the card.

## Decision

Keep Qwen outside the automatic evidence-card path for now. It can be used for
manually reviewed draft extraction, with the original passage shown beside the
card. A next prompt/validator iteration should require separate quoted support
for the method and each reported result, preserve useful qualitative outcomes,
and explicitly identify missing direct comparisons. Test that change on more
questions and papers before opting it into the MCP. This probe does not show an
RL gain over V5; the earlier paired QASPER evaluation was also inconclusive.
