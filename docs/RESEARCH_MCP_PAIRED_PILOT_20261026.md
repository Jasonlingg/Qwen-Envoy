# Paired research-decision pilot — protocol fixed before runs

## Objective

Test whether the personal research MCP helps an assistant make and later recover
better-supported ML experiment decisions than ordinary web research plus local
Markdown notes. This is a five-question pilot, not a broad productivity study.
The data-mixing question used to build the demo is excluded.

## Common context and five held-out questions

Both arms receive this context: a Qwen3-8B code-execution paper reader has only
35 replay-verified training trajectories. On an earlier 40-question development
comparison, v5 SFT reduced episodes with recorded execution errors from 22 to
1, but provisional supported-answer passes on the 20 answerable cases remained
3 for both base and v5. The next decision should improve source-supported
answers within a small compute and annotation budget. Those numbers are
development evidence, not a held-out model-quality result.

1. **Training method:** With no new GPU budget yet, should the next investment
   be more supervised examples or another small GRPO run?
2. **Retrieval:** For long research papers, should the reader keep BM25 passage
   search or test hybrid retrieval plus reranking?
3. **Serving:** For two concurrent Qwen3-8B reader requests, should the next
   latency test focus on prefix caching or speculative decoding?
4. **Context:** For paper QA, should the reader receive an entire long paper or
   iteratively search and inspect selected passages?
5. **Training data:** With capacity to review about 100 more trajectories,
   should the team prioritize teacher-generated traces or manually corrected
   failure cases?

## Arms and budget

Use separate fresh assistant contexts with the same model, task text and source
budget. Arm W may browse the web and save ordinary Markdown notes. Arm M may use
the local research MCP's discovery, source and experiment-record tools, then
save its returned IDs and a minimal local export. Neither arm may read the
existing data-mixing demo or the other's files. For each question, allow at most
three search queries and three opened primary-source papers/docs. No paid model
API calls, GPU launches or training. Record start/end UTC, queries, sources,
tool calls, answer, atomic source-linked claims, simpler baseline, cheapest
matched test, decision rule, and uncertainty. Each arm must leave durable notes
for a later session. A fresh assistant context later receives only the five
questions and its arm's saved-note location and must recover the decisions.

## Review and decision rule

Normalize answers to the same shape, hide arm labels, randomize A/B order per
question with a recorded seed, and ask a separate reviewer to inspect primary
source context. Score each atomic factual claim 0 unsupported/contradicted,
1 partially supported or missing a named comparator, 2 directly supported.
Score the proposed experiment 0 if baseline or matched test is absent, 1 if
both exist but the acceptance threshold is unset, 2 if all three are present.
Recovery succeeds only if the fresh context retrieves the same source-backed
rationale and decision rule, not merely a similar conclusion. Report per-case
scores, severe errors, wall time, source/tool counts and model tokens if a
reliable meter is available. Never substitute exact quote matching for semantic
support or assistant review for independent human review.

A promising *pilot signal* requires no severe unsupported claim in M, better
decision recovery in at least three of five pairs, claim-support score no worse
in at least four of five, and median time and host-model tokens at most 20%
higher than W. If tokens cannot be measured comparably, mark that gate unknown.
Five cases and assistant review cannot establish a general advantage over
ordinary browsing, even if all gates pass. Report failures and incomplete cases.

## Results

Pending execution. The protocol above was committed before either arm ran.
