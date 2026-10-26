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

The protocol above was committed as `092a647` before either arm ran. The
[artifact bundle](../examples/research-mcp-paired-pilot-20261026/README.md)
contains the arm outputs, Markdown notes, MCP record exports, anonymized review
inputs, blind assistant reviews, unblinding map, call events and
[machine-readable scores](../examples/research-mcp-paired-pilot-20261026/summary.json).
W is ordinary web research plus Markdown notes; M is the research MCP. Claim
scores below are the mean of each arm's *own* reviewed atomic claims (0–2), so
the arms have different denominators. Decision scores use the predeclared 0–2
rubric. Both recovery tests used fresh assistant contexts.

| Question | W claims | M claims | W/M decision | W/M recovery | M claim no worse? |
| --- | ---: | ---: | ---: | --- | --- |
| 1. SFT vs GRPO | 2.00 (6) | 2.00 (2) | 2 / 1 | yes / yes | yes |
| 2. BM25 vs hybrid | 2.00 (4) | 1.50 (2) | 2 / 2 | yes / yes | no |
| 3. Prefix cache vs speculation | 2.00 (3) | 2.00 (2) | 2 / 2 | yes / yes | yes |
| 4. Whole paper vs selected passages | 1.67 (3) | 1.50 (2) | 2 / 2 | yes / yes | no |
| 5. Teacher vs corrected traces | 2.00 (3) | 2.00 (2) | 2 / 2 | yes / yes | yes |

Across the five cases, W earned **37/38 claim points** and **10/10 decision
points**; M earned **18/20 claim points** and **9/10 decision points**. There
were no score-0 unsupported claims in either arm. M was no worse on claim
support in **3/5** pairs, below the 4/5 gate. Each arm recovered **5/5**
decisions; M was better in **0/5**, below the 3/5 gate. The separate recovery
review flagged extra, unverified details in some recovered answers. Recovery
success means the original rationale and rule were retained; it does not certify
those additions. The **predeclared pilot signal did not pass**.

W used 15 searches and opened 15 primary sources after replacing question 1.
M used 15 searches, opened 10 papers and made 52 MCP calls. M's caller made
ten Python SDK attribute errors while parsing search responses (`structuredContent`
and `isError` instead of the SDK's snake_case fields). The searches executed
and count against its budget; **zero MCP server errors** were observed. This is
a real workflow reliability failure, but it does not show that the MCP search
tool itself failed. M also repeated the same query three times per question,
reducing search diversity. Two M notes needed append-only corrections to
overbroad source wording.

The original W1 exceeded the cap with four searches; it remains in the bundle
for audit and a fresh, compliant three-search W1 is used for all primary scores
and recovery. W2–W5 were timed as a batch, while M question windows overlap.
The W batch took 122 seconds, the replacement W1 77 seconds, and M took 250.2
seconds overall. Even 199 seconds, a conservative sum of both W runs that
includes the excluded W1, is shorter than M, but **comparable per-question
median latency cannot be calculated**. Host-model tokens were not comparably
metered, so the token/cost gate is unknown. No paid inference, GPU, or training
was used.

An independent assistant, blind to arm labels, checked 29 atomic claims
against source context and reviewed the proposed tests. A different fresh
assistant checked recovery. **Neither was an independent human reviewer**;
they did not validate local project measurements, and some qualitative source
checks relied on abstracts. The five cases are a small pilot, not an estimate
of general research quality. The demonstrated MCP benefit is a structured,
source-linked and recoverable record, while this paired test gives **no evidence
that it improves decisions over web research with ordinary Markdown notes**.
