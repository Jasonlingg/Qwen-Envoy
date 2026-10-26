# Research decision workflow evaluation — data mixing

This is a predeclared, single-case component evaluation of the personal MCP,
not a comparison against ordinary web browsing or a Qwen training result.

## Task and hypothesis

Question: Given WebOrganizer (arXiv:2502.10341) as the lead paper, should an ML
engineer add domain mixing to quality-filtered Qwen post-training data? The
engineer wants a cheaper baseline and a test before spending training compute.
The user's report of an author's private comment is context, not paper evidence.

Hypothesis: the MCP helps preserve an auditable, cautious experiment decision
across sessions. The expected signal is a packet with versioned paper spans,
at least one relevant quality-curation alternative found by source search, a
simpler baseline, transfer limits, and a matched test that survives a new
server process.

## Procedure fixed before this run

1. Search with the MCP for alternatives using at most three queries about
   quality filtering and web-text pretraining. Record candidate IDs and rank;
   the lead paper may be opened by its supplied ID. Discovery passes if one
   substantively relevant alternative appears in the top ten of a query.
2. Run `scripts/demo_research_justification.py` against public versioned papers.
   This demonstration uses selected known paper IDs, so it does not test
   autonomous paper selection. Save its report and tool-call failures.
3. Read the selected span in its paper context and mark each narrow attributed
   finding `supported`, `overstated`, or `unclear`. This is implementing-assistant
   review, not independent human review. Exact byte matching alone never earns
   a semantic-support pass.
4. Retrieve the same packet through a fresh stdio MCP process using its saved
   record root. Compare parsed packet JSON and check that source hashes and
   versions remain visible.
5. Check whether the packet names a simpler baseline, a matched test, limits on
   transfer to Qwen SFT, and a numerical acceptance threshold set **before**
   training. Without that threshold, mark it useful for planning but not ready
   to authorize compute.

The workflow gate passes only if discovery, all three narrow claim reviews,
receipt integrity, restart equality, and baseline/test/transfer fields pass,
with no model or GPU request. Training-readiness passes only if the numerical
threshold is also present. Any unsupported claim or source mismatch fails the
workflow gate. A single case cannot establish a productivity or answer-quality
gain over browsing.

## Follow-up comparison required for a product claim

Use at least five new ML experiment decisions not used to build the demo.
Randomize whether the same host assistant works with ordinary web browsing or
web browsing plus this MCP; give both the same time, source and model budget.
The web-only arm may save ordinary Markdown notes, so the memory comparison is
against a reasonable low-tech alternative, not an artificially amnesiac chat.
Capture source links, paper versions, claims, proposed baseline, decision rule,
elapsed time and tool/model cost. After a fresh session, ask each arm to recover
the prior decision. Have a reviewer blind to arm check claim support against
paper context, missing counterevidence, actionability and recovery. Report
per-question results and paired differences, including regressions. Do not
count exact quotes as support or an assistant's own review as independent.
Score each atomic paper claim 0 if unsupported or contradicted, 1 if partly
supported or missing its stated comparator, and 2 if directly supported in
context. Separately score the decision 0 if its baseline or test is absent, 1
if both are present but the acceptance threshold is unset, and 2 if all three
are specified. Recovery succeeds only when the later session retrieves the
same source-backed rationale and decision rule, not merely a similar summary.
For a promising pilot signal, require no severe unsupported claim in the MCP
arm, at least three of five better recovered decisions with MCP, no worse
mean claim-support score on four of five cases, and no more than 20% higher
median time and host-model token use. Report the individual cases even if this
gate passes; five tasks are not enough for a broad productivity claim.

## Results

The [discovery record](../examples/research-decision-eval-20261026/discovery.json)
contains two actual `search_papers` calls. Semantic Scholar rate-limited both,
so the MCP fell back to arXiv relevance search. FineWeb appeared at rank 7 of
the second query. Search is not exhaustive; neither query surfaced DCLM in its
top ten. The subsequent [packet](../examples/research-decision-eval-20261026/packet.json)
uses three known paper IDs and completed nine MCP calls with zero errors,
model requests or GPU activations.

All three selected quotes match their saved offsets in cached versioned paper
text. The packet now exposes those offsets alongside the source hashes so a
reader can inspect the exact span. A fresh
stdio server, pointed at the saved record and an **empty** source cache,
returned identical packet JSON, SHA-256
`385bfe0c89ac08c0db7a9719102657fceb3cac6751aa8ab474374c64aea3c0d3`.
The [machine checks](../examples/research-decision-eval-20261026/results.json)
and [claim-by-claim assistant review](../examples/research-decision-eval-20261026/claim_review.json)
are saved separately.

The three *narrow* attributed findings are supported by their selected text.
The implementing assistant scores them 2/2 each under the narrow-claim rubric;
the decision is 1/2 because its acceptance threshold is unset. These scores
are provisional and do not change `semantic_review: not_reviewed` in the MCP.
Review exposed a real defect in the example: its DCLM and FineWeb comparator
fields originally named baselines the selected quotes did not identify. The
demo now explicitly says the comparator is unspecified by those passages;
the packet does not imply that those papers directly beat WebOrganizer mixing.
The `qualifies` labels and independence notes remain interpretations needing
human review. All three results are about pretraining, not this Qwen SFT task.

**Outcome:** the revised single-case workflow gate passes under implementing-
assistant review. Training-readiness **fails** because the decision rule has no
numerical acceptance threshold agreed before compute. The proper decision is
to keep quality filtering as the baseline and specify the held-out split,
minimum worthwhile gain and error/cost ceilings before a matched training
comparison. No Qwen gain, productivity advantage or superiority to ordinary
web browsing has been measured here.
