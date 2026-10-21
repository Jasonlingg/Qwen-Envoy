# Experiment paper discovery repair

The experiment MCP now uses a separate arXiv relevance fallback when Semantic
Scholar is unavailable. In the revised source check, the original PagedAttention
paper ranked second, FlashAttention first and QLoRA third; an invented topic
returned no candidates. This repairs a tested discovery failure, while useful
Qwen research remains unestablished. No GPU, model inference or training ran.

## Cause and implementation

The failed research smoke passed plain query text into a dated discovery path
sorted by submission date. Its fallback supplied unrelated recent metadata.
The [arXiv API manual](https://info.arxiv.org/help/api/user-manual.html)
documents explicit field terms, Boolean composition and relevance sorting.

Commit `a6063f8` adds `PaperDiscovery.search_arxiv_relevance`, used only by
`ExperimentDiscovery` after primary-provider failure. Existing default Semantic
Scholar search and explicitly dated arXiv recency search retain their behavior.
No trained policy, reward, sandbox or serving configuration changed.

The new fallback treats input as plain text. It constructs quoted title/abstract
terms with explicit OR operators, including generic joined adjacent words to
match spellings such as Flash Attention and FlashAttention. It requests at most
50 relevance-ranked metadata candidates through the existing redirect-checked,
size-bounded HTTPS transport and process-wide arXiv throttle. There are no model
calls or automatic transport retries. New cache keys distinguish this path and
ranking version from the old recency cache.

A multi-term query requires at least two matched terms; one-term queries require
that term. Results prioritize the leading two-word phrase or compound in titles,
then term coverage, title coverage and provider rank. This lexical heuristic can
miss paraphrases and favor the wrong topic when query wording is poor. It does
not evaluate semantic relevance or support. Empty results carry `no_candidates`
and refinement guidance into the controller observation and saved trace.

Independent review found that unrestricted relevance search can include valid
legacy arXiv identifiers unsupported by our reader. Those entries now skip
individually with their IDs disclosed, preserving modern candidates from the
same feed. Malformed entries and provider error feeds still fail closed. A second
review closed the finding with no remaining material issue.

## Source check and decision

The [initial plan](../../examples/experiment-discovery-repair/plan.json) fixed two
recorded failed queries, FlashAttention, QLoRA and an invented negative query.
The hypothesis was that explicit title/abstract queries and relevance ranking
would recover useful candidates after a primary-provider failure. The gate
required all three named foundational papers in the top five and no candidates
for the invented topic; the shorter recorded query required separate metadata
inspection.

Each check injected Semantic Scholar's rate-limit failure locally and fetched
real arXiv responses. No actual Semantic Scholar request occurred. Across the
initial and revised checks, ten logical arXiv reads were made; existing per-hop
redirect and throttle limits still applied. Raw Atom responses, request URLs,
results and caches are retained separately for both checks.

The first check missed PagedAttention in the top five although the provider
returned it at rank eight. Local ranking favored additional incidental query
words. The [explicit amendment](../../examples/experiment-discovery-repair/amendment.json)
records the subsequent title-phrase priority and legacy-parser repair, five
additional source reads, and the unchanged decision rule. Initial results were
not overwritten. The leading-phrase regression uses a generic synthetic topic;
implementation contains no fixed method names or paper IDs.

| Query | Revised result |
| --- | --- |
| Recorded PagedAttention / KV-cache / concurrent-reader query | Original paper 2309.06180 at rank 2 |
| Recorded KV-cache allocation / concurrent-reader query | Five KV-cache allocation/budgeting candidates |
| FlashAttention exact attention IO-aware | Original paper 2205.14135 at rank 1 |
| QLoRA quantization memory | Original paper 2305.14314 at rank 3 |
| Invented topic | `no_candidates` |

The shorter query's returned titles and abstracts concern KV-cache allocation and
memory budgeting. They do not establish memory requirements for two Qwen readers
on our hardware. This was implementing-assistant metadata inspection, not human
semantic validation. The revised discovery gate passed on these development
queries after inspecting the initial results; it is not a held-out comparison or
an improvement score for Qwen. Both checks are in
[the source-check summary](../../examples/experiment-discovery-repair/summary.json).

## Verification and remaining work

Focused verification passed **77 tests** covering query construction, the saved
bad-result scenario, generic compound names, empty results, cache separation,
legacy IDs, malformed feeds, network controls and controller feedback. The full
offline repository suite passed **1,773 tests, zero skipped**, with three existing
SWIG deprecation warnings in 70.70 seconds. Ruff and whitespace checks passed.
All ten original protected file hashes remain unchanged.

[Validation and artifact hashes](../../examples/experiment-discovery-repair/validation.json)
record the tested implementation and exact commands. The saved source-check
runner is an execution snapshot with a fixed output location; use a fresh output
directory for a new check rather than overwriting these results.

The MCP user can request papers through the experiment workflow and receive
bounded metadata candidates with a disclosed fallback or discovery gap. The
failed Qwen research smoke has not been rerun. Next, review a fixed set of tasks
and required supported facts before choosing a prompt/data or RL intervention.
Any future paid model test needs its own explicit approval and bounds.
