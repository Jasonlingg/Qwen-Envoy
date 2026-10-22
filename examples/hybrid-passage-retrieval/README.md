# Offline two-passage hybrid retrieval comparison

Plan and implementation were frozen before scoring. The comparison is complete;
the preselected reranked candidate did not meet the acceptance rule.
This evaluates passage selection for the existing multi-step Qwen code-execution
reader. It makes no Qwen calls, launches no GPU and changes no product defaults.

[Protocol and decision rule](../../docs/implementation/2026-10-22-hybrid-passage-retrieval.md).
Compare exact previous BM25 with dense, equal-weight RRF hybrid, and top-10 hybrid
cross-encoder reranking. Every arm returns two disjoint 1600-character windows,
3200 characters total, selected from the full known target paper. Candidates and
boundaries never use gold annotations. The exported proposed assignments retain
question, source ID, text SHA and exact offsets; retrieval is not source inspection.
Qwen would still need to inspect those passages through Docker before submitting.

Both `dev` and `test_candidate` were already exposed. Dev is the primary decision
set, test_candidate is a secondary development regression diagnostic. This is not
an untouched held-out test or official QASPER full-test score. Original QASPER
questions, annotations and S2ORC paper text come from AllenAI QASPER (Dasigi et al.,
2021), dataset revision `13b496d2a5359329b110e3419628de3cf791843b`, CC BY 4.0.
See [dataset provenance](../../benchmarks/envoybench/data/README.md).

CPU-only pinned models:
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf), 256-token inputs, normalized cosine embedding search.
- [MS MARCO MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a), 512-token query/passage cross encoder.

Both model cards list Apache-2.0. Model training exposure to these questions or
papers is unknown. Files and software are hashed/pinned in the plan. One public
reranker weight download was necessary; then evaluation is local-files-only with
HF/Transformers offline mode. No paper/question content is sent to remote inference.

After installing the project's vector/QASPER dependencies and caching those exact
model snapshots, reproduce with a new output directory:

```
python scripts/compare_passage_retrieval.py prepare --output out/my-local-retrieval
python scripts/compare_passage_retrieval.py run --output out/my-local-retrieval
```

Each output directory is single-attempt and refuses overwrite. Preparation verifies
all original corpus/question hashes. The run checks implementation/model/data hashes,
uses seed 42 and four CPU threads, and exports score reports plus label-free proposed
reader assignments. It does not contact Runpod or any model-serving API.

Metrics cover full original annotated paragraphs, any-character overlap and gold
character coverage. Unanswerable cases are not retrieval successes. Complete
paragraph coverage is a useful component check, not answer correctness or semantic
support. Independent human review remains pending; no training or model gain is
claimed. Small rerankers may truncate inputs; counts and source bounds are recorded.


## Results

80 existing questions were processed, 40 per split, with 320 method/question
outputs. Each split contains 20 answerable and 20 unanswerable rows. One dev
answerable row has no stored gold spans and is explicitly unscored; it remains
in the report. Unanswerable rows are not counted as retrieval successes.
All 320 selections completed; every BM25 window and score exactly matched the
previous smoke retriever for all 80 questions.

**Primary metric: at least one original annotated evidence paragraph fully
covered by the two selected windows.** This is a retrieval metric, not answer
correctness; it may omit other paragraphs needed for a complete answer.

| Method | Primary dev, full paragraph | Secondary exposed test, full paragraph | Any overlap, dev / secondary |
| --- | --- | --- | --- |
| BM25 | 4/19 | 9/20 | 10/19 · 12/20 |
| Dense only | 5/19 | 11/20 | 10/19 · 14/20 |
| Hybrid RRF | 5/19 | 14/20 | 11/19 · 17/20 |
| Hybrid + cross encoder | 5/19 | 10/20 | 11/19 · 12/20 |

[Decision](decision.json): **fail**. The preselected `hybrid_rerank` candidate
needed at least two more fully covered dev questions; it gained one. It passed
aggregate secondary non-regression, zero-selection-failure and CPU latency gates.
Dev had three full-paragraph wins and two losses; secondary likewise had three
wins and two losses. Aggregate non-regression does not imply no individual
regressions. All paired IDs are saved in the reports.

Plain hybrid was the strongest secondary ablation: five full-paragraph wins and
zero losses against BM25. Its primary gain was still only one question. We do
not promote a different method after inspecting secondary scores. The production
MCP, tool prompts, four-action reader budget and default retrieval remain unchanged.
No Qwen answer, fine-tuning or workflow improvement was measured.

[Previous live-smoke cases](previous-smoke-review.json): plain hybrid's humor
window 2400..4000 contains all seven cue/category terms in the two original
reference answers; BM25 and the cross encoder omit those terms. The complete
annotated paragraph starts at 2045, so hybrid gets partial paragraph coverage
under the fixed metric despite including those answer terms. For translation,
hybrid's 4800..6400 window includes LSTMs and attention, while the reranker
selects an ending/acknowledgment window and loses the annotated paragraph.
This is a posthoc implementing-assistant/literal-term diagnostic, not independent
human review or a new acceptance criterion. Qwen has not read these new windows.

## Cost, latency and limitations

Everything ran locally on **Apple M3, 16 GiB RAM, four CPU threads**, seed 42,
PyTorch 2.10.0 / sentence-transformers 5.3.0 / transformers 5.3.0. No GPU launch,
Runpod API operation, Qwen call or paid inference request occurred. One public
reranker-weight download was followed by offline-only model loading/evaluation.
Pinned weights loaded successfully; loader warnings concerned obsolete
`position_ids` buffers, with no missing learned weights reported.

Warm-query p95, primary / secondary:
- BM25: 0.00014 / 0.00019 seconds.
- Dense: 0.0043 / 0.0049 seconds.
- Hybrid: 0.0040 / 0.0044 seconds.
- Hybrid + reranker: 0.226 / 0.321 seconds.

These are query timings on in-memory indexes, not end-to-end MCP/Qwen latency.
Model loading in the new process took 2.97 seconds from cached files; this is not
a network cold start. Per-paper dense-index construction medians were 0.249 /
0.240 seconds. The evaluation reused one dev paper index. Full model/index/query
records are in [environment](environment.json), [hardware](hardware.json) and the
per-split reports. Timings were measured without the test suite running.

**Encoder truncation matters:** all 1252 dev and 1091 secondary per-question
window records exceed the embedding model's 256-token maximum. The dense model
therefore embeds a prefix of each 1600-character window, although the exported
reader assignment includes the complete window. These records include the
reused dev index. Cross-encoder truncation affected 2/395 and 5/397 query/passage
pairs at its 512-token maximum. These limitations prevent a broad claim about
hybrid retrieval or reranking in general. Candidate-pool union coverage was
17/19 and 18/20 before selecting the final pair, which uses substantially more
text and is not comparable to the two-window final metric.

Both splits were previously exposed, annotations can be incomplete or disagree,
and fixed windows can split evidence. Whole-paragraph coverage may be stricter
than answer-term availability and weaker than complete semantic support. Online
paper discovery, multi-paper synthesis, Qwen response quality, user productivity
and training improvement remain unmeasured.

## Using the candidate without changing the MCP default

The reusable helper returns source offsets that a host can explicitly supply to
the existing `read_experiment_passages` tool for an **already opened** cached
paper. It does not fetch papers, invent IDs, create notes, inspect source on
Qwen's behalf or dispatch Qwen. Both local model snapshots are required by this
comparison helper. For example, in a local Python caller:

```python
from src.research.hybrid_passages import LocalModels, build_index, retrieve

models = LocalModels()
# `document` is the exact document from an already opened, pinned paper.
index = build_index(document["text"], models)
selection = retrieve(index, question, "hybrid", models)
windows = selection["windows"]  # Check status before an explicit reader call.
```

This is opt-in experiment code. No new MCP tool or automatic reranking path is
installed. `reader-assignments.json` contains label-free proposed assignments for
the frozen QASPER evaluation adapter; those dataset IDs are not arXiv product IDs.
The next research question is whether full-window/paragraph-aware representations
and selection improve coverage at the same budget. Do not tune on the reserved
new-paper set or start another paid reader attempt from this result alone.

## Validation and artifacts

[Summary](summary.json), [primary dev report](dev.json), [secondary report](test_candidate.json)
and [proposed reader assignments](reader-assignments.json) preserve per-question
rankings, exact offsets, scores, truncation, paired differences and timing.
The initial implementation was `e79bd99`, with plan frozen in `ce63cc3` before
scoring. [35 focused tests](focused-tests-final.txt) passed, followed by the
standalone [full suite](full-tests.txt): **1993 passed** in 108.40 seconds, no
skips, three existing SWIG warnings. Ruff/diff checks passed. Existing V1 tools,
legacy prompts, 11 protected user files, earlier live results and paid-attempt
markers are unchanged; [preservation check](preservation.json) covers 774 prior
files. Human review remains null. Saved metric reconstruction and artifact
hashes are in `validation.json`.
