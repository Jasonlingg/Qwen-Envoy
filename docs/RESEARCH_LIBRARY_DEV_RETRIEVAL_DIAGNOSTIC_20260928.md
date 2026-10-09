# Public-paper development retrieval diagnostic — September 28, 2026

This is a cheap source-discovery reference, not a Qwen evaluation or an
answer-quality score. The four provisional development questions `D01`–`D04`
were run against the existing 20-paper snapshot. Their reference answers still
need human review; the twelve held-out question texts were not used.

**Hypothesis:** a fixed lexical retriever may recover the needed paper IDs
without a model if allowed a wider candidate list. **Signal:** required-paper
recall as candidate count grows. **Decision:** do not claim Qwen adds value for
paper discovery if a cheap, fixed retriever finds the same source papers; judge
Qwen on multi-step passage selection and supported answers, including the
context and cost needed to do so. This question does not change the registered
held-out promotion rule.

The baseline is `scripts/prepare_ai_paper_id_diagnostics.py`: BM25 over
512-character text passages with 64-character overlap and a doubled title.
It ranks documents by each document's highest-scoring passage. The query is
the raw development question; no model-generated rewrite or answer label enters
retrieval. The source IDs are compared with each question's provisional
`required_doc_ids` only after retrieval.

| Top paper candidates | Required IDs found, out of 7 |
| ---: | ---: |
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 4 | 5 |
| 5 | 6 |
| 6 | 6 |
| 8 | 7 |
| 10 | 7 |

At top four, `D03` and `D04` each miss one required paper. At top eight, all
seven required paper references are present. This tiny, provisional set does
not measure passage relevance, semantic support, final-answer quality, latency,
or how many tokens a host must read. It gives Qwen a meaningful comparison:
recover and inspect the right passages with less irrelevant text, or abstain
well when the papers do not establish the requested local conclusion.

Reproduction details: development benchmark
`data/research/research_library_transfer_dev_v1.json` SHA-256
`797e5bc40ac1559fc957a19edef2a94a3597992399501f006808af5a134f3154`;
snapshot `out/research/ai-agents-development-v1-20260912` corpus hash
`9a9c1d750daf5647898eb18681a5f75775ba76047713316dcc5583f6ed201598`;
retrieval-script SHA-256
`75081d3daf925c19892fb67dfd9bf051b919e326b1dab27a67987bc9f3e62b6b`;
repository HEAD `c52341c8c958a3cd47df599fd1ff6ff41604afa8`; Python 3.11.9 on
macOS arm64. Checkpoint, policy decoding, reward version, GPU hardware, and
random seed do not apply: this run used no model and no randomness. The
top-four per-question audit is in
`out/research/research-library-transfer-dev-v1/candidate-ids/retrieval_audit.json`.

## Repair of Qwen's built-in search tool

The external BM25 reference above is distinct from the `search()` function
available to Qwen in the executable Python environment. Before any new Qwen
run, a check of the old document-length-sensitive `search()` on the same four
raw development questions found **2/7** required paper IDs in its top five.
The default search was then changed to passage-level BM25Okapi with title terms
boosted twice, returning the best matching passage and its offset per paper.
The frozen corpus and question wording were unchanged; the held-out question
drafts were not inspected. The new default finds **5/7** required IDs in its
top five and **7/7** in its top eight. Short follow-up searches using paper
names find the missed top-five papers, so the multi-step query-reformulation
task remains real. The
legacy `search(method="chunk")` path is unchanged.

The new search protocol is `passage-bm25-okapi-v1` in `src/env/tools.py` (SHA-256
`56128534a3f24ff1d95008a473be80e28c843d0f7ebdfb013733361771d96208`).
All new model arms must run with this exact tool revision; older Qwen scores
using the prior search implementation are historical and cannot be compared
as matched controls. `tests/test_repl.py` passes 40 focused checks, including
a long-generic-document regression test. This tool repair is an infrastructure
change, **not** trained-model improvement or an answer-quality result.
