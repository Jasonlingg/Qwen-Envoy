# Offline hybrid passage retrieval comparison

User approved evaluating stronger retrieval before further Qwen calls/training.
Scope: CPU-only, existing exposed QASPER documents/questions, unchanged two-passage
reader contract. No Runpod activation, paid inference, Qwen request, model training,
reward change, reserved research-reader pack access or production default change.
Public model weights may be downloaded once; model loading/evaluation is then
local-only. No question text or paper content goes to a remote inference provider.

Hypothesis: hybrid lexical/semantic retrieval with a small cross encoder selects
more complete annotated evidence under the same context budget than the last
smoke's keyword baseline. This is retrieval coverage, not answer quality.

Freeze four arms before looking at their scores:
- `bm25`: exact previous BM25Okapi defaults and greedy disjoint selection.
- `dense`: normalized cosine embeddings, source-start tie order.
- `hybrid`: equal-weight reciprocal rank fusion (k=60), full BM25/dense rankings.
- `hybrid_rerank`: cross-encoder ranking of the first 10 hybrid candidates.
All use full-paper 1600-character windows at stride 800, including the final
end-aligned window; return two disjoint windows in source order, at most 3200
characters total. No gold-guided boundaries, answer-aware queries, score tuning,
replacement questions or parameter search. Query is `source_question` only.
The old smoke implementation and recorded inputs stay byte-identical.

Models: cached `sentence-transformers/all-MiniLM-L6-v2` at
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, max 256 tokens, and
`cross-encoder/ms-marco-MiniLM-L6-v2` at
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, max 512 tokens, PyTorch bin
(existing repository reranker convention), batch size 16. Record model file
hashes, installed library versions, CPU hardware/threads, seed 42 and tokenizer
truncation counts. Both are Apache-2.0; upstream training exposure is unknown.
Source model cards: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
and https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 .

Use the existing exposed `dev` 40 questions as the primary decision set and the
existing exposed `test_candidate` 40 as a secondary regression diagnostic. These
are not untouched held-out data or a full QASPER leaderboard. Preserve all 80
question IDs; score answerable rows with gold spans and report missing-gold rows
explicitly. Unanswerable rows have no retrieval correctness label and are not
counted as retrieval successes. A preparation/selection failure counts as a
failure, not a reason to drop a question. No gold enters the retriever interface.

Primary metric: fraction of answerable questions for which selected windows
fully cover at least one original annotated evidence paragraph. Also report
any-character overlap, mean fraction of gold characters covered, all-gold
paragraph coverage, candidate-pool coverage, paired wins/losses and per-question
results. Paragraph overlap alone does not prove a complete answer, and full
paragraph coverage is not semantic support. Keep original annotation disagreement.

Decision rule for the preselected `hybrid_rerank` candidate: on primary dev,
fully-covered-question count increases by at least 2, without reducing any-overlap
count; secondary exposed test fully-covered count must not decrease. Steady-state
CPU query p95 must be <=2 seconds on both splits; model load and per-paper indexing
are measured separately and excluded from that query threshold. Report all timing
and indexing costs rather than presenting the warm timing as end-to-end latency.
Any candidate retrieval/selection failure blocks promotion. Other arms are
ablations, not candidates selected after seeing test scores.
If any gate fails, retain BM25 as default and report the candidate's limitations.
Even a pass authorizes no claim of Qwen answer improvement or automatic paid run.

Implement reusable selection returning exact source offsets; export proposed
reader assignments separately from scoring labels. Focused tests must cover
fusion/reranking, annotation separation, bounds, disjointness, failure accounting
and exact BM25 compatibility. Run the full relevant suite before finishing.
