# Offline two-passage hybrid retrieval comparison

Plan and implementation are frozen before scoring. Results are pending.
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
