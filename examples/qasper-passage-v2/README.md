# QASPER passage-v2 development smoke

This directory freezes three questions, BM25-selected passages, complete target
paper text, first model requests and scoring-only references before one bounded
live V5 run. Results are pending. Scripted responses used during preparation test
protocol plumbing, not Qwen quality.

QASPER attribution: Dasigi et al., *A Dataset of Information-Seeking Questions and
Answers Anchored in Research Papers* (2021), AllenAI QASPER, CC BY 4.0. Dataset
revision `13b496d2a5359329b110e3419628de3cf791843b`; frozen previously exposed
40-question `test_candidate` development selection. These are not original
questions, an untouched held-out test, or an official full-test QASPER score.
Upstream pretraining exposure is unknown. Original source question spelling is
preserved. See [data provenance](../../benchmarks/envoybench/data/README.md).

Selection and retrieval are fixed in the [scope](../../docs/implementation/2026-10-22-qasper-passage-smoke.md).
The reader is V5 checkpoint 50, assigned `passages-v2`, temperature 0, top-p 1,
thinking disabled; remote inference seed is not enforced. Two source windows
are selected using BM25 across the entire known target paper, without gold
annotations. Only selected passages can be inspected. An atomic gap is scoped
to those passages; it cannot establish whole-paper unanswerability.

The evaluation-only in-process MCP adapter calls the actual Docker-only reader
and packet verifier. It does not add QASPER ingestion to the personal arXiv MCP,
attach experiment notes or test public-network transport, online discovery or
open-ended research. Identical in-session tool retry returns its cached result
without inference. Durable experiment-note retry has separate existing tests.

`frozen/review-input.json` is scoring-only and never read by the model runner.
`frozen/first-inputs.json` shows the exact initial requests, with no references or
gold evidence. Snapshots retain QASPER/S2ORC coverage and dataset revision rather
than fabricated arXiv version IDs. The original corpus remains unchanged.

Preparation can be repeated locally with the frozen QASPER dataset, cached pinned
Qwen tokenizer, Docker image and project dependencies:

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/try_qasper_passages.py prepare --output out/my-qasper-offline-preflight
```

This dispatches scripted responses locally, not model API calls. Live `run` has a
plan-bound authorization, exclusive global attempt marker, shared 12-request and
98,304-reserved-token limits, four requests per case, a 900-second activation
watchdog including cold start, and shutdown/read-back. Preparation is not paid
run authorization. No training or reward change is included.
