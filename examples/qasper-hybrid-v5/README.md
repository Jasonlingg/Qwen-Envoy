# V5 with matched hybrid-selected passages

This freezes the same three exposed QASPER questions, V5 model, system prompt,
decoding, source snapshots, Docker sandbox and four-request budget as the
[historical BM25 smoke](../qasper-passage-v2/README.md). Only passage ranges/read
examples change to the saved plain-hybrid windows. Results are pending.

The user explicitly authorized this one new live experiment after the offline
retrieval comparison. This does not promote hybrid to a product default, reset
an earlier allowance or authorize any subsequent paid attempt. No base-model
comparison, new fine-tuning, RL, reward change, new questions or reserved-pack
access is included. The reranker is not used.

[Predeclared scope, quality rule and budget](../../docs/implementation/2026-10-22-qasper-hybrid-v5-live.md).
[Exact first-request comparison](frozen/first-input-comparison.json) confirms each
payload matches its historical counterpart after replacing only the passage
scope JSON; system prompt, query, source identity and decoding remain identical.
Scoring references are separate and never sent to Qwen. Two disjoint windows,
1600 characters each, total 3200 characters. They were selected without gold
annotations by the frozen [hybrid comparison](../hybrid-passage-retrieval/README.md).

Success requires both answerable questions to have complete source-supported
requested content and a question-specific useful scoped gap for the unanswerable
case, plus valid packets/no execution errors. Report partial gains separately.
For humor, accept at least one complete original cue list and record both lists;
for translation require LSTMs and attention rather than generic NMT. Placeholder
gaps fail quality even when structurally valid. Exact excerpts prove provenance,
not semantic support. Implementing-assistant review is labeled; independent human
review remains null. This is not an official QASPER score.

QASPER attribution: Dasigi et al., AllenAI QASPER (2021), CC BY 4.0, revision
`13b496d2a5359329b110e3419628de3cf791843b`. Both the three questions and the
retrieval development sets were already exposed. QASPER/S2ORC paper snapshots
retain dataset revision and text coverage, without fabricated arXiv version IDs.
See [dataset provenance](../../benchmarks/envoybench/data/README.md).

The run uses the real in-process evaluation MCP adapter, Docker-only source
Python, pinned V5 checkpoint 50 and bounded queue/lifecycle. It tests selected
passage reading, not online discovery, multi-paper synthesis or product QASPER
ingestion. Historical BM25 is a saved reference, not a concurrent randomized
control; service/sampling variation is not isolated. No inference seed guarantee
or general fine-tuning improvement is implied by a difference in answers.

One existing A5000 activation, at most 900 seconds including startup; max 12
requests/98304 reserved tokens shared and four/32768 per case. No inference
retries or replacement cases. A fresh serverless rate <=$0.69/hour implies at
most $0.1725 estimated active compute for 15 minutes, not a provider invoice cap.
The local allowance is $0.50. Shutdown and full configuration restoration must
be verified separately. Earlier global attempt markers remain consumed.

To reproduce scripted preparation with cached QASPER/tokenizer data and the pinned
Docker image, choose a new output directory:

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/try_qasper_hybrid_v5.py prepare --output out/my-hybrid-scripted-preflight
```

This makes no paid calls or GPU launches. Live `run` additionally requires a
fresh plan-bound authorization and an exclusive new marker; it cannot reuse this
allowance. Opening saved artifacts makes no model call.
