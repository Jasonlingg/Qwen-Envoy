# V5 with matched hybrid-selected passages

This freezes the same three exposed QASPER questions, V5 model, system prompt,
decoding, source snapshots, Docker sandbox and four-request budget as the
[historical BM25 smoke](../qasper-passage-v2/README.md). Only passage ranges/read
examples change to the saved plain-hybrid windows. The run is complete, with
one improved answer and continuing reader failures.

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


## Actual live result

Nine actual V5 requests, three per case: **27,771 input + 499 output = 28,270
reported tokens**; 73,728 worst-case tokens reserved. Historical BM25 used nine
requests / 27,936 tokens. All requests used `envoy`; the worker verified the same
V5 adapter/hash, Qwen base revision, PyTorch 2.13.0+cu130 and vLLM 0.31.0 on one
A5000. This was not a new model or a training run.

| Case | Historical BM25 result | Actual hybrid-input V5 result | Reviewed outcome |
| --- | --- | --- | --- |
| Humor lexical cues | Could not identify cues; retrieval omitted annotated paragraph | Read both hybrid passages but said they “do not specify lexical cues” | Fails: all seven reference cue/category terms were in the actual request and returned quotes. |
| Translation models | Generic “NMT models” | NMT with an LSTM-based architecture, attention and dropout | Passes requested details; dropout is also supported, at source offset 5659. |
| Racial/ethnic bias | Atomic gap with copied generic placeholders | A question-specific prose statement that the passages do not establish the extension, submitted as `evidence_found` | Scoped uncertainty is more specific, but there is no atomic gap or follow-up query; the predeclared gap criterion fails. |

**Complete requested answers: 0/2 → 1/2** under implementing-assistant diagnostic
review. Useful structured gaps: 0/1 → 0/1. Overall predeclared quality gate:
**not met**. The one old mechanically terminal gap became a cited prose
non-answer, so gap protocol completion regressed even though its wording became
more specific. No unsupported racial capability was invented.

[Claim-level review](claim-review.json) separates source support, answer
completeness and gap classification. It is not an independent human review or
an official QASPER score. In translation, the original annotated paragraph
5164..5336 and the dropout sentence fall inside the returned verified quote
4800..6000. In humor, the verified 2400..3600 quote itself contains all reference
terms. [Actual-input receipt](actual-input-facts.json) confirms those terms were
present in the final request sent to Qwen, not merely in a local source file.
Literal presence alone does not establish semantic support; the review also
compares the answers with the source statements and original references.

Three terminal packets passed MCP provenance/protocol verification, with six
successful Docker source inspections. There were **zero Python execution errors,
action rejections, submission rejections or gap rejections**, and no inference
retry/resubmission. All raw actions matched the policy-cleaned actions in this
run. Same-request in-session MCP retry dispatched nothing. Raw API outputs,
[packets, verified quotes and MCP responses](live/) remain saved alongside the
[request ledger](session.requests.jsonl). A provenance-valid packet can still
contain an incorrect refusal or a semantically misclassified non-answer.

The existing investigator clips authored 1600-character citation ranges to
1200 characters. Both authored and actual verified ranges are retained. No
clipping behavior, validator, prompt example, model weight or tool was changed
after observing these results. The experiment did not repair the unresolved
semantic classification of prose inability as `evidence_found`.

## Interpretation and limits

Hybrid retrieval supplies useful missing facts, and one answer became more
complete in this historical matched smoke. It does not fix this V5 reader's
behavior throughout the task: the humor question now exposes failure to use
available evidence, and question-specific structured gap reporting still fails.
That is a concrete target for grounded extraction/abstention demonstrations and
subsequent controlled evaluation. This run does not prove more fine-tuning is
necessary or sufficient; it does not compare trained versus base Qwen.

The comparison uses three already exposed questions, an ablation selected after
an exposed offline diagnostic, and a historical reference. GPU workers/datacenters
and service timing differ; remote inference seed is not guaranteed. Causality,
generalization, model improvement, full QASPER performance, workflow usefulness
and a productivity advantage are not established. The fixed humor window begins
mid-paragraph; weights, prompt wording and conservatism are not separately
isolated. The product default remains unchanged, and the reserved new-paper pack
was not opened or used. No further paid attempt or training is authorized by this
completed test.

## Boundaries and validation

The watchdog did not fire. Lifecycle records 253.44 seconds before cleanup
began; that is not billed time or an invoice figure. Actual provider spend is
unknown. At the checked $0.69/hour rate the authorized 900 active seconds estimate
$0.1725 compute, not an enforceable invoice cap.

[Fresh 16:35:47 UTC reads](fresh-shutdown.json) verified workers **0/0**, no workers
or pending jobs, full original configuration equality, and nine additional
completed jobs (186 → 195). The pre-existing failed count remained one. All
[six owned Docker containers](container-cleanup-proof.json), three scripted and
three live, were independently confirmed absent. Other resources were untouched.
The new `out/qasper-hybrid-v5-LIVE_ATTEMPT` marker is consumed; all older markers
remain unchanged.

Scope `1de2485`, implementation `7ba4e52`, frozen inputs `6ff9404`. Final focused
checks: **64 passed**, including the real pinned Docker integration. Initial
focused invocation used the default image tag and skipped one Docker test;
rerunning with the pinned image passed all 64 with no source change. Standalone
[full suite](full-tests.txt): **2012 passed** in 98.49 seconds, no skips, three
existing SWIG warnings. Ruff/diff checks passed. [Preservation](preservation.json)
verified 804 prior file hashes, including all 11 protected files and old attempts.
Human-review fields remain null. `validation.json` records artifact hashes,
reconstructed packet checks and exact first-input comparisons.
