# QASPER passage-v2 development smoke

This directory freezes three questions, BM25-selected passages, complete target
paper text, first model requests and scoring-only references before one bounded
live V5 run. The run completed, but the useful-output gate was not met.
Scripted responses used during preparation test protocol plumbing, not Qwen quality.

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

## Actual live results

V5 made **9 requests**, three per question, with **27,529 input + 407 output =
27,936 reported tokens**; 73,728 worst-case tokens were reserved. No inference
resubmission, action rejection, submission rejection, gap rejection or Docker
Python execution error occurred. All six document inspections were actual
Docker passage reads. All three raw packets passed the MCP/verifier provenance
and protocol checks; identical in-session MCP retry made no additional call.

| QASPER question | Retrieval | Actual reader output | Claim-level review |
| --- | --- | --- | --- |
| What lexical cues are used for humor recogition? | Missed annotated paragraph 2045..3050 | Submitted inability to identify cues as an `evidence_found` answer | Scoped inability is reasonable for the inspected windows, but it is not an answer or structured gap. No cue names were invented. |
| What translation models are explored? | Entire annotated paragraph 5164..5336 was inspected | “Neural machine translation (NMT) models are explored.” | Broadly supported, incomplete: omits available LSTM/attention details. |
| Can their approach be extended to eliminate racial or ethnic biases? | Original annotations mark unanswerable | Terminal `submit_gap` after two reads | Protocol worked, but missing/query fields copied generic prompt placeholders and are not useful. No racial-bias capability was invented. |

[Claim-level review](claim-review.json) was performed by the implementing
assistant against the exact inspected text and original references. It is not
independent human review; those fields remain null. Mechanical completion is
3/3. Useful answer or question-specific gap quality did **not** meet the gate.
No fine-tuning gain, generalization, semantic absence, productivity advantage or
research-workflow success is established. There was no base-model comparison,
new training, RL, reward calculation, Nebius call or reserved-set access.

The existing investigator clipped authored 1600-character evidence spans to
1200 characters each. Raw actions and the actual verified spans are both saved;
the second translation citation therefore stops at 5200, although Qwen inspected
the complete window through 5600. The abstract independently supports its broad
NMT claim. Verifying those quotes does not establish answer completeness or
validate prose inability as `evidence_found`.

[Raw requests/responses](session.requests.jsonl), [packets and MCP responses](live/),
[retrieval overlap check](retrieval-review.json) and [usage](result-summary.json)
are inspectable. The runtime records implementation commit `9291514` with the
pre-existing protected workspace changes retained; implementation hashes pin
those bytes. Preparation caught an empty required-citation list on unanswerable
rows before any paid activation; source-paper identity now handles those rows.
The aborted preparation remains private under `out/`; it was not a model run.

The worker spent most of this session on image/model startup. Hardware identity:
A5000, PyTorch 2.13.0+cu130, vLLM 0.31.0, CUDA 13.0, same pinned V5 adapter. The
watchdog did not fire. At the observed $0.69/hour rate, 900 active seconds imply
$0.1725 estimated compute, not an invoice cap or actual billed spend. Actual
provider spend is not available here. No further paid attempt is included.

[Fresh 14:32:21 UTC shutdown reads](fresh-shutdown.json) verified workers **0/0**,
no workers, no pending jobs and full configuration equality with the preflight.
Completed-job count increased from 177 to 186; the pre-existing failed count
remained one. All [six owned Docker containers](container-cleanup-proof.json)
were independently confirmed absent. Other containers were left alone.

Focused checks: initial 60 passed; after the preparation fix, 11 QASPER checks
passed. Final standalone [full suite](full-tests-final.txt): **1974 passed** in
107.95 seconds, no skips, three existing SWIG warnings. Ruff/diff checks passed.
The V1 preamble SHA-256 remains
`233adfef3cd72c8b8221e45dcaa58c76809f36bd717d1d342514a7c332c3dd71`.
Eleven protected files, 637 prior artifacts/markers and 15 additional atomic-gap
files remain unchanged. Earlier paid-attempt markers were not reset.

The next work should address the measured issues separately: retrieval coverage,
answer completeness, and useful gap content. This smoke does not establish that
changing weights is necessary or sufficient. It also does not show the new
atomic-gap protocol improved quality relative to the old protocol: questions,
retrieval and assignments differ from the previous three-case trials.
