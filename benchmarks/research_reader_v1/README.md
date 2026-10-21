# Research reader evaluation preparation

This pack fixes the questions, source snapshots, expected facts and review rules
for a future comparison of Qwen paper-reading behavior. It contains **12 exposed
development questions and 12 reserved evaluation questions**, with four distinct
paper IDs in each split. No model has attempted the new evaluation questions in
this preparation. No training or paid inference is authorized by these files.

The task is multi-step assigned-paper reading: Qwen searches and inspects an
immutable paper using document-tool Python in Docker, then submits supported
facts or reports a gap. Discovery, full research synthesis and local ML
experiments are separate tasks and are not measured by this pack.

## Questions and source versions

Development reuses the previously evaluated PagedAttention, QLoRA,
FlashAttention-2 and AWQ cases from
[the base versus v5 comparison](../../examples/base-v5-reader-comparison/cases.json).
They remain development prompts; their historical strict references and reported
results are not rewritten. The source audit records inherited limitations.

The new evaluation split has two answerable questions and one target gap for
each paper:

| Paper version | Answerable questions | Target gap |
| --- | --- | --- |
| GQA 2305.13245v3 | Head sharing and limiting cases; conversion and uptraining conditions | Peak KV-cache footprint for two Qwen readers on A5000 |
| SmoothQuant 2211.10438v7 | Equivalent smoothing and bit widths; calibration data and migration strengths | Target latency and quality comparison |
| KIVI 2402.02750v2 | Key/value quantization dimensions and residual precision; efficiency-test conditions | Total target peak VRAM |
| DoRA 2402.09353v6 | Magnitude/direction adaptation; training-memory modification and its comparator | Target DoRA-versus-LoRA memory/quality comparison |

Exact questions are in [cases/evaluation.json](cases/evaluation.json). Expected
facts and exact source spans are in
[references/evaluation.json](references/evaluation.json); these are grader data.
Paper documents include original titles, authors, versioned source URLs and
extraction coverage. All eight bundled documents use HTML paragraph extraction,
which can omit figures, tables and equations.

## Review and integrity

A separate agent checked all 12 new questions and their criteria against the
complete cached sources. Revisions removed unasked grading requirements,
distinguished cache footprint from total VRAM, preserved numeric comparators,
and aligned gap questions with the existing `report_gap(missing,
follow_up_query)` plus abstention interface. A next-search suggestion is not an
executed search or experiment proposal.

This is **provisional model-assisted source review**, not human validation or
grading of model answers. The [source audit](reviews/source-review.json) retains
per-case judgments, resolved findings and reviewed file hashes. The
[human answer-review template](reviews/human-answer-template.json) stays blank.
For an actual run, copy the template into that run's review directory and bind it
to the run manifest before a person fills it; do not edit the frozen template.

```sh
python scripts/check_research_reader_eval.py
```

The check requires no network, tokenizer, model or GPU. It validates the
[freeze lock](freeze-lock.json), paper-ID split isolation, exact reference spans,
source-review binding and blank human-review fields. These checks establish
artifact consistency, not correctness of the scientific judgments. The lock is
a reproducibility record, not an authentication or access-control mechanism.

`src.eval.reader_eval_bundle.model_cases(root, split)` exports only the question
and paper version for an explicitly selected split. Case-kind labels, expected
facts and reference excerpts are not included. There is no execution or training
command. The older comparison runner remains bound to its historical dataset;
this pack is not silently substituted into it.

## Split policy and future decisions

The new questions were reserved before model inference and their paper IDs do
not overlap development. The four public papers are not certified absent from
Qwen pretraining or earlier training data. A limited project search found no
direct earlier references to these IDs; LoRA was excluded because it appeared
in a previous web-evidence task. This is not a complete contamination audit.

Keep evaluation questions, reference facts and complete evaluation paper
snapshots out of training and prompt/reward tuning. Public files do not enforce
that policy. Once evaluation outputs inform changes, treat the set as exposed
development and reserve new papers for the next assessment.

The [protocol](protocol.json) pins the existing prompt hash, baseline identities,
decoding settings, semantic rubric and a limited exploratory decision rule.
Report eight answerable cases separately from four gap cases. Valid Python,
real quotation offsets and extra refusals cannot substitute for more supported
answers. Candidate answers must carry their own supporting evidence; reference
excerpts cannot repair missing citations. Score only requested facts and relevant
unsupported additions, accepting equivalent supported wording.

Before any paid execution, separately freeze the candidate checkpoint, actual
hardware, source/question revisions and aggregate request/token/time limits.
This preparation has no RL reward or training job. Existing MuSiQue reward is
not a research-answer correctness score. Human answer review, a baseline run and
any later training intervention remain pending.
