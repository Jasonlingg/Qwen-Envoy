# Local NVIDIA / Nebius integration — October 6, 2026

The Studio paper reader made real Nebius Token Factory calls, executed the
returned actions in the local Docker sandbox, persisted the trace, and displayed
the resulting answer and exact source spans. This is a **functional integration
check**, not a benchmark comparison or proof of reliable source-supported answers.

## Fixed question and source

Source: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), fetched as
a PDF through the app and frozen before inference. Question: “How many layers
are in each encoder and decoder stack in the Transformer base model described
in this paper?” Each stack has six layers.

All four attempts in this finishing pass are included below. They ran on
October 6 in America/Toronto (October 7 UTC). The per-attempt plans were written
before the requests. The later attempts are diagnostic changes and post-hoc
model selection; they are not independent held-out trials. No Qwen benchmark
grades, training data, or rewards were changed.

| Attempt | Model | Requests | Observed input / output tokens | Seconds | Outcome |
|---|---|---:|---:|---:|---|
| 1 | Nemotron Lightning | 7 | 18,280 / 1,520 | 10.66 | Parser failure after six recorded actions; rejected text was not captured at that point. |
| 2 | Nemotron Lightning, thinking disabled | 2 | 2,699 / 138 | 2.06 | Fabricated source ID rejected, then ambiguous-thinking parse failure. |
| 3 | Nemotron Lightning, thinking disabled | 12 | 50,249 / 456 | 13.27 | Reached the action limit without submitting. |
| 4 | Nemotron Ultra, thinking disabled | 10 | 42,982 / 1,014 | 15.37 | Submitted a correct answer with incomplete supporting citations. |

Token totals are provider-reported observations, not a billing receipt. Failed
requests retain their usage-completeness flags. Ultra reported 26,112 cached
input tokens. No new GPU pod or model training was used.

## What the completed run actually shows

Run `run_3666f88fff934274b76c3c1e6b4e4017` used
`nvidia/Nemotron-3-Ultra-550b-a55b` at
`https://api.tokenfactory.nebius.com/v1`. It searched the paper, read it, inspected
passages, and finally submitted “N = 6” for both stacks. It also produced two
Python syntax errors and three repeated-action rejections along the way.

The source checker accepted two real text spans. **That did not make them good
citations.** The first span, offsets 7500–7600, describes sub-layers and misses
the encoder layer count, which starts in the sentence at offset 7405. The
second span, 8000–8100, includes the decoder's count but truncates its sentence.
The answer is consistent with the paper, but the submitted evidence does not
fully support both claims. This is an assistant source inspection, not an
independent human review; the app correctly leaves semantic support unreviewed.

This is a useful Studio demonstration: a plausible answer, valid source offsets,
and a completed protocol can still conceal evidence-selection and execution
failures. The trace and source reader make those failures inspectable.

## Reproducibility and limits

[`live-validation.json`](live-validation.json) records model declarations,
source and prompt hashes, decoding, limits, usage, outcomes, original run-file
hashes, and the assistant review. Plans are in `live-plans/`. Full original run
records remain locally in `out/envoybench/paper-workspace/runs/`; open `/papers`
and select a saved run to inspect or export one. The public summary intentionally
does not duplicate the paper's full extracted text. Re-extracting a PDF with a
different library version can change offsets; use the recorded hashes to
identify the tested snapshot.

Provider model revisions and hardware are unpinned and not endpoint-attested.
Temperature was zero, top-p one, and the maximum output was 1,024 tokens per
request. The configured seed was 42 but was **not sent** to this endpoint.
Each attempt was capped at 12 action rounds and a 300-second between-request
budget. This is one selected, well-known paper question; upstream training
exposure is unknown. Nano is offered in the example configuration but has not
been tested in this pass. A successful submission is not a reliability estimate.

For a live demo, choose `nemotron_ultra_nebius`, fetch the paper, and ask the
question. Each click on **Run** spends API credits and may produce a different
result. Reopening the saved run makes no model call. The launch command and
Docker setup are in the [release README](README.md).

The live integration requirement has local evidence. Public deployment, the
video, final rules review, and Devpost submission are still unfinished.
