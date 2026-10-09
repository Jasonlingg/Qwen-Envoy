# Known-paper retrieval comparator

This comparator asks whether the multi-step agent earns its complexity on the
current known-paper questions. It preserves the executable agent as the project
task and adds a simpler architecture for comparison.

**Current status: prepared, not run.** Forty packets exist locally at
`out/envoybench/retrieval-baseline-prepared-20260930/`. No baseline answers, grades,
quality result, provider token usage, or inference cost have been measured.

The hypothesis, expected signal, and decision rule are written into `plan.json`
before inference. Compare supported answers on answerable questions, correct
refusals, unnecessary refusals, latency, and actual reported usage separately.
An aggregate score on the balanced answerable/unanswerable set is insufficient.
Ties or inconclusive grades do not demonstrate value from the agent loop.

## What stays the same and what changes

| Matched to the completed agent run | Deliberately different |
| --- | --- |
| Frozen question IDs and corpus bytes | Fixed retrieval instead of model-authored exploration |
| Base/adapter revision and adapter hash | One answer request, no code or tool loop |
| Temperature, top-p, maximum output tokens, thinking setting, seed behavior | Short answer prompt with retrieved passages |
| Exact citation document IDs and exclusive character spans | No verifier retry or retrieval/query rewrite |
| Explicit server context limit | Context chosen by deterministic BM25 |

This is an **architecture comparison**, not a pure model or training comparison.
Any future fine-tuning claim still needs base and v5 under the same architecture.
Hardware/runtime can differ when renting another machine and are recorded anew;
if they differ, do not describe the latency difference as an architecture effect.

Every frozen question already names its target paper. The baseline reads that ID
and the original question from the model-visible prompt. It does not use gold
answers, answerability labels, annotated evidence, or other grading fields for
retrieval. It divides only the named paper into 1,400-character windows with 200
characters of overlap, ranks them with BM25 (k1=1.5, b=0.75, positive IDF), breaks
ties by source offset, and includes up to twelve complete windows that fit.

The pinned Qwen tokenizer is loaded **from local cache only**. It counts the full
rendered chat template, including instructions, question, passage metadata, and
generation prefix. The prepared 8,192-token context limit reserves 1,024 output
tokens and a 128-token safety margin. Missing local tokenizer files fail instead
of triggering downloads or substituting approximate counts. The server must use
that tokenizer/template and context limit when executing; model identities remain
operator declarations, not endpoint attestations.

Local prompt counts are not provider usage or latency measurements. The current
40 packets contain 2,457–5,452 local prompt tokens each; these are input-size
diagnostics only. The prospective paired run would require 80 answer requests.

## Prepare without inference

```bash
python -m benchmarks.envoybench.retrieval_baseline \
  --prepare \
  --source-run out/envoybench/provisional-full-20260930 \
  --output out/envoybench/retrieval-baseline-prepared-20260930 \
  --context-window 8192
```

Preparation writes `plan.json` with `status: prepared_not_run` and gold-free
`packets.json`. It never resolves an endpoint, loads API credentials, starts a
GPU, or sends a model request. The output directory must be new.

## Execute later, explicitly

Only run this after choosing to spend inference compute and making a matching
endpoint available. Use a model config with its actual current hardware/runtime
and price. Keep checkpoint pins and decoding unchanged. If the config declares
`serving_context_window_tokens`, it must match the prepared limit.

```bash
python -m benchmarks.envoybench.retrieval_baseline \
  --execute \
  --prepared out/envoybench/retrieval-baseline-prepared-20260930 \
  --source-run out/envoybench/provisional-full-20260930 \
  --models out/envoybench/models-a40-provisional.json \
  --context-window 8192 \
  --output out/envoybench/retrieval-baseline-run
```

Execution revalidates the frozen corpus, source manifest, tokenizer/template,
implementation, packet bytes, and model settings before any request. It writes
ordinary `envoybench-run-v1` manifest/results with model keys
`qwen_base_retrieval` and `qwen_v5_retrieval`; `source_run_id`, architecture, and the
whole preparation protocol remain explicit. It does not execute generated code.
Each failure records elapsed request time; usage is null unless the endpoint
returns it. All attempts are saved after each question. There are no automatic
retries or hidden follow-up model calls.

The existing `score prepare-provisional` → grading → scoring → Studio workflow
can consume a completed paired baseline run. Preparation alone is deliberately
not a scoreable run. A baseline's answers must be reviewed using the same rubric
and grading status as the agent answers. Source-span existence remains a
mechanical check; it does not establish semantic support. Keep baseline and agent
run identities separate when reporting the architecture comparison.

Because current agent traces have already been inspected, this is a diagnostic
comparison on a known set. Do not describe it as an untouched confirmatory test.
