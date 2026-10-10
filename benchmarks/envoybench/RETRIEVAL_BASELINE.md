# Known-paper retrieval comparator

This comparator asks whether the multi-step agent earns its complexity on the
current known-paper questions. It preserves the executable agent as the project
task and adds a simpler architecture for comparison.

**Current status: paired answer run complete; independent support review pending.**
The 40 BM25 packets at `out/envoybench/retrieval-baseline-prepared-20260930/`
were answered by both Qwen checkpoints. Saved answers and the official QASPER
Answer F1 subset score are in `out/envoybench/retrieval-baseline-run-20261010/`
and `out/envoybench/retrieval-baseline-official-score-20261010.json`.

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
diagnostics only. The paired run used 80 answer requests.

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

## Offline retrieval diagnostic

The CPU diagnostic at `out/envoybench/retrieval-diagnostic-20261010/report.json`
compared the prepared BM25 packets with pinned MiniLM dense retrieval and fixed
reciprocal-rank fusion (BM25 + dense). All three methods used the same paper,
1,400-character windows, and prompt budget. On the 20 questions marked
answerable in the **unreviewed** QASPER annotations, each method selected windows
covering at least 80% of an annotated evidence span for 19/20 questions. Across
43 annotated spans, BM25 covered 36, dense 35, and hybrid 35 at that threshold.
This is evidence-window overlap, not answer correctness or semantic citation
support. It gives no reason to replace BM25 for the answer comparison on this
known set. The diagnostic made zero model API requests.

Reproduce with a new output directory and the pinned MiniLM snapshot cached
locally:

```bash
python -m benchmarks.envoybench.retrieval_diagnostic \
  --prepared out/envoybench/retrieval-baseline-prepared-20260930 \
  --source-run out/envoybench/provisional-full-20260930 \
  --output out/envoybench/retrieval-diagnostic-reproduction-20261010
```

MiniLM encodes at most 256 tokens from each 1,400-character window, although a
selected window is supplied in full to the answer prompt. Its ranking can miss
evidence later in a window; these results do not establish that dense retrieval
is generally worse.

## Completed paired answer run

The paired run used the prepared packets, matching checkpoint pins and decoding,
and `out/envoybench/models-a40-bm25-20261010.json`, which records the actual
Runpod A40 and vLLM runtime. This is the execution command; a reproduction must
choose a new output directory because the runner refuses to overwrite results.

```bash
ENVOY_QWEN_ENDPOINT=http://127.0.0.1:8000/v1 \
python -m benchmarks.envoybench.retrieval_baseline \
  --execute \
  --prepared out/envoybench/retrieval-baseline-prepared-20260930 \
  --source-run out/envoybench/provisional-full-20260930 \
  --models out/envoybench/models-a40-bm25-20261010.json \
  --context-window 8192 \
  --output out/envoybench/retrieval-baseline-run-reproduction
python -m benchmarks.envoybench.qasper_official score \
  --run-dir out/envoybench/retrieval-baseline-run-reproduction \
  --output out/envoybench/retrieval-baseline-official-score-reproduction.json
```

Execution revalidated the frozen corpus, source manifest, tokenizer/template,
implementation, packet bytes, and model settings before any request. It wrote
ordinary `envoybench-run-v1` manifest/results with model keys
`qwen_base_retrieval` and `qwen_v5_retrieval`; `source_run_id`, architecture, and the
whole preparation protocol remain explicit. It does not execute generated code.
All 80/80 requests submitted an answer; no endpoint errors were recorded. The
Runpod pod was stopped after the run.

| System | Official Answer F1 | Answerable (20) | Unanswerable (20) | Missing predictions |
| --- | ---: | ---: | ---: | ---: |
| BM25 + Qwen base | 41.52 | 33.04 | 50.00 | 0 |
| BM25 + Qwen v5 | 58.86 | 17.72 | 100.00 | 0 |
| Original agent + Qwen base | 19.92 | 29.83 | 10.00 | 11 |
| Original agent + Qwen v5 | 30.69 | 21.37 | 40.00 | 0 |

On the current QASPER reference labels, BM25 + v5 refused all 20 unanswerable
questions but also refused 9/20 answerable ones. The v5 agent falsely refused
1/20 answerable questions and answered 12/20 unanswerable ones. Thus the BM25
v5 aggregate gain is driven by abstention; its answerable F1 is lower than the
v5 agent's. All nine of those false refusals had at least one annotated evidence
span fully visible in the BM25 packet; eight had every annotated span fully
visible. This points toward an answer/abstention problem, though the unreviewed
annotations do not establish that the packet was sufficient. These are official
token-overlap Answer F1 scores on 40 selected
QASPER test questions, not a full QASPER leaderboard result. The references
have not been independently reviewed, and Answer F1 cannot establish whether
the cited passages support an answer. The baseline and original agent used
different runtimes, so their observed latency difference cannot be attributed
to architecture alone.

The existing `score prepare-provisional` → grading → scoring → Studio workflow
can consume the completed paired baseline run. A baseline's answers still need
review using the same rubric and grading status as the agent answers before a
supported-answer comparison. Source-span existence remains a
mechanical check; it does not establish semantic support. Keep baseline and agent
run identities separate when reporting the architecture comparison.

Because current agent traces have already been inspected, this is a diagnostic
comparison on a known set. Do not describe it as an untouched confirmatory test.
