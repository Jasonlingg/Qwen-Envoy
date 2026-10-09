# Personal-memory worker readiness screen — registered September 26, results September 28, 2026

This is a **synthetic development fixture** for getting a useful product out soon without new training. Nine public Markdown notes cover a life habit, a homework correction, and an AI-research decision. Each thread contains dated attempts, corrections, or plans; the ten named questions include temporal revision, conflicting evidence, a cross-domain question, and three absent-outcome traps. The notes, questions, reference answers, and anchors were agent-authored and source-checked. They have **not** received independent human reference review and are **not** a held-out private-vault test.

**Hypothesis:** the existing v5 Qwen checkpoint can use its multi-turn Python document tools to return source-grounded answers and honest uncertainty for this narrow memory-investigation job, without any new SFT. **Expected signal:** on the same frozen snapshot, v5 completes the tool loop, inspects the right dated notes, and has more useful evidence for revision or cross-note questions than a cheap BM25 passage packet, with no invented outcomes. **Decision rule:** use v5 as an optional, read-only demo worker only if independent blind review gives at least 8/10 fully supported answers, all three absent-outcome cases avoid invented facts, at least two of the three temporal-revision cases pass, no more than one execution-error episode, and no more than one escalation. On at least two questions, its five-passage packet must add an essential, source-supported passage absent from BM25's packet; it may miss no more than one question for which BM25 supplied all essential passages. Judge packet usefulness from the inspected sources, not exact anchor-string overlap alone. Base Qwen3-8B runs under exactly the same prompt and tool settings; report its paired outcomes even if the product uses v5. If these conditions fail, ship the capture/recall product with deterministic retrieval and Nemotron, and keep Qwen behind a visible experimental option. Passing this authored fixture would justify a user-approved real-vault confirmation, not prove private-vault readiness or trained-model improvement.

The frozen source hash is `a5470628a40fe6114271f3b2923ba025967bdc2552157543d2a7224440e43c47`. The sample vault is [`data/product_memory/sample_vault`](../data/product_memory/sample_vault), the current questions and source anchors are [`questions_v2.json`](../data/product_memory/questions_v2.json), and the shared prompt addition is [`prompt_v1.txt`](../data/product_memory/prompt_v1.txt). Version 2 corrected the three absent-outcome cases to require the relevant source note and removed an unasked date requirement before either Qwen arm ran on this fixture. Version 1 remains a historical development artifact. Do not train on this fixture or tune the prompt against its model outputs and then call the same ten questions held out. No oracle source IDs are inserted into the model questions. Both Qwen arms search the same corpus themselves. The lexical baseline sees only the question text and frozen corpus, never reference answers or anchor locations.

`scripts/personal_memory_eval.py` freezes the fixture through the existing opt-in vault importer, verifies the corpus hash and source anchors, emits a CPU BM25 passage baseline, and produces a per-question diagnostic report for any code-execution transcripts. This BM25 baseline is a distinct comparison system; the current local product CLI uses `ResearchTools.search_papers`, which has different retrieval behavior. The diagnostic report separates valid spans, required-note recall, tool action steps, syntax/runtime errors, repeated actions, escalation, and latency from human answer support. Exact spans establish provenance, not whether an answer claim is true. Run the baseline locally:

```bash
python scripts/personal_memory_eval.py freeze --output out/research/personal-memory-synthetic-v2
python scripts/personal_memory_eval.py baseline \
  --snapshot out/research/personal-memory-synthetic-v2 \
  --output out/research/personal-memory-synthetic-v2/bm25.json
python scripts/personal_memory_eval.py diagnose \
  --questions data/product_memory/questions_v2.json \
  --corpus out/research/personal-memory-synthetic-v2/corpus \
  --run out/research/personal-memory-synthetic-v2/bm25.json \
  --output out/research/personal-memory-synthetic-v2/baseline-diagnostics.json
```

`freeze` validates an existing snapshot and refuses to overwrite changed source files. A model run requires a GPU and is **not** launched by the CPU commands. On a GPU pod with enough storage, `scripts/run_personal_memory_qwen_eval.py` downloads pinned base and v5 weights (unless local paths are provided), runs each once at greedy decoding with the same `15`-step budget, seed `42`, evidence verifier, and recovery settings, then creates a blind answer-review bundle using the existing `scripts/review_code_exec_pilot.py`. It is a **trusted evaluation harness**, because the current REPL can fall back to a local Python process; do not expose it as a public arbitrary-user endpoint before isolation fails closed.

```bash
python scripts/run_personal_memory_qwen_eval.py \
  --snapshot out/research/personal-memory-synthetic-v2 \
  --output out/research/personal-memory-qwen-screen-v2
```

This command does not start a Runpod pod. It is intended for a trusted GPU machine with the project dependencies installed and sufficient disk for the pinned Qwen3-8B base. `--base-path` and `--adapter-path` can reuse already downloaded local weights; their identity then needs to be checked against the pinned revisions in the generated `model-identity.json`.

If an episode reports `status=error`, `run_eval.py` may exit nonzero after saving all ten rows. The paired runner now checks the complete transcript and manifest before continuing to the other arm; the error remains visible in diagnostics and review. If the process stops after a complete arm, rerun the same command with `--resume`. It validates `model-identity.json`, `run-config.json`, the frozen question/corpus/prompt hashes, per-arm policy and decoding settings, and every row before reusing that arm. It also validates and skips completed BM25, diagnostics, and blind-review outputs without overwriting human verdicts. Missing arms run; incomplete or mismatched artifacts stop the resume. In that case inspect and preserve the partial output, then use a **new output directory** for a fresh paired run rather than deleting individual transcript files or claiming a partial arm completed.

### CPU baseline result, September 27

The BM25 passage baseline ran on all ten version-2 synthetic questions. Its top-five packet included **all 20 required note occurrences** (17 for answerable questions and 3 for absent-outcome questions), and every returned character span was valid. It contained **9 of 21 exact reference-anchor needles**, a strict window-coverage proxy. It did not generate an answer, execute Qwen tools, or demonstrate correct abstention on the three absent-outcome questions. The strong note recall makes this a useful low-cost product fallback; Qwen must add clearer evidence selection, temporal handling, or safer uncertainty to earn its extra cost. These numbers come from `out/research/personal-memory-synthetic-v2/baseline-diagnostics.json`; source references remain agent-authored and human review is pending. The older version-1 baseline had 17/17 required notes and 7/18 anchors under its less demanding absent-outcome references; do not compare these counts as model improvement.

The existing September 26 base/v5 files are a separate **eight-question historical AI-paper development run**, not this memory fixture. Re-diagnosing those transcripts with their actual retrieved-ID question file shows base: 5/8 submitted answers, 4 execution-error episodes, 3 escalations, 87 document-tool action steps; v5: 8/8 submissions, 0 execution-error episodes, 0 escalations, 19 tool steps. The earlier blind semantic review found **zero fully supported v5 answers** on those eight questions. These are tool-behavior diagnostics only. They cannot establish answer support or transfer to dated personal notes. To regenerate the separate historical report:

```bash
python scripts/personal_memory_eval.py diagnose \
  --questions out/research/learning-loop-code-pilot-run-20260926/candidate-ids/retrieved_ids.json \
  --corpus out/research/starter-2026-09-12/corpus \
  --run out/research/learning-loop-model-sweep-20260926/base3.json \
  --run out/research/learning-loop-model-sweep-20260926/v5_3.json \
  --output out/research/historical-learning-loop-tool-diagnostics-20260926.json
```

Human review has two distinct jobs. First check the ten authored references against the frozen notes and mark the source-check sheet; without that review, results remain provisional. Then fill the blind answer review `pass`/`partial`/`fail`: a full pass needs every requested part, accurately dated source support, honest uncertainty, and no unrecorded result. Inspect the trajectory on tool failures and the cited passages for actual support. A real quote alone is insufficient. Score the bundle using `scripts/review_code_exec_pilot.py score`. Compare BM25 source coverage, base, and v5 in the diagnostic report, while remembering the BM25 artifact is an evidence packet rather than a generated answer. Never report the historical eight-paper development sweep as a run on this personal-memory fixture.

One GPU screen is enough to decide whether Qwen belongs in the initial demo; the product can proceed without another training job. The separate research gate still requires a held-out, human-reviewed, real-vault base-versus-trained comparison with acceptable latency and cost.

### Paired Qwen result, September 28

The registered ten-question paired run completed on a Runpod NVIDIA GeForce RTX
4090 (24 GB). Both arms used the same frozen questions and corpus, prompt hash
`d6465f8f7483add3a6a683eddef32edcc9b94014319c64cdc9df84573dd08ec7`,
15-step budget, seed `42`, greedy decoding (`temperature=0`, `top_p=1`, maximum
1,024 output tokens), and evidence-verifier settings. The base was
`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`; v5 applied
`jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5@27b912a863ff914ad45baa03624d8911dc1e17fb/artifacts/full/checkpoint-50`
with adapter SHA-256 `7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3`.
The [run configuration](../out/research/personal-memory-qwen-screen-v2/run-config.json),
[model identity](../out/research/personal-memory-qwen-screen-v2/model-identity.json),
[diagnostics](../out/research/personal-memory-qwen-screen-v2/diagnostics.json),
and full [base](../out/research/personal-memory-qwen-screen-v2/base.json) and
[v5](../out/research/personal-memory-qwen-screen-v2/v5.json) trajectories were
copied locally. The `outcome-v1` reward in the manifests is the existing MuSiQue
reward configuration; it is **not** a semantic research-answer score.

| Measured outcome | BM25 packet | Base Qwen | v5 Qwen |
| --- | ---: | ---: | ---: |
| Submitted answers / 10 | 0 (retrieval only) | 9 | 3 |
| Escalations / 10 | — | 1 | 6 |
| Episodes with execution errors / 10 | — | 2 | 10 |
| Successful document actions / attempted | — | 100 / 100 | 19 / 109 |
| Required-note recall proxy | 100% | 75% | 15% |
| Mean recorded episode time | — | 15.69 s | 21.05 s* |

*The v5 mean understates wall time: the `pm04_homework_correction` CUDA
out-of-memory episode was recorded as zero seconds despite taking several
minutes. Neither duration includes a measured production serving cost.

The main failure was tool grounding. V5 repeatedly supplied guessed dates,
numeric strings, or invented short labels as document IDs to `read()` instead
of discovering actual IDs with `search()`. Most such calls returned
`Document ... not found`. On `pm04`, a long sequence of invalid actions grew
the context until the 24 GB GPU ran out of memory. The base generally found
valid documents, although it also had two execution-error episodes and one
escalation. Mechanically valid quotation spans in either arm only prove that
the byte ranges exist; they do not prove that each answer claim follows from
the cited text.

Two independent AI-assisted reviewers scored the 20 anonymized answers from
the [blind bundle](../out/research/personal-memory-qwen-screen-v2/blind-review/review.md)
without opening the [identity key](../out/research/personal-memory-qwen-screen-v2/blind-review/blind-key.json)
until after their verdicts. They agreed on all 20: after key reveal, v5 had
**0 pass, 2 partial, 8 fail**, and base had **5 pass, 4 partial, 1 fail**.
The [AI-assisted verdict artifact](../out/research/personal-memory-qwen-screen-v2/blind-review/ai-assisted-review-20260928.json)
records all 20 blind IDs and the post-key counts. This is a provisional
answer-quality comparison, not independent human review;
the bundle's `review.json` remains unscored for that reason. Some answers that
reviewers marked pass have narrow spans that may not support every claim under
the stricter registered rule, so even base's five passes need human checking.
The direction of the v5 result is unambiguous because it submitted only three
answers and failed the execution and escalation limits by large margins.

### Live product-path smoke, September 28

We also served the pinned v5 adapter through a local OpenAI-compatible bridge
on the Runpod GPU and called the actual web worker over an SSH tunnel. The
worker executed model-authored code in the local Docker sandbox against the
same frozen nine-note snapshot. This was a separate integration smoke, **not**
an additional matched base/v5 evaluation or a semantic answer score. It used
the product prompt and a 12-step limit; its complete configuration and traces
are in the [strict smoke artifact](../out/research/qwen-product-smoke-v2-20260928-strict.json).

The first smoke exposed a fail-open counter: `search_within()` on a nonexistent
document produced an `{'error': ...}` result, but the worker counted it as a
successful inspection. We fixed that, rejected empty inspection results, and
required a discovery search before ID-based tools. Focused unit tests pass.
With those checks active, v5 **failed the first smoke question**
(`pm03_life_missing_outcome`): 12 model calls, zero successful source-text
inspections, and no accepted answer or abstention. The second question was
intentionally skipped after that failure. The prior apparent mechanical pass
is invalid because its counters accepted missing-document and empty-window
results; it must not be used as a readiness claim. The current worker fails
closed on this trace, and the web chat retains its lexical fallback when
Qwen does not return usable evidence.

**Readiness decision: v5 fails the registered screen and stays disabled as a
personal-memory worker in the initial multi-agent product.** It misses the
minimum eight fully supported answers (zero provisional passes), the maximum
one execution-error episode (ten), and the maximum one escalation (six). The
three absent-outcome cases did not all produce useful, source-supported
uncertainty. We did not score whether v5 added essential passages beyond the
BM25 packet, because the mandatory answer-quality and reliability conditions
already failed. The BM25 result is a low-cost source packet baseline, not a
chatbot-answer comparison. This authored fixture can screen a demo decision,
but cannot establish transfer to a real personal vault or demonstrate that
training improved Qwen over its base model; here the observed trained arm was
worse. The separate [hosted Qwen3-30B development smoke](../reports/nebius_qwen30b_memory_smoke_review_20260927.md)
was a different model and does not change this result. No new training job is
justified until real-vault questions, reviewed outputs, and a task-aligned
improvement rule are in place.
