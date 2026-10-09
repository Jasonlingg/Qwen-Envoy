# Learning-loop model sweep — 2026-09-26

## Question and limits

On the frozen eight-question, six-paper learning-loop development pilot, which of the
user's published Hugging Face adapter repositories performs best, and how does a
Sonnet code-execution agent compare? These eight questions and reference answers
were written and inspected by Codex. This is **model selection on a consumed
development set**, not a held-out result, personal-vault evaluation, or product
improvement claim. Do not train on these questions or use this ranking as a final
model promotion.

**Hypothesis:** adapters trained for the executable document loop will execute more
reliably than their corresponding bases, but the older known-paper checkpoints may
not improve cross-paper learning answers. Sonnet should be a strong external
reference, not an equal-cost deployment candidate.

**Expected signal:** higher source-supported pass/partial/fail score on the same
eight question IDs, with valid submissions and exact evidence, without large
execution, repetition, or latency regressions.

**Decision rule:** rank complete eight-question runs first by blinded semantic mean
(pass=2, partial=1, fail=0), then number of passes, then fewer execution-error
episodes, then lower mean latency. Report paired changes against the matching
untrained base and against Sonnet. A winner is only a **development candidate**;
deployment or a claim of trained-model improvement requires an independently
reviewed, held-out personal-vault task set. If all adapters tie or trail their
base, keep the simpler base/RAG path and diagnose the harness before training.

## Frozen protocol

- Questions: `data/research/learning_loop_code_pilot_v1.json`, IDs `learn_01`
  through `learn_08`, SHA-256 `806cbfb693c98a79d2b5f169713c5c8a9c15092cc1eb97c58a5fb325c9950f12`.
- Source corpus: `out/research/starter-2026-09-12/corpus`, content hash
  `ca28990a801741357e84438b36685c8f521817d941b203bbbd8735aba884d2aa`.
- Use the previously generated label-free candidate IDs, the common
  `CODE_EXECUTION_SYSTEM_PROMPT` plus
  `data/prompts/learning_loop_evidence_v1.txt` (SHA-256
  `dae40f66836d1b28919b9a066fc1b8b5b139fc9a508ae06f259036241ca3942e`),
  same corpus/tools, 15-step cap,
  1,024-token action cap, Qwen greedy temperature 0 and seed 42, required exact
  evidence, no vector index, four verifier recovery turns and escalation.
- One worker per run. Record adapter repo revision and exact downloaded weight
  SHA-256, base revision, run manifest, hardware, elapsed time, and billing.
- Keep anonymous outputs together for source-visible review **before** reading
  the assignment key. Codex review is disclosed as agent self-review; human
  confirmation remains outstanding. Automatic MuSiQue-style outcome reward and
  span validity are diagnostics, not the semantic ranking.

## Frozen arms

| Label | Model or adapter | Base | Selected adapter path |
| --- | --- | --- | --- |
| `base25` | untrained Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` | — |
| `sft25` | `jasonlingg/doctracerrl-sft-qwen2.5-7b` @ `71ecc4c6748694c7f9bfac24e53b63c72fd1c5ff` | same | root |
| `grpo25` | `jasonlingg/doctracerrl-grpo-qwen2.5-7b` @ `f84aa524a8f02b39e9bc0e461fe1454b994592bc` | same | root |
| `grpo25_50` | `jasonlingg/doctracerrl-grpo-qwen2.5-7b-50steps` @ `3e28a2f44ed3085a8f7eab16586bab4befd5c0b8` | same | root |
| `base3` | untrained Qwen3-8B | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` | — |
| `early3` | `jasonlingg/rlm-explorer-qwen3-8b-qasper-sft` @ `7cbe6289760d7718886fe57e3a6f4c45905a5704` | same | `epoch-1` (epoch 2 failed prior behavioral evaluation) |
| `aligned3` | `jasonlingg/qwen-envoy-qwen3-8b-qasper-sft` @ `012d3c7aa13956702846892a1176ed2a170e923e` | same | `final` |
| `v5_3` | `jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5` @ `27b912a863ff914ad45baa03624d8911dc1e17fb` | same | `artifacts/full/checkpoint-50` (card's recommended candidate) |
| `targeted3` | `jasonlingg/qwen-envoy-qwen3-8b-qasper-targeted-sft-v1` @ `fc4bf6a47dd238dd1f3b745bc8322ae8250093a2` | same | root (selected epoch-1 checkpoint) |
| `sonnet5` | `claude-sonnet-5` | provider API | same code-execution prompt and action cap |

`base3` and `targeted3` have earlier eight-question transcripts, but **must be
rerun** under this clarified prompt for the sweep. The first Sonnet run under
the old prompt is an excluded calibration artifact: its answers repeatedly
omitted the required evidence fields, sometimes because the 1,024-token
response cap cut off the end of the response, and one response had no text
block. A second Sonnet calibration run revealed a Claude action-cleaning bug
that removed the `EVIDENCE` suffix even when Sonnet wrote it. That run was
stopped and is also excluded. The cleaner now preserves the complete `SUBMIT`
line, with a regression test. A third interrupted Sonnet calibration showed
that Sonnet 5's default adaptive thinking can consume its entire response
budget and return no text. The final Sonnet arm explicitly disables thinking
per Anthropic's Sonnet 5 API and uses the same 1,024-token action cap as Qwen.
Sonnet does not accept the same deterministic sampling controls as local Qwen;
report that limitation and its API cost. The final Sonnet run is capped at
$2.6 after $2.367 in excluded calibration reservations/charges, keeping the
recorded total below the declared $5 cap.
The three archive repositories are each represented by one preregistered
checkpoint, not every intermediate optimizer step. The older Qwen2.5 arms use
their own base; cross-family rankings do not isolate training effects.

The first `base3` sweep transcript was made with PEFT 0.13.2 / Transformers
4.56.2. The early archived adapter requires the previous successful inference
environment's PEFT 0.21.0 / Transformers 4.57.6, so the pod was upgraded.
`base3_preupgrade` is retained for provenance but excluded from the ranking;
rerun `base3` under the upgraded versions after the adapter arms finish.

The GPU sweep uses the user's stopped Runpod pod if it can restart, with a
$5 incremental GPU-compute ceiling. Sonnet API spend is capped at $5 by
terminating the run if the projected cost reaches that limit. Save partial
transcripts and mark incomplete arms explicitly; never score an incomplete arm
as if all eight questions finished. Stop the pod after artifacts are copied and
verified.

## Results

All ten arms completed the same eight question IDs. The nine local Qwen arms
ran on pod `l9pxr8ptq4krop` (NVIDIA RTX PRO 4500 Blackwell Server Edition,
PEFT 0.21.0, Transformers 4.57.6). The Sonnet API returned `claude-sonnet-5`
on all 64 final-run requests. Both used the same question, corpus, prompt,
tools, step limit, evidence verifier, and 1,024-token action cap. Qwen used
greedy decoding with seed 42; Sonnet used provider-default sampling with
thinking disabled. Its client was macOS and provider hardware is undisclosed,
so latency is descriptive rather than a controlled hardware comparison.

Codex scored all 80 shuffled, source-visible answers before opening the model
assignment key. An empty or escalated answer is a fail. A real citation is not
enough if the selected quote does not support the claim. The score is 2 for a
pass, 1 for partial, 0 for fail; the mean is over eight questions. The
individual verdict notes, blinded assignments, automatic diagnostics, run
manifests, exact adapter hashes, and cost ledger are in
`out/research/learning-loop-model-sweep-20260926/`. This is **agent
self-review of agent-authored development questions**, not an independent
human or held-out evaluation.

| Rank | Arm | Pass / partial / fail | Mean / 2 | Submitted | Error steps | Mean seconds / question |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Sonnet 5 | 3 / 3 / 2 | 1.125 | 6/8 | 9.4% | 47.9 |
| 2 | Qwen3 v5 SFT, checkpoint 50 | 0 / 5 / 3 | 0.625 | 8/8 | 0% | 15.6 |
| 3 | Base Qwen3-8B | 0 / 4 / 4 | 0.500 | 5/8 | 5.4% | 21.8 |
| 4 | Qwen3 targeted SFT v1 | 0 / 2 / 6 | 0.250 | 7/8 | 34.2% | 94.2 |
| 5 | Qwen3 aligned SFT | 0 / 1 / 7 | 0.125 | 1/8 | 6.5% | 71.4 |
| 6 | Qwen3 early SFT | 0 / 0 / 8 | 0.000 | 0/8 | 0% | 38.9 |
| 7 | Base Qwen2.5-7B | 0 / 0 / 8 | 0.000 | 0/8 | 28.0% | 28.1 |
| 8= | Qwen2.5 GRPO and GRPO 50-step | 0 / 0 / 8 each | 0.000 | 0/8 each | 28.0% | 39.1 each |
| 10 | Qwen2.5 SFT | 0 / 0 / 8 | 0.000 | 0/8 | 28.0% | 43.7 |

The two Qwen2.5 GRPO repositories contain **byte-identical adapter weights**
(SHA-256 `1f44ac5cb716f2947fb23af7426865a4d8af31074b8df13934b1a7b4813c748d`),
so they are one unique model despite two repository names. They tied on the
answers and automatic diagnostics; their seventh/eighth place ordering is
merely tiny measured latency variation, not evidence of a quality difference.
The archived repositories are represented by the selected checkpoints listed
above, not by every saved optimizer step.

Within Qwen3, v5 is the leading **development candidate**. Against base Qwen3
on the eight paired questions, it has 2 wins, 1 loss, and 5 ties, with a mean
gain of 0.125 points out of 2, but **neither model received a single full
pass**. The v5 outputs were faster here and had no detected execution-error
steps; several accepted answers still misstated a result or conflated a valid
citation with semantic support. The targeted SFT arm loses 3 questions to base,
wins 1, and ties 4; one repeated-action case took 686 seconds. None of the
Qwen2.5 arms produced an answer. The published GRPO adapters cannot be called
improvements on this task.

Sonnet beat v5 on 5 paired questions, lost 1, and tied 2. It fully answered the
Adaptive-RAG/CRAG comparison, Search-R1 masking, and historical-corpus
freshness-limit questions. It still missed the ReAct result's exact supporting
span and escalated the personal-vault-transfer and local-experiment questions.

| Question | Base Qwen3 | v5 SFT | Sonnet 5 |
| --- | --- | --- | --- |
| ReAct loop and reported result | Fail | Fail | Partial |
| RAG versus Self-RAG | Fail | Partial | Partial |
| Adaptive-RAG versus CRAG | Partial | Partial | Pass |
| Search-R1 loss masking | Partial | Partial | Pass |
| Citation validity versus support | Partial | Fail | Partial |
| Transfer to the personal vault | Partial | Partial | Fail (escalated) |
| September 2026 freshness limit | Fail | Partial | Pass |
| ReAct versus one-pass RAG experiment | Fail (escalated) | Fail | Fail (escalated) |

Its final eight-question run used 64 settled API requests at a ledger-estimated
**$1.120368**. Three excluded Sonnet calibration runs added approximately
$2.367346 in settled ledger charges, so the recorded settled total was
$3.487714; two interrupted calibration reservations remain unsettled and are
not included as charges. The first two calibration runs revealed a harness
problem that discarded the `EVIDENCE` suffix of `SUBMIT`; the action cleaner
was fixed and tested before the final run. The third showed that default
adaptive thinking could exhaust the action cap, so thinking was disabled for
the final matched action budget.

The pod was stopped after copying the nine GPU runs locally and verifying all
42 remote-output checksums. Runpod reported `desiredStatus: EXITED` on a
subsequent `pod get`. The score is a screening result only: the questions and
references were authored and already seen by Codex, the corpus is six
historical papers rather than the user's Obsidian vault, there is one run per
arm, and no independent human review. The **model-improvement gate is not
met**. A claim that v5 beats base Qwen on the product task needs a separately
reviewed, held-out personal-vault set with the same code-execution protocol,
blind answer-support review, and acceptable failure, latency, and cost
trade-offs. This sweep does not justify another training job yet.
