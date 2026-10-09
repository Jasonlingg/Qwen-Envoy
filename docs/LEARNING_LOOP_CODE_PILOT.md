# Learning-loop code-execution development pilot

## Question and decision, registered 2026-09-25

This pilot tests whether the existing Qwen code-execution harness can answer *learning* questions about a frozen six-paper collection: explain a mechanism, compare two methods, critique evidence, propose a local test, and acknowledge when the collection cannot answer. It is a development test, not a held-out personal-vault or weekly-digest product evaluation. The papers were used in an earlier pilot, and the eight questions and reference answers were authored by Codex and have not been independently reviewed. Exclude the questions, answers, grader notes, and source anchors from training.

**Hypothesis:** once label-free retrieval supplies valid candidate paper IDs, the targeted QASPER SFT adapter will produce more complete, source-supported learning answers than base Qwen3-8B, without more execution failures or unacceptable extra time. The expected signal is paired gains in blind pass/partial/fail review, especially on comparison and calibrated-uncertainty questions. We will record evidence validity, submission rate, execution failures, steps, wall time, and GPU cost separately. MuSiQue reward and citation-span validity are diagnostics, not semantic quality.

**Decision rule:** retain the adapter for a larger, independently reviewed learning-task study only if it has a higher mean blind-review score (pass=2, partial=1, fail=0), wins more question pairs than it loses, and does not reduce submission rate or valid-evidence-question rate. If tied or worse, do not train on this recipe; inspect the failure classes and improve the product/harness baseline first. No outcome on eight development questions proves model improvement or passes either product gate. Human reviewers must confirm the reference answers and score anonymized model outputs before the blind key is opened.

## Frozen materials and protocol

- Source snapshot: `out/research/starter-2026-09-12`, corpus SHA-256 `ca28990a801741357e84438b36685c8f521817d941b203bbbd8735aba884d2aa`.
- Question set: `data/research/learning_loop_code_pilot_v1.json`, IDs `learn_01` through `learn_08` (full IDs in the file). Six answerable and two insufficient-evidence questions.
- Source-check sheet: `out/research/learning-loop-code-pilot-v1/source-check.md`. The anchor strings were mechanically found in the frozen papers; human reference review remains pending.
- Candidate discovery: BM25 over 512-character passages, 64-character overlap and title boost, top four paper IDs. This uses question text only, never answers or required-document labels. The candidate-ID variant and retrieval audit are in `out/research/learning-loop-code-pilot-v1/candidate-ids/`. Required-paper candidate recall is 9/9 across the six answerable questions; the two abstention questions have no required paper.
- Base: `Qwen/Qwen3-8B` at its recorded loaded revision. SFT: the same base plus `jasonlingg/qwen-envoy-qwen3-8b-qasper-targeted-sft-v1` (the earlier targeted QASPER adapter, **not** either rejected continued-training checkpoint). Its published `adapter_model.safetensors` SHA-256 is `70143ec6933494efe60cab726e792fcc182965fd13a8561453202b41bba9bfb4`; confirm the loaded adapter against that hash and retain the run manifest.
- Both arms: same candidate-ID question file, corpus, code prompt, tools, max 15 steps, greedy temperature 0, seed 42, evidence required, no vector index, verifier enabled with four recovery turns and escalation after persistent failure. Use one worker on the same GPU class and record software and hardware. The run manifest records hashes and decoding settings.
- Blind review: each answer must address every part, cite evidence that semantically supports substantive paper claims, label illustrative examples and proposed experiments, and avoid unsupported transfer or current-best claims. Review source quality and the reference answer before assigning a grade. Report results by question ID and learning skill as well as overall.

## Reproduce the comparison

On a GPU machine with the repository, frozen snapshot, and Python dependencies installed:

```bash
BASE_MODEL_PATH=Qwen/Qwen3-8B \
CHECKPOINT_PATH=jasonlingg/qwen-envoy-qwen3-8b-qasper-targeted-sft-v1 \
bash scripts/run_learning_loop_code_pilot.sh
```

The script creates a dated output directory under `out/research/`, validates the benchmark and source anchors, regenerates label-free candidates, runs base and SFT on the same questions, and prepares a blinded review bundle. It leaves the score pending until a reviewer fills `blind-review/review.json` and runs `scripts/review_code_exec_pilot.py score`. Keep `blind-key.json` closed until grading is complete. Record GPU rental duration and price alongside the output manifests.

## Development result — 2026-09-26

Both eight-question runs completed on one NVIDIA RTX PRO 4500 Blackwell. The same candidate-ID questions, corpus hash, Qwen3-8B base revision, greedy decoding, seed 42, verifier settings, and 15-step cap were used in both arms. The trained arm used the existing targeted QASPER adapter. The remote copy was made from an uncommitted checkout, so its two original manifests have an empty `git_commit`; `provenance.json` records the local commit and a SHA-256-identified source bundle. The Runpod pod `7ccwpe64qjlaye` was stopped after the results were copied locally.

Codex assigned verdicts from the anonymous `blind-review/review.json` before opening its assignment key. This is **agent self-review of agent-authored questions**, not the independent human review required by the registered primary metric. The score command calls its output a “human score” because it is a generic reviewer tool; the file is explicitly named `agent-score.json` here to avoid that interpretation.

| Development diagnostic | Base Qwen | Targeted SFT |
|---|---:|---:|
| Codex blind pass / partial / fail | 0 / 4 / 4 | 0 / 4 / 4 |
| Codex mean review score (0–2) | 0.50 | 0.50 |
| Paired wins | 1 | 1 |
| Submitted answers | 5 / 8 | 6 / 8 |
| Questions with valid evidence spans | 5 / 8 | 6 / 8 |
| Required-paper citation recall | 4 / 9 | 4 / 9 |
| Mean steps | 12.25 | 5.88 |
| Mean model episode time | 31.32 s | 117.90 s |

Six question pairs tied under this review. SFT gained a partial answer on the RAG-versus-Self-RAG question; base gained a partial answer on Adaptive-RAG versus CRAG. Both failed the ReAct explanation, the current-best question, and the local-experiment note. The apparent 100% valid-span rate applies only to spans the models *submitted*; it does not mean the spans support their claims. Both models made the unsupported assertion that Adaptive-RAG was the best current method in September 2026. On the Adaptive-RAG-versus-CRAG question, the trained model repeated an approximately 6,049-character action and took 765 seconds before escalating, driving its mean latency to about 3.8 times base.

The trained adapter does **not** meet the registered improvement signal in this provisional review: answer quality tied and latency regressed. Do not promote it for the learning-loop worker or start more training from this result. Independent reference and output review could change individual grades; even a favorable regrade of eight intentionally difficult, agent-written questions would remain a development diagnostic. The next meaningful evaluation needs ordinary questions drawn from the user's actual learning workflow, independently reviewed answers, a frozen richer vault, and a simple one-pass RAG baseline.

Artifacts: `out/research/learning-loop-code-pilot-run-20260926/` contains both transcripts and manifests, `blind-review/review.agent.json` with per-answer reasons, `blind-review/agent-score.json`, automatic diagnostics, the full log, package versions, hardware, source bundle, and provenance. The historical six-paper collection cannot establish what is best in September 2026 or how the assistant performs on the user's vault.
