# Envoy / Qwen research-agent handoff

> **October 9: QASPER attribution and official scoring added.** The visible name
> is **QASPER Agent Studio — a Qwen Envoy case study**, with Ai2/Dasigi et al.
> attribution and no claim that the questions are original. Existing repository,
> package, model and artifact identifiers are unchanged. Read
> `docs/QASPER_AGENT_STUDY.md` and `release/qasper-agent-study/README.md`.
>
> Official Answer F1, scored offline with the pinned unmodified evaluator and
> all 86 original annotations, is **19.9154 base / 30.6873 v5** on the selected
> 40 questions. The fixed answerable group worsens **29.8307 → 21.3746**;
> the unanswerable group rises **10 → 40**. Missing predictions remain zero,
> literal refusal paraphrases are not rewritten, and original inference/review
> JSON is unchanged. This is not the full QASPER leaderboard or independent
> support review. The report documents the short-target construction and
> 38/40 three-action v5 episodes as a hypothesis, not a causal training finding.
>
> One authorized NVIDIA/Nebius reference attempt stopped at its conservative
> $2 estimated-cost guard: **$1.947862**, 366 API requests, 38 completed episodes,
> the 39th interrupted, and the 40th unattempted. Preserve
> `release/qasper-agent-study/nebius-run/results.partial.json`, manifest and
> usage budget. The run is explicitly incomplete; there is no full-run F1 or
> completed-run Studio row. An asynchronous request to allow up to $3 total
> was sent but no approval was received before this update. Do not infer approval
> from elapsed time. API calls have stopped; no Runpod pod was launched.
>
> Source/scoring changes are in `e99b165`; Studio changes are in `815ea5f`.
> Every recorded implementation hash from the hosted attempt matches `e99b165`
> (the run started from a dirty tree over `e82e33b`; see EXPERIMENT.md).
> The viewer adds official F1 alongside the existing provisional support scores.
> It supports separately labeled complete supplemental runs without merging
> their reviews. The original four run rows remain, and logprob smokes stay
> unscored. The packaged export includes all 562 original turns.
>
> Validation: **784 tests passed** after the feature changes; 14 focused viewer
> tests passed again after the fairness-label follow-up. Official Qwen scores
> reproduce byte-identically with isolated standard-library Python. Local Studio
> was restarted on port 8765 after verifying there were no active paper runs;
> opening saved runs makes no model calls. Hackathon live judge access, public
> video, sponsor feedback and actual submission remain outside this completed
> case-study packaging work.

> **October 9: public technical case study shipped.** The user chose to finish
> Qwen-Envoy as an inspectable training/evaluation project. The saved Studio is
> live at <https://jasonlingg.github.io/Qwen-Envoy/>. Commit `b53d70b` publishes
> `docs/TECHNICAL_CASE_STUDY.md`, original September 19 training records under
> `release/qwen-v5-training/`, the standard-library-only
> `python scripts/reproduce_case_study.py` checker, and GitHub Actions for
> artifact verification and Pages deployment. Public demo links and release
> status are updated in the following documentation commit.
>
> Verification: **755 tests passed** offline; the 14 new corruption/binding
> tests also passed in a clean Python 3.11 environment with only pytest.
> GitHub's verification and Pages workflows passed for `b53d70b`. Hosted HTML
> byte-matches the packaged snapshot. Browser QA confirmed all four run entries
> and exercised question navigation, full saved traces, JSON export, and token-likelihood
> expansion with no JavaScript errors or model/API requests. Six training
> archive checksums and three original data bindings passed. The historical
> training manifest still has a blank Git commit and a script hash that differs
> from the current script; exact original weight reproduction is not claimed.
>
> The saved inference/review JSON is unchanged: base/v5 provisional passes
> 5/40 versus 15/40, answerable passes 3/20 versus 3/20, recorded-error episodes
> 22/40 versus 1/40. Independent source/answer review remains unfinished. No
> trained-model promotion, new training, paid inference, or Runpod launch was
> performed. The public viewer is read-only and needs no credentials. It is
> not a live hosted paper-agent service or a submitted hackathon entry; live
> judge-access instructions, video, sponsor feedback, and submission remain.
> Preserve this scope rather than starting another product pivot.

> **October 8: release scope frozen and accumulated work committed.** The user
> chose to ship the current text-only EnvoyBench Studio, with simple runs,
> scores, answers, and complete tool traces. GTLM, diagram generation, new
> partial-paper tasks, and further training remain outside this release.
> README now leads with the packaged demo and September 30 provisional results,
> explains PDF/table/image limitations, and separates historical prototypes.
>
> The user explicitly authorized committing all pending project work. Changes
> are grouped into runtime, training/data, research tools, Studio/release, and
> documentation commits. Credentials, ignored corpora, checkpoints, and local
> `out/` artifacts remain outside Git; no push or paid inference was performed.
> The offline full suite passed **741 tests** after restoring the existing
> web-evidence navigation link on the three Studio pages. The standalone HTML
> has the same link and unchanged embedded run data; all 22 release checksums
> pass. Existing model-assisted scores and training artifacts are unchanged.

> **October 6: Studio finishing pass completed locally.** The user wants a simple
> runs list, scores, and single-run transcripts, not another product pivot or
> comparison dashboard. Launch `python scripts/launch_studio.py`; optionally use
> `--live-nebius` with the local `.env` key and Docker. Port 8765 serves the four
> packaged Qwen runs. `release/envoybench-v0.1/studio.html` is standalone and
> embeds all 562 September 30 turns; the two-question October 3 token diagnostic
> is separate and unscored. The 40-question model-assisted grades are unchanged.
> No demonstrated held-out supported-answer improvement is claimed.
>
> Four bounded live NVIDIA/Nebius attempts are documented in
> `release/envoybench-v0.1/LIVE_VALIDATION.md`: three Lightning failures and one
> Ultra completed protocol with a correct answer but incomplete citations.
> Do not call it a semantic-support pass or reliability estimate. All local
> original runs are in `out/envoybench/paper-workspace/runs/`; the release includes
> plans, usage, hashes, and an assistant source-review summary. No more live tests
> are needed for this finish pass. No Runpod GPU or training was launched.
>
> Focused checks: 73 policy/investigator/paper API tests plus 9 Studio/export/
> verifier tests passed; frozen artifact verification passed. UI navigation is
> Runs → Questions → Trace, with optional per-turn token likelihood. New paid
> paper runs require an explicit Run click. Demo script and Devpost draft are
> in the release directory. Public hosting, video, sponsor feedback, and actual
> submission remain pending; local completion is not hackathon submission.
> Preserve the large preexisting dirty worktree and do not blanket-commit it.

> **October 1: Studio paper workflow and ML recipe.** User authorized completing
> the live-paper integration and packaging the trained-model case study. Studio
> adds `/papers` (web link or upload/read/question/run/history/evidence/export) and `/model`
> (observed v5 QLoRA recipe, pinned identities, current limitations). See
> `benchmarks/envoybench/PAPER_READER.md`. The new API uses the existing
> Docker-only public-paper investigator; one bounded job at a time, persisted
> progress, immutable source checks, no benchmark grades for uploads. Enable it
> with `--paper-models benchmarks/envoybench/paper-models.example.json` and an
> already served v5 endpoint. No endpoint is configured in the current shell or
> `.env`; port 8000 is not serving a model. Do not describe scripted integration
> tests as a real v5 answer. No paid compute, new training, or prefix-cache
> measurement was run in this change. The original 40-question scores remain
> unchanged. Follow the reader guide for focused verification and connection.

> **September 30 current priority: EnvoyBench case study and Studio.** The user
> approved finishing the narrow base/v5 comparison and improving its diagnostics,
> rather than starting more training or another product pivot. See
> [`benchmarks/envoybench/COMPARISON_2026_09_30.md`](benchmarks/envoybench/COMPARISON_2026_09_30.md)
> and the benchmark README for current commands; the older product states below
> remain historical. The completed 40-question QASPER-derived known-paper run is
> in `out/envoybench/provisional-full-20260930/`. Separate Codex-assisted blind
> grading gives base 5/40 and v5 15/40, but **both pass only 3/20 answerable
> questions**. Source references remain unvalidated. Sonnet grading stopped at
> eight questions when API credits ran out. There is no human-reviewed promotion
> or demonstrated substantive-answer improvement.
>
> Current improvements: analytical always-refuse expectation 20/40 with zero
> useful answers; separately bound AI response-behavior labels (v5 abstains on
> eight answerable-labeled questions); three curated paired correctness/support
> diagnoses; automatic-v2 returned-error/timing fixes; future token telemetry;
> and a standalone read-only Studio export. Version-2 errors affect 22/40 base
> episodes and 1/40 v5 episodes. Base timing covers only 29/40; missing duration
> and usage are not reconstructed. Original run and quality-grade files were
> preserved. Relevant checks: **115 tests passed**, Ruff clean, actual desktop
> and mobile browser QA passed.
>
> The deterministic within-known-paper retrieval comparator has 40 prepared
> packets / 80 planned requests at
> `out/envoybench/retrieval-baseline-prepared-20260930/`, **not executed**. Its
> pre-run hypothesis and decision rule ask whether the agent loop earns its
> complexity; preparation alone is not a result. An eight-question independent
> calibration packet at `out/envoybench/calibration-20260930/review-packet.json`
> is entirely blank. Do not fill human judgments automatically. The improved
> snapshot is `out/envoybench/studio-snapshot-improved-20260930.html`; the local
> viewer uses port 8766. The temporary A40 was terminated after the original run;
> the subsequent improvements made no new GPU jobs or external model API calls.
> Keep remaining inference and independent-review work explicit; do not claim
> the prepared retrieval baseline has been measured.

> **September 28 scope correction:** the user has no substantive existing
> Obsidian vault. The first new model task is multi-step Qwen investigation of
> a frozen public AI-paper library; Nemotron/the reviewed writer creates the
> Obsidian notes. Do not keep blocking the public-library baseline on the
> one-note personal vault or describe a public-paper result as private-vault
> transfer. See
> [`docs/QWEN_RESEARCH_LIBRARY_CORPUS_AMENDMENT_20260928.md`](docs/QWEN_RESEARCH_LIBRARY_CORPUS_AMENDMENT_20260928.md).

> **September 28 implementation update:** the original-paper library now has a
> separate read-only MCP server in `scripts/research_library_mcp.py`. A real
> stdio client searched the W39 paper snapshot, opened an exact source span,
> and confirmed that unconfigured Qwen made zero model requests. The product
> Qwen investigator has an explicit `public_papers` prompt; no live Qwen call
> has been made on this path. The executable `search()` default was repaired
> from whole-document term counting to versioned passage BM25 before any new
> model comparison: first-query top-five source-ID coverage on four provisional
> development questions rose from 2/7 to 5/7 (and 7/7 at top eight). This is retrieval infrastructure,
> not Qwen improvement. See
> [`docs/RESEARCH_LIBRARY_DEV_RETRIEVAL_DIAGNOSTIC_20260928.md`](docs/RESEARCH_LIBRARY_DEV_RETRIEVAL_DIAGNOSTIC_20260928.md).
> Twelve held-out question drafts are structurally checked but remain blind,
> unreviewed, and unused for model selection. The W39 Obsidian seed is still an
> unpublished agent-authored draft awaiting a person's claim/relevance review.
> The development-only product-path runner in
> `scripts/run_research_library_qwen_dev.py` has a current no-model dry run at
> `out/research/research-library-dev-qwen-dryrun-v2-20260928/` with a pending
> reference-review template; live mode rejects that pending record before any
> Docker or model action. No base/v5 paper-library model run has occurred.
> A September 28 Runpod MCP read found zero pods and zero endpoints, plus one
> 50 GB standard network volume in EU-RO-1. A 32 GB RTX PRO 4500 was listed
> there at $0.72/hour with medium availability; this is a timestamped price
> and stock observation, not a provisioned resource. Account credit balance
> was not exposed by these MCP reads.

**Last updated:** 2026-10-08 (America/Toronto)  
**Repository:** `/Users/jasonling/Documents/GitHub/rlm-explorer`  
**Purpose:** working memory for a new Codex/Claude/other engineering agent. Read `AGENTS.md`,
`docs/GPT_ASTRA_IMPROVEMENT_BRIEF.md`, and this file before acting.

## Current September 27 product state

- The chat-first learning-memory demo has a separate ten-note **synthetic** linked vault at
  `data/product_memory/linked_demo_vault`. `python scripts/launch_sample_demo.py` serves it on
  localhost in evidence-only mode, and normal Ctrl-C/SIGTERM cleanup removes its temporary copy.
  Its one-command HTTP smoke returned exact passages and linked notes without remote model calls.
- The browser now has a source-triggered revision loop: a captured **Result** gets a local
  prior-decision candidate, exact passages from both notes, and a dated timeline. The user can
  inspect both, investigate in chat, replace a local correction scaffold, and explicitly approve
  an append-only Obsidian note that supersedes the old decision. A normal source/plan gets no
  correction scaffold. Candidate retrieval uses links and word overlap, not semantic conflict
  detection. The optional **Draft with Nemotron** button makes one explicit model request for a
  candidate result and still requires human review. Browser flow and focused mocked-API tests
  pass; no paid model call was made for this new draft feature. See
  `docs/PERSONAL_MEMORY_LOCAL_DEMO.md`.
- A separate **real-report-derived** Qwen learning thread can be launched with
  `python scripts/launch_qwen_experiment_demo.py`. Its three linked Obsidian notes are generated
  from `reports/qasper-scale-sft-2026-09-25.json` with source hashes: retrospectively inferred
  continuation strategy, measured 15/10/4-of-40 QASPER result, and the report's rejection
  decision. They explicitly disclose model-assisted review and that this is neither base-vs-SFT
  nor personal-vault evidence. The UI now lists existing result notes for later impact review.
  An isolated integration test checks local impact, evidence-only chat, test-only approval, later
  recall, and read-only MCP source access. No trained Qwen or Nemotron call occurs in this demo;
  no actual user-approved revision is bundled. See `docs/QWEN_EXPERIMENT_MEMORY_DEMO.md`.
- Two separately triggered **live Nemotron** turns over this public thread are recorded in
  `reports/qwen_experiment_nemotron_smoke_20260927.json` and
  `reports/qwen_experiment_nemotron_smoke_20260928.json`. Both are partial after Codex source
  review: the first confused a reported rule with an inferred strategy, and the second grouped
  citations at paragraph end and omitted a factual sentence from its claims. The source note was
  clarified, and `NemotronChatClient.answer` now rejects grouped end citations across multiple
  factual sentences; a local replay of the saved second response was rejected. This checks
  citation placement only, not semantic support. No trained Qwen checkpoint was run.
- A live Nebius/Nemotron synthetic chat path answered one dated plan/correction question and
  correctly said another planned Qwen comparison had **no measured outcome**. Reports are in
  `reports/nebius_linked_demo_smoke_20260927_v3.json` and
  `reports/nebius_linked_demo_no_outcome_20260927.json`. Exact spans were checked; semantic
  support still awaits independent human review. Nemotron sometimes groups citations at the end
  of an answer. A stricter prompt caused an evidence-only fallback and was reverted.
- `scripts/personal_memory_mcp.py` is a working local, read-only stdio MCP server over a frozen
  vault snapshot. It offers `search_memory` and review-bound `get_memory_source`; it does not
  invoke Qwen or expose a weekly-digest tool. The real stdio client test passes.
- `scripts/weekly_radar_candidates.py` now persists two-topic, version-aware candidate stores
  using arXiv submission search and a bounded latest-update scan. Live September 21–27 stores in
  ignored `out/research/weekly-radar-candidates/` hold 54 tool-agent and 11 small-model-training
  candidates; these are unranked metadata, not a recall benchmark. See
  `docs/WEEKLY_RADAR_CANDIDATES.md` for coverage limits.
- `scripts/weekly_digest.py` validates a frozen paper snapshot, renders linked paper notes and a
  weekly Obsidian digest, and publishes only a selection explicitly marked `human_reviewed` with
  `--confirm`. The historical 2025-W32 and current 2026-W39 selections under `data/research/`
  are **agent-authored abstract-based drafts only**. Their rendered previews are in ignored
  `out/research/weekly-digest-*-agent-draft-preview.txt`; neither was written to the vault.
  Both preview batches have zero unresolved links. See `docs/WEEKLY_DIGEST_DEMO.md`.
- The paired personal-vault Qwen base/v5 evaluation remains **unrun**. The pinned v2 questions,
  source bundle, and runner are prepared (see `docs/PERSONAL_MEMORY_READINESS_EVAL.md`), but the
  Runpod CLI has no API credential in this workspace. **The Runpod OAuth MCP connection does
  work through Claude Code without an API key**; direct Runpod tools are not exposed in this
  Codex chat. On September 27, a read-only MCP call showed all 24 pods `EXITED`, and a START
  attempt on cached-weight pod `l9pxr8ptq4krop` failed with Runpod's
  `There are not enough free GPUs on the host machine to start this pod.` The pod remained
  `EXITED`; no new pod or GPU runtime was created. A separate Nebius-hosted Qwen3-30B
  Docker-tool smoke ran on two synthetic questions:
  the original prompt made no tool calls, a first-search instruction enabled tool calls, the
  temporal answer had incomplete cited support, and the absent-outcome answer mistook a plan for
  a measured result. See `reports/nebius_qwen30b_memory_smoke_review_20260927.md`. This is not
  the pinned 8B base/v5 comparison. Do not describe CPU lexical results, Nemotron smoke tests,
  or the hosted 30B smoke as Qwen checkpoint improvement.
- Focused product/radar/MCP/eval checks: **80 passed**, including the launcher SIGTERM test;
  Ruff passed on the touched product files. No
  training or reward code was changed in this product pass.

## One-paragraph state

Envoy is a small Qwen research worker that writes Python in a persistent REPL to search, read, and
extract evidence from AI papers or an Obsidian snapshot. A larger model such as Claude or GPT is the
orchestrator. The latest experiment continued a Qwen3-8B rank-4 QLoRA adapter on 995 QASPER-derived
code-execution trajectories. Its epoch-1 and epoch-2 checkpoints were recovered and evaluated on
the same reserved 40-question set as the starting adapter. Identity-blind model-assisted semantic
review found 15/40, 10/40, and 4/40 passes respectively; neither continuation checkpoint was
promoted. All 120 episodes are backed up locally, and the migrated RunPod pod is stopped. The next
product experiment is a reviewed development baseline on the frozen public-paper library,
followed by a separate held-out comparison only after a teachable failure is identified. Another
long training run is not yet justified. The RunPod recovery and pending-evaluation
sections below are historical records;
do not act on their old operational instructions.

Final artifacts: `out/research/qasper-scale-sft-v1/eval-20260925-decision.json`,
`out/research/qasper-scale-sft-v1/eval-20260925-failure-analysis.json`, and
`reports/qasper-scale-sft-2026-09-25.json`. A no-GPU demo with actual recorded Qwen actions is
documented in `docs/DEMO_AND_EVIDENCE.md`.

## Product goal and architectural boundary

The product is a weekly AI-research radar and personal second brain:

1. Discover and freeze relevant new AI papers.
2. Let Qwen investigate a bounded research question through executable Python tools.
3. Return a compact evidence packet with exact source spans and limitations.
4. Save durable notes and weekly digests in Obsidian.
5. Expose high-level read-only research operations through MCP to Claude, GPT, or another host.

Qwen is not meant to replace the larger reasoning model or serve as a general chat interface. Its
job is the cheap, repeatable middle layer between the host model and the paper library. It still
needs enough local judgment to reformulate a search, inspect a paper, recover from a weak result,
and abstain when evidence is insufficient.

Primary product documents:

- `docs/CODE_EXECUTION_SECOND_BRAIN.md`
- `docs/WEEKLY_RESEARCH_RADAR.md`
- `docs/QASPER_TARGETED_SFT.md`
- `docs/QASPER_FAILURE_ATTRIBUTION.md`
- `docs/QASPER_RL_EXPERIMENT.md`

## Current training data

The latest dataset is `out/research/qasper-scale-sft-v1/` (ignored by Git).

- Total conversations: **995**
- Answerable: **955**
- Previously reviewed abstentions: **40**
- Train: **796 conversations / 472 papers**
- Validation: **199 conversations / 122 papers**
- Train SHA-256: `9db1059ccaa8065dd23c0d337d3d18aa2994be75cf4fd56a6501b89036163a04`
- Validation SHA-256: `8bc3b0fad76ff0579ca1610609b7535ee1c12fc96ea97faa19ec9a67a2066dfc`
- Dataset manifest SHA-256: `96a0a9271dda5f003d682fb6e7b1067a2cbe50cfc5379e3c23646ce05effe201`
- Seed: `20260921`

The 955 answerable trajectories were constructed deterministically from QASPER train annotations.
Every trajectory was replayed mechanically; train/validation splits are paper-disjoint; reserved
papers do not overlap; identical question/action traces are deduplicated; and five deliberate
corruptions were rejected. Twenty answerable examples received a source-visible semantic spot
check by Codex and all passed. This was **model-assisted review, not independent human review**.

Audit: `out/research/qasper-annotation-scale-v1/audit-final-reviewed.json`  
Audit SHA-256: `6eaeb77293dc037c75bddd92b9d0c009e47cb4ff147d6acead1c3159fe78bc3c`

Data-building code added or changed during this work includes:

- `scripts/prepare_qasper_rl.py` (`--answerability` filter)
- `scripts/build_qasper_annotation_sft.py` (paper-level splitting and deterministic deduplication)
- `scripts/verify_qasper_synthetic_data.py` (replay, split, identity, and mutation checks)
- `scripts/assemble_qasper_scale_sft.py` (combines answerable data with reviewed abstentions)
- `tests/test_qasper_annotation_builder.py`

## Latest Qwen3-8B SFT run

The run continued from the public adapter:

`jasonlingg/qwen-envoy-qwen3-8b-qasper-targeted-sft-v1`

The public repository corresponds to the earlier selected `checkpoint-150`; its
`adapter_model.safetensors` SHA-256 is:

`70143ec6933494efe60cab726e792fcc182965fd13a8561453202b41bba9bfb4`

Latest training configuration:

- Base: `Qwen/Qwen3-8B`
- Base revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Method: 4-bit NF4 QLoRA continuation
- LoRA: rank 4, alpha 8, dropout 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Format: one next-action example per assistant action, with exact inference-prefix verification
- Loss: assistant action tokens only; retrieved/tool-output text is context and receives no SFT loss
- Native Qwen3 thinking: disabled
- Maximum sequence length: 8,192 tokens
- Longest train example: 5,641 tokens
- Train actions: 3,711; validation actions: 924
- Learning rate: `5e-5`, cosine schedule, 5% warmup
- Epochs: 2
- Batch size: 1; gradient accumulation: 4
- Seed: `20260921`
- Hardware: secure RunPod NVIDIA RTX PRO 4500 Blackwell, 32 GB VRAM
- Runtime: 3h 19m 54s for 1,856 optimizer updates

The one-step smoke test succeeded before the full run:

- Smoke validation loss: `0.1317699999`
- Smoke validation token accuracy: `0.9675968`
- It completed a real update, full validation, and saved a reloadable adapter.

Full-run final metrics:

- Final validation loss: `0.0610487126`
- Final validation token accuracy: `0.9810336`
- Final train loss: `0.0538664`
- Final and checkpoint-1856 adapters are byte-identical.

These metrics show that the model learned the trajectory distribution. They do **not** establish
better autonomous research behavior.

Remote run directory:

`/workspace/envoy/out/research/qasper-scale-sft-v1/run-20260921-epoch-checkpoints`

Saved candidates and checksums:

- Epoch 1: `checkpoint-928/adapter_model.safetensors`
  - SHA-256: `7c259d36b196a7a9541d152be9f192ebb600ad1fd3845ee9ae07847c71f40b1c`
- Epoch 2: `checkpoint-1856/adapter_model.safetensors`
  - SHA-256: `6ae2fdcd33ddc18c9fd2ed04011953106268c6b735f5ce7211ba7603e7175d66`
- Final: `final/adapter_model.safetensors`
  - SHA-256: `6ae2fdcd33ddc18c9fd2ed04011953106268c6b735f5ce7211ba7603e7175d66`

The epoch-1 validation metrics still need to be extracted from its `trainer_state.json` or the full
remote training log after the pod becomes accessible.

## Historical RunPod recovery (resolved September 25)

The operational notes below describe the pre-migration state. The adapters and eval transcripts
were recovered, and target pod `1t43mrc8ko4ajp` is **EXITED**. No recovery action is pending.

Pod ID: `nhbh2q7mcnfcdi`  
Name: `envoy-qasper-scale-sft-secure-20260921`  
Cloud: secure  
Original GPU: RTX PRO 4500 Blackwell  
Rate when running: `$0.72/hour`  
Persistent mount: `/workspace`, 40 GB  
Status on 2026-09-24: **EXITED / not billing**

The pod cannot currently restart because its local volume is tied to a host whose GPU was rented by
someone else. Several API start attempts returned: `There are not enough free GPUs on the host
machine to start this pod.` At the latest check, the general RTX PRO 4500 pool had low stock and the
A40 pool was out of stock, but capacity changes frequently.

The available RunPod MCP can start, stop, inspect, create, and delete pods, but it does not expose
the console's **Automatically migrate your Pod data** action. The user's logged-in RunPod tab was
not available to browser control. The user must either:

1. click **Automatically migrate your Pod data** in the RunPod console; or
2. start the stopped pod with CPUs, allow the agent to SSH in and back up/transfer the adapters,
   then create an available GPU pod.

Do not terminate this pod. The latest two adapters have not been downloaded locally or uploaded to
Hugging Face and currently exist only on its persistent disk.

Once the pod is accessible, immediately back up both `checkpoint-928` and `checkpoint-1856` before
another stop. The user previously authorized saving the trained weights to their Hugging Face
account, but behavioral evaluation should select the promoted checkpoint. Back up both candidates
even if only one is ultimately published.

## Completed behavioral evaluation (original plan)

The controlled comparison described below was completed on September 25. The starting adapter
passed 15/40 semantic reviews, epoch 1 passed 10/40, and epoch 2 passed 4/40. Both continuation
checkpoints failed the preregistered promotion rule. The following text preserves the original
plan for provenance; it is not a to-do list.

The immediate experiment is a controlled three-way comparison:

1. Starting public adapter (`qasper-targeted-sft-v1`, checkpoint-150)
2. New epoch-1 adapter (`checkpoint-928`)
3. New epoch-2 adapter (`checkpoint-1856` / `final`)

Use the exact same question IDs, corpus, system prompt, tool interface, maximum steps, decoding
settings, and seed. The primary question is whether either continuation checkpoint improves
supported-answer quality over the starting adapter. Training loss and token accuracy are only
diagnostics.

Reserved confirmation data:

- Benchmark: `out/research/qasper-code-confirm-v1/benchmark.json`
- Benchmark questions: **40**
- Benchmark SHA-256: `23c98a28bfe2d4ed9f3e44b74f0f28f6ea7d244b265048f13ccbc4f45d6f2a7e`
- Corpus: `out/research/qasper-code-confirm-v1/corpus/`
- Manifest: `out/research/qasper-code-confirm-v1/manifest.json`
- Manifest SHA-256: `86c9e7a67056ef3d79d87bfb5fdc11780c4b36e03c67a3e734cb8a44add5eb55`
- Source split: QASPER validation; latest scale training uses QASPER train.

This confirmation set was reserved for the latest SFT comparison. Keep it distinct from the
already-consumed GRPO confirmation under `out/research/qasper-rl-v3/confirmation/`.

Before evaluation, freeze a machine-readable plan with model/checkpoint hashes, exact 40 IDs,
corpus hash, prompt hash, decoding settings, maximum steps, seed, GPU, and code revision. Run an
identity-blind semantic review of all outputs. Report at least:

- semantic pass / partial / fail;
- supported-answer rate;
- answerability/false-refusal behavior;
- citation and exact-span validity;
- required-paper recall;
- valid submissions;
- syntax/runtime errors and repeated actions;
- paired per-question changes; and
- latency, generated tokens, and estimated GPU cost.

The earlier project gate was at least 28/40 semantic passes plus a paired improvement over the
starting adapter. Do not silently relax that gate after seeing outputs. If a new gate is chosen,
record it before inference.

The training pod bundle contains `src/`, `scripts/train_sft.py`, the Python environment, base-model
cache, and training data. It likely does **not** contain the local confirmation corpus or the full
evaluation scripts. After migration/CPU recovery, upload an evaluation bundle containing the
needed current code plus `out/research/qasper-code-confirm-v1/` before running inference.

## Important prior evidence

Do not rewrite history around the newest run:

- On 50 common MuSiQue dev questions, Qwen2.5-7B base scored `0.158`, SFT scored `0.176`, and the
  available GRPO adapter scored `0.172`. The two old GRPO Hub repositories contain byte-identical
  adapters, so they do not show checkpoint progression.
- Base Qwen3-8B on the 10-question AI-paper pilot scored `0.445` outcome reward with citation
  precision/recall `0.85/0.95`. Retrieval was fairly strong; answer discipline, instruction
  retention, semantic accuracy, and abstention were the main failures.
- The earlier small targeted QASPER SFT run improved development behavior only slightly and failed
  its preregistered gate. The selected epoch-1 result was 11 pass / 5 partial / 9 fail versus the
  prior adapter's 10 / 4 / 11, with only 1/15 target failures rescued.
- Centered GRPO improved an already-consumed development set but did not replicate on its fresh
  paper-disjoint confirmation: automatic reward changed from SFT `0.3462` to GRPO `0.3531`, while
  blind semantic review favored SFT 13 pass / 2 partial / 25 fail versus GRPO 12 / 3 / 25.

The latest scale-SFT run was motivated by the conclusion that tens of trajectories were too small
and that reliable annotation-derived trajectories could be generated programmatically at scale.
Its behavioral result is still unknown.

## Reasoning / chain-of-thought decision

Current Qwen training is **not chain-of-thought training**. It learns visible, executable tool-use
trajectories: write Python, observe tool output, choose the next action, then submit a cited answer.
Qwen3 native thinking is disabled because an earlier baseline spent its token budget inside
`<think>` and never reached executable code.

Search-R1 is relevant precedent for a later experiment: it uses RL to interleave bounded reasoning
and search while masking retrieved tokens from policy and KL losses. If the new SFT behavioral
evaluation still shows planning failures, a legitimate follow-up is a controlled GRPO condition
with a short reasoning budget (for example, at most 96 tokens per action), explicit reasoning
delimiters, and code executed separately from the reasoning text.

Do not implement this as merely `enable_thinking=True`. Required changes would include:

- parser/harness support for bounded reasoning plus executable code;
- a hard reasoning-token cap and action budget;
- regression tests proving injected tool outputs receive zero policy and KL loss;
- unchanged no-reasoning control;
- a small development diagnostic before a costly run; and
- evaluation based on supported answers, not whether thoughts sound plausible.

This is optional. Because a larger host model performs global reasoning, Qwen only needs local
planning if the evaluation shows it chooses poor searches or gives up despite available evidence.

## Working-tree safety

The repository is heavily dirty with user and prior-agent work. Do not use blanket `git add`,
`git reset`, or checkout commands. Preserve unrelated changes. No commit was made for the latest
dataset-generation and training-preparation work.

Especially preserve the GRPO fixes and related documentation called out by `AGENTS.md`. Before a
commit, inspect `git status --short` and stage only explicitly reviewed paths.

This handoff file itself is newly added and is not committed unless a later agent/user explicitly
commits it.

## Recommended next actions

1. Ask the user to complete RunPod's **Automatically migrate your Pod data** action, or CPU-start
   the old pod for data recovery.
2. Verify the migrated/recovered checkpoint hashes against the values above.
3. Back up both new adapters locally or to Hugging Face before stopping or terminating anything.
4. Upload the current evaluation code and frozen 40-question confirmation corpus.
5. Record the evaluation manifest and decision rule before inference.
6. Run the starting adapter, checkpoint-928, and checkpoint-1856 under identical settings.
7. Perform blind semantic review, reveal identities, and report paired changes.
8. Stop GPU billing immediately after artifacts and logs are safely copied.
9. Decide from failure attribution whether the next intervention is data, harness/verifier work,
   bounded-reasoning GRPO, or no further training.

## Suggested prompt for the next agent

> Read `AGENTS.md`, `CODEX_HANDOFF.md`, `docs/GPT_ASTRA_IMPROVEMENT_BRIEF.md`, and
> `docs/CODE_EXECUTION_SECOND_BRAIN.md`. Preserve the dirty worktree. Recover or migrate RunPod pod
> `nhbh2q7mcnfcdi`, back up both scale-SFT adapters, then run the preregistered three-way behavioral
> evaluation on `out/research/qasper-code-confirm-v1`. Do not claim improvement from training loss.
> Keep the Qwen worker as a code-execution evidence subagent for a larger Claude/GPT host, and stop
> GPU billing as soon as all artifacts are safe.
