# Targeted Qwen vault transfer test — registered protocol, 2026-09-26

> September 28 source-domain amendment: [start from a growing public-paper
> research library because no pre-existing personal vault is available](QWEN_RESEARCH_LIBRARY_CORPUS_AMENDMENT_20260928.md).
> The original personal-vault entry gate below is historical for this first test.

> September 28 prospective amendment: [initialize the new candidate from base
> Qwen and use base as the primary held-out comparator](TARGETED_QWEN_VAULT_TRANSFER_AMENDMENT_20260928.md).
> The v5-continuation treatment and promotion rule below are retained as the
> historical September 26 registration.

## What this test answers

Test whether one targeted code-trajectory SFT round makes Qwen3-8B answer
personal Obsidian research questions more completely and with better source
support than both base Qwen3-8B and the current v5 checkpoint. The proposed
intervention is **training data**, not a new prompt, sampler, retriever, tool
interface, or reward. The hypothesis is that examples connecting each requested
answer part to a checked source passage will correct v5's observed incomplete
answers and unsupported claims without breaking its reliable submissions.

This is a small before/after **transfer test**, not an attempt to make a larger
version of the eight-question development pilot. The old eight questions,
reference answers, reviews, and inspected outputs are excluded from training
and from this test. The six-paper pilot corpus is also not a substitute for a
personal-vault evaluation.

**Status:** protocol written; no test questions, reviewed gold passages, model
run, or training result yet. The repository's demo snapshot contains only two
short example notes, and the separate `Documents/AI research` Obsidian vault
currently contains only `Welcome.md`. Neither can support this test. Do not
manufacture twelve answers from them or report this protocol as an executed
benchmark.

## Source and question lock

1. Freeze a substantive, read-only Obsidian collection with
   `scripts/research_vault.py import`. Save its manifest, corpus hash, source
   hashes, parser version, and snapshot path outside the live vault. The notes
   must support actual cross-note questions; the current two-note demo fails
   this entry condition. Personal commentary must remain distinguishable from
   cited paper findings.
2. Write **four development questions** (`D01`–`D04`) and **twelve untouched
   transfer questions** (`H01`–`H12`) against that snapshot. Each needs a
   reviewed reference answer or justified insufficiency, expected answerability,
   required note IDs where applicable, exact source anchors, and pass conditions.
   A human checks the references against the frozen notes before the question
   files are locked. If only Codex reviews them, label the results provisional.
3. Keep the twelve transfer question texts, gold answers, source anchors, and
   model outputs out of all prompt tuning, data generation, and training. Record
   SHA-256 hashes of the locked snapshot and both question files. Candidate
   training source notes and questions must be disjoint from the transfer
   answers and their supporting source notes. Store training provenance so this
   exclusion can be audited.

The twelve transfer slots specify coverage **before** their wording is written:

| IDs | Skill | Pass requires |
| --- | --- | --- |
| H01–H03 | Explain a finding from one note | Correct mechanism/result, numerical context where relevant, exact supporting passage. |
| H04–H07 | Compare or connect two notes | Both sides and their limits; at least one inspected passage per material side. |
| H08–H09 | Critique a plausible but unsupported claim | Identify what the cited passage does and does not establish. |
| H10–H11 | Handle missing or stale evidence | Explicit uncertainty, no invented current ranking or vault-specific result, and what evidence is missing. |
| H12 | Apply a finding to a local project decision | Separate published evidence from the proposed local test and name a stop condition. |

Development questions cover one example each of multi-part explanation,
cross-note synthesis, citation support, and an insufficient-evidence case.
Their job is to confirm a teachable failure and a working corpus before GPU
training. They are **not** added to the twelve-question final score.

Use the `research-benchmark-v1` question shape already used by
`data/research/learning_loop_code_pilot_v1.json`. Give the development and
transfer sets separate files with `split: pilot_evaluation`, distinct IDs and
the same frozen `corpus_hash`; that split field is a parser requirement, not a
claim that these sets are interchangeable. Run
`scripts/research_benchmark.py validate` and
`scripts/validate_learning_loop_pilot.py` on each file, then
inspect the generated source-check reports. Use **retrieved** candidate IDs
from `scripts/prepare_ai_paper_id_diagnostics.py`, never oracle IDs, and give
all models the same candidate list per question.

## Baseline gate before training

Run base Qwen3-8B and v5 on the four development questions with identical
tools, corpus, prompt, decoding, seed, step cap, and verifier. Blindly review
complete answers against their source passages and classify failures as
retrieval, tool protocol, evidence selection, unsupported synthesis,
answerability, or missing question parts. Inspect trajectories as well as
final answers.

**Decision:** proceed to a bounded SFT candidate only if a human-approved
reference set and the reviewed baseline expose recurring, teachable answer
construction or evidence-support failures. If the source snapshot cannot answer
the questions, fix the collection. If code execution or retrieval dominates,
fix that measured bottleneck first. No costly training begins merely because
the old eight development questions scored poorly.

## Single training intervention

Build a small, source-checked batch of new full Python trajectories from
**different** notes/papers. Each trajectory must actually run with the
`search()`, `read()`, and `extract()` tool protocol and end in a concise answer
whose important claims map to inspected source passages. Include multi-part
questions, multi-note comparisons, unsupported-number traps, and honest
abstention. Reject malformed, non-submitting, or ungrounded demonstrations.
Train one candidate from the pinned Qwen3-8B/v5 starting setup; preserve the
dataset IDs, source-note IDs, training script/version, seed, adapter checkpoint,
and weight hash. Do not change the active MuSiQue reward or run GRPO in this
experiment. `scripts/train_sft.py --adapter-checkpoint` supports continuing the
existing v5 LoRA adapter; use that path so the treatment is the added,
source-checked trajectory data rather than a different starting checkpoint.

## Final comparison and decision

After the candidate checkpoint is fixed, run **base Qwen3-8B**, **v5**, and
**the candidate** once on the same twelve transfer questions. Use the same
non-thinking greedy decoding (temperature 0, top-p 1), 1,024 new tokens per
action, 15 steps, evidence requirement/verifier, four recovery turns, prompt
suffix, candidate IDs, seed 42, and GPU class. Pin and record the base revision,
v5 adapter revision, candidate adapter hash, snapshot hash, question IDs, prompt
hash, reward version, library versions, hardware, duration, and cost. Save
transcripts and manifests before reviewing answers. The existing
`scripts/review_code_exec_pilot.py` can randomize identities and materialize
quoted spans for a three-arm blind review.

For each answer, grade **pass / partial / fail**. Pass requires every requested
part or a justified abstention, accurate attribution, and source passages that
actually support every material claim. A real span alone is not semantic
support. Also report paired per-question outcomes, submitted answers,
execution-error episodes and steps, repeated actions, citation/span validity,
mean latency, generated tokens, and estimated inference cost.

**Pre-registered promotion rule:** the candidate needs at least **three more
full passes out of twelve than v5**, more full passes than base Qwen, no more
than one paired v5 pass-to-fail regression, no lower submission rate, no more
execution-error episodes, and mean latency and inference cost no more than
twice v5's under the same hardware. A failed condition rejects this candidate
for the product. A pass warrants an independent confirmation set before a
strong model-improvement claim; twelve cases alone do not establish a robust
population effect.

If base matches or beats both trained models, revisit data alignment. If all
three miss the same evidence, inspect the corpus and retrieval. If the candidate
fixes supported synthesis but raises tool failures or cost, refine the
trajectory data and repeat on a **new** transfer set rather than tuning against
the sealed twelve. Sonnet may be run as a separately budgeted reference, but
its result does not replace the required base-versus-trained Qwen comparison.

## Reproducible workflow

The source import and question validation use existing commands:

```bash
python scripts/research_vault.py import \
  --vault /path/to/real/Obsidian-vault \
  --collection 'AI Research' \
  --output out/research/vault-transfer-snapshot-v1

python scripts/research_benchmark.py validate \
  --benchmark data/research/vault-transfer-dev-v1.json \
  --snapshot out/research/vault-transfer-snapshot-v1

python scripts/validate_learning_loop_pilot.py \
  --benchmark data/research/vault-transfer-dev-v1.json \
  --snapshot out/research/vault-transfer-snapshot-v1 \
  --review-md out/research/vault-transfer-source-check-dev.md
```

Repeat both validation commands for the locked transfer file. Use the fixed
`scripts/run_eval.py` settings from `scripts/run_learning_loop_code_pilot.sh`
for each arm, then prepare a blind review with
`scripts/review_code_exec_pilot.py prepare`. Do not start a Runpod pod until
the real snapshot, reviewed question files, and development baseline meet the
entry gate above.
