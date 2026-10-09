# Qwen3: evidence gathering and stopping

Prepared September 18, 2026. This is the next data experiment, not a claim that a new
checkpoint works. The model still writes Python in the persistent REPL.

## Why another data pass

The prefix-aligned adapter executed code reliably, but its final checkpoint used all ten
steps on 38/40 questions and took about three times base's latency. The provisional
assistant semantic review favored final on 14 questions and base on seven, with 19 ties;
that difference was not statistically decisive. False refusals increased. The review
sheet hid labels, but the assistant had already seen labeled outputs: it was neither
truly blind nor independent human review.

The 40-question test set has been inspected and used for diagnosis. It is now consumed.
Do not put its questions, answers, passages, or corrections into demonstrations or use
it to choose the next checkpoint.

The CPU audit found problems in the existing demonstrations:

| Check | Train | Validation |
|---|---:|---:|
| Conversations | 189 | 47 |
| Answerable / unanswerable | 110 / 79 | 29 / 18 |
| Immediate submission without investigation | 8 | 3 |
| At least ten actions | 98 | 19 |
| Eleven actions despite the ten-step environment budget | 36 | 3 |
| First action lacks a statically identifiable direct paper read/search | 77 | 18 |

The conversation-level random split also put questions from **20 papers in both splits**.
That weakens a claim of generalization to unseen papers from validation loss. It does
not mean the official QASPER test papers were in training.

The old exporter could append a gold answer after a timed-out episode and accepted
immediate abstentions based on the teacher-only label. Keyword overlap with a gold
answer is not enough to establish support. These findings suggest a mechanism for bad
stopping and refusal habits; they do not prove the training data is the sole cause.

Audit: `out/diagnostics/qasper-sft-behavior-audit-20260918.json`.
The original `data/sft/qasper-v2/` files and archived adapters are preserved.

## Changes in the preparation pipeline

- The teacher starts inside the supplied paper, reads actual passages, and stops once
  every part is supported. It distinguishes an explicit negative result from missing
  information. Both answerability classes use the same workflow instructions.
- Teacher-only answerability hints remain supervision, not student inputs. They never
  justify skipping investigation. Generated comments and answers still need review for
  leaked hints and claims about searches the model did not perform.
- Candidate demonstrations target **2–6 actions including SUBMIT**. The environment still
  allows ten steps: the shorter target is a data-selection choice, not a forced runtime
  cutoff or an instruction to guess. Complex examples can be reconsidered separately.
- Export rejects invalid code, recorded errors, immediate submissions, actions after
  termination, and excessive length. Automatic gold-answer recovery is removed.
- Export first writes a candidate file and review template. Training files require
  recorded replay, source-support and stopping checks, a named reviewer, and a review
  hash matching the actual trajectory. Assistant review is explicitly labeled as such.
- New train/validation splits group by paper. Development and consumed test papers are
  excluded. The student system prompt and per-action loss/prefix implementation remain
  unchanged, so this iteration tests the demonstration changes as a package.

These are conservative candidate checks, not a semantic evaluator. A statically visible
tool call does not prove useful evidence was read; replay and source review address that.

## Frozen development set

`out/research/qasper-code-dev-v2/` contains 40 questions: 20 answerable and 20 unanswerable,
over 37 target papers in a 227-paper corpus. It comes from QASPER's official validation
split, revision `13b496d2a5359329b110e3419628de3cf791843b`, seed `20260918`.

Papers targeted in the earlier validation question set were excluded. Full-corpus checks
found zero overlap with either the training snapshot or the consumed test snapshot.
The balanced mix is a diagnostic design, not the natural frequency of unanswerable
questions in production.

Corpus hash: `927f97bf4b07d4318d216467c00a5cc59f55fe20de554de1be93596d45df481f`.
Question IDs, source/exclusion hashes, and the teacher-pilot selection are tracked in
[`qasper_code_dev_v2_selection.json`](../data/research/qasper_code_dev_v2_selection.json).
The full corpus remains an ignored local artifact and must be copied with the run.

This is **development**, available for error analysis and model selection. Any final
improvement claim will need a separately locked, uninspected confirmation set and the
personal-vault usability evaluation.

## Bounded teacher pilot: decision declared before generation

**Hypothesis:** the revised instructions produce shorter, faithful investigations with
appropriate abstention, without giving up evidence gathering or answer completeness.

**Pilot:** 12 questions from rejected old training demonstrations, six per answerability
class on distinct papers. Selection shuffles sorted question IDs with seed `20260918`.
The full training corpus and ten-step budget stay fixed. Use the configured Sonnet
teacher explicitly, at most 4,096 output tokens/action, two concurrent episodes, and
save the model ID, requested temperature, prompt/hint hashes, corpus and question hashes.
No GPU is needed. The 12 labels are for demonstration generation, not model evaluation.

**Expected signal:** the teacher directly reads the supplied paper, completes the answer
or explains a bounded evidence gap, and stops within six actions.

**Decision rule:** require at least 10/12 fully reviewed passes, including at least 5/6
in each answerability class. A pass needs exact replay, relevant observed evidence,
correct and complete conclusions, justified stopping, and no label leakage. An immediate
abstention or an invented conclusion is a fail. If the pilot misses the gate, inspect
the failures and revise the data process before scaling or renting a training GPU.

Existing data provides 73 structural candidates (72 answerable, only one unanswerable);
all 73 replayed exactly. They remain **unreviewed candidates**, not a training set.
Filtering alone therefore cannot fix the answerability mix.

## Pilot result

The 12-example teacher pilot completed on September 18 using the configured
`claude-sonnet-5` endpoint. These are **teacher demonstration results**, not a new Qwen
evaluation or evidence that the student improved.

| Check | Result |
|---|---:|
| Mean actions, same questions in old demonstrations | 8.17 |
| Mean actions, new demonstrations | 3.92 |
| New action range | 2–5 |
| Immediate submissions | 0/12 |
| Exact replay | 12/12 |
| Episodes with a runtime error | 1/12 |
| Strict review pass / partial / fail | 5 / 6 / 1 |
| Accepted answerable / unanswerable | 2/6 / 3/6 |

The gate **failed**: five accepted examples is below ten, and both class-specific gates
also failed. Do not scale this batch or start Qwen training. The old/new action comparison
is descriptive: the teacher prompt/workflow changed together, provider determinism is
not established, and these questions were selected for earlier failures.

All 12 replays reproduced the recorded behavior, including the runtime error. Reproducible
does not mean correct. The review is an assistant review of the actual observations and
reference/source passages, not independent human evaluation. Its acceptance verdict also
includes stopping and protocol quality; 5/12 is not a standalone answer-accuracy measure.

Named remaining failures:

- `...5a2c0c55...`: the teacher prints an uninitialized variable before recovering.
- `...7889ec45...`, `...827464c7...`, `...a7d72f30...`: repeated observations or extra
  reads for unrequested details; some final answers are otherwise useful.
- `...a0963358...`: useful MTurk details, but it invents an author attribution for an
  unresolved `BIBREF5` and broadens which models use additive attention.
- `...53712f0c...`: finds a real 75%-English filter, then rationalizes the unanswerability
  hint through the plural word “datasets.” Quarantine the ambiguity rather than teach
  that reasoning.
- `...e414d819...`: the source gives a 30–50 minute range, while the expert answer says
  40 minutes. The range alone does not establish the average. Quarantine this reference
  ambiguity; do not append the gold number to the trajectory.

Next: preserve the five accepted examples, quarantine the two ambiguous questions, and
revise/replay the remaining demonstrations with concise answers containing only requested,
observed facts. Do not resolve anonymized citation names from model memory. Review the
resulting data before expanding the teacher batch. Twelve examples are a process pilot,
not a sufficient training-set claim.

The tracked [review record](../data/research/qasper_sft_v3_pilot_review.json) contains every
question ID, decision, note, and trajectory hash. Raw trajectories, exact teacher prompts,
generation identity, logs, and replay reports are in `out/research/qasper-sft-v3-pilot/`.
No `train.jsonl` or `val.jsonl` was exported from this pilot.

Verification for the preparation changes: 33 focused tests passed, Ruff passed, and
`git diff --check` passed. The original 189/47 training files retain their prior hashes.

### September 18 follow-up: source-guided teacher generation

The user requested generating the examples and assessing a better SFT approach. The
next bounded run is recorded in `out/research/qasper-sft-v4/plan.json`: a 12-question
calibration batch, followed by 12 additional questions only if the first batch meets
the same 10/12 overall and 5/6-per-class acceptance gates. The two ambiguous v3 questions
are excluded and replaced deterministically, so this is not a matched quality comparison
with the previous 12.

The new teacher receives human reference answers and source passages for answerable
training questions through an optional `--reference-guidance` flag. It must still find
the support through the actual REPL before answering. The unanswerability hint now
explicitly permits disagreement with a noisy annotation. Neither hint enters student
messages. Reviews must also check that the teacher did not copy unseen reference text
into code or claim it had retrieved a passage it never observed.

**Result: reference-guided generation failed the gate.** One first query included the
hidden numeric answer `10700`; another included the hidden dataset names. Separately,
two episodes used the wrong `search_within()` result key, and several answers contained
details absent from their observations. None of these 12 passed all of the strict
evidence, protocol, and stopping checks as generated. This is not a zero answer-accuracy
claim: some failures were unnecessary actions or reference-dependent searches.

The original plan is preserved. `plan-revision.json` declares a second, bounded
12-question process pilot on the preselected additional training papers, with reference
answers/passages **disabled** and the tool-result schema explained. This was a revision
after failure, not an expansion justified by a passed gate. Answerability-only hints
remained teacher supervision. No more batches were launched after this second pilot.

The second pilot produced four unedited passes. Six more became usable after explicit
assistant edits and fresh execution. Two were excluded: the acoustic-model question's
reference describes a different model in related work, and the spell-prediction question
is ambiguous about what counts as sufficient word association. The first pilot yielded
11 repaired examples; the missing country-table question was excluded.

| Final data artifact | Count |
|---|---:|
| Raw Claude episodes generated across both pilots | 24 |
| Accepted unchanged | 4 |
| Accepted after Codex edits and fresh execution | 17 |
| Excluded | 3 |
| Accepted answerable / insufficient | 11 / 10 |
| Exact replay of accepted episodes | 21 / 21 |
| Mean actions, including submission | 3.19 |
| Action range | 2–5 |
| Paper-disjoint training / validation conversations | 17 / 4 |

The repaired observations were generated by executing the entire revised episode from
scratch, then replayed independently in another fresh environment. They were not edited
to agree with the answer. The source review was performed by the same assistant that
made the edits, with references and identities visible. These are **assistant-reviewed
seed demonstrations**, not independently validated gold data or a measured Qwen gain.
Four validation conversations cannot establish generalization. Both raw-generation
pilots missed the original 10/12 gate; do not scale this prompt unattended.

Artifacts:

- [Readable examples](QASPER_SFT_V4_EXAMPLES.md).
- `data/sft/qasper-v4/train.jsonl` and `val.jsonl`: complete conversations in the current
  student protocol. The existing per-action training loader supplies action-only loss.
- `data/sft/qasper-v4/manifest.json`: hashes, paper-group split, IDs and reserved benchmarks.
- [Review and exact action edits](../data/research/qasper_sft_v4_review.json): original and
  repaired verdicts, parent/output hashes, exclusion reasons, and reviewer disclosure.
- `out/research/qasper-sft-v4/`: original generations, frozen prompts, edited episodes,
  replay reports, and plans. Full artifacts are local ignored files and must be archived
  with any future training run.

Provider responses recorded 300,673 input tokens and 13,051 output tokens across the
24 original episodes. The CPU edits/replays made no additional teacher API calls.
These usage counts are not a billing invoice. The original 189/47 training files retain
their prior hashes. No training or reward code changed, and no GPU was started.

The practical next data step is to review and repair the existing executable candidates
using this process, with occasional new generation for missing behaviors. Do not replace
the existing training corpus wholesale with 17 conversations or simply mix the rejected
old long trajectories back in. An update budget and evaluation must be declared before
any student run.

The workflow now asks for only requested facts, disallows invented names for anonymized
citations, and asks the teacher to read around a truncated window instead of repeating
searches. Generation records provider-returned token usage and resolved model IDs when
available, and persists exact prompts as well as their hashes. Usage is not an invoice:
provider pricing, failed requests, and billing adjustments require separate accounting.

This remains supervised distillation of verified multi-turn demonstrations. The current
per-action, completion-only loss matches the intended action-prediction objective;
[TRL documents completion-only SFT](https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-completion-only).
[Unsloth's dataset guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
also emphasizes reviewing synthetic examples for quality and relevance. Our inference
from the repo's failures is that demonstration quality is the next useful intervention;
neither source establishes that this particular change will improve Qwen.

If clean demonstrations still fail to transfer, a later option is teacher feedback on
states Qwen actually reaches on **training** questions. This addresses the distribution
mismatch discussed in [TRL's on-policy distillation guide](https://huggingface.co/docs/trl/main/en/gkd_trainer).
The documented GKD trainer uses teacher token probabilities; it is not a drop-in method
for this Claude API pipeline. Start with reviewed corrective action targets if needed,
and preserve complete histories and paper-disjoint evaluation. No new algorithm or GPU
training is part of this generation run.

### September 18 expansion: review existing episodes on CPU

The user approved growing the reviewed set. This pass sampled 32 existing episodes
with seed `20260920`: 16 answerable structural candidates and 16 insufficient-label
episodes, including previously rejected ones. All targeted different papers from the
v3/v4 pilots and reserved evaluation snapshots. Selection was fixed before semantic
review. Because the two pools had different structural filters, this is a salvage
experiment, not a representative estimate of QASPER or teacher accuracy.

The declared data gate was at least 24 accepted, including at least ten per class.
The result was **23 accepted: 15 answerable and eight insufficient**. The gate failed.
The accepted work is retained; the nine exclusions were not replaced after review to
make the gate pass. No teacher API calls, GPU rentals, or model training were performed.

Three episodes were kept unchanged. Twenty were repaired, executed from scratch, and
checked against their actual observations; all 23 accepted episodes reproduced exactly
in a separate fresh replay. The same assistant authored the edits and performed source
review, with labels visible. This is not independent or blind review.

The exclusions are substantive:

- A placeholder question (`asdf`) is not a useful research target.
- The citation-generation paper explicitly describes corpus collection despite an
  insufficient label. The IOC paper identifies a prior system's spelling features
  despite an insufficient label. Neither should become a forced refusal.
- Other questions leave material distinctions unresolved: source versus derived
  dataset, a universally optimal strategy versus the paper's proposed method,
  semantic event classes versus noun/verb types, and translation granularity under
  different experimental conditions. Broad requests for performance or other biases
  also do not establish a unique missing fact.

For accepted uncertain answers, the episode reports available evidence and the specific
gap. For example, TPU hardware and runtime do not determine measured energy consumption;
a list of annotation labels does not provide interannotator agreement. For answerable
cases, edits remove repeated searches and echoed facts, retain complete dataset/model
lists, and distinguish qualitative manual inspection from a scored human study.

Combined with the 21 unchanged seed episodes, the dataset now has **44 conversations**
on 44 distinct papers: 26 answerable and 18 insufficient. Mean episode length is three
actions, including submission. The paper-group split has 35 training conversations
(19/16 by answerability) and nine validation conversations (7/2).

The new version repartitions the unused seed pool before any v4 student training:
two former v4 validation papers enter v5 training, and two former v4 training papers
enter v5 validation. Both old exports remain unchanged. The separately frozen
development and consumed-test reservations are unchanged and have no paper overlap
with either new split. Do not compare v4/v5 validation losses as though their splits
were identical. Nine validation conversations, only two insufficient, are a small
diagnostic; use the frozen development benchmark for later behavioral selection.

The actual cached Qwen3-8B tokenizer at revision
`b968826d9c46dd6066d109eabc6255188de91218` passed preprocessing for every action using
the existing training functions and patched chat template. With local Transformers
5.3.0, all 132 targets match the inference generation prefix, the longest sequence is
2,579 tokens (limit 8,192), and the longest supervised target is 177 tokens (inference
budget 1,024). There are 106 training actions and **4,812 supervised training tokens**;
that is still a small amount of supervision, not a demonstrated sufficient dataset.
This CPU check does not replace validation in the pinned GPU training environment.

Files:

- [`data/sft/qasper-v5/README.md`](../data/sft/qasper-v5/README.md): dataset card and file map.
- `data/sft/qasper-v5/train.jsonl`, `val.jsonl`, `manifest.json`, and
  `tokenization-check.json`: export, split identities, and actual preprocessing results.
- [Expansion review](../data/research/qasper_sft_v5_review.json): all 32 decisions, exact
  repairs, source/trajectory hashes, and the failed data gate.
- `out/research/qasper-sft-v5/`: frozen selection plan, original/rebuilt episodes,
  source probes, and replay report. Preserve these local artifacts with future runs.

Verification: 23/23 fresh replays, no reserved-paper overlap, v2/v4 file hashes unchanged,
and 12 focused export/training-preprocessing tests passed. Training and reward code
remain unchanged. Next, review the remaining pool for clearly answerable questions and
well-specified evidence gaps; prioritize reliable insufficient examples. The personal
vault baseline and declared model-comparison budget remain prerequisites for GPU work.

## Later model comparison

Before training, record a bounded update budget based on the accepted data and run the
personal-vault baseline required by `AGENTS.md`. Keep Qwen3-8B, the rank-4 adapter family,
per-action supervision, and the common inference prompt fixed for this data experiment.
Compare base, the archived prefix-aligned final adapter, and the new adapter on the same
development set, with greedy decoding, 1,024 tokens/action, ten steps, seed 42, and common
hardware. Save checkpoint revisions, corpus identity, and all trajectories. No GPU job
has been launched by this preparation pass.

Development promotion gates, declared before seeing runs on these new questions:

- Fully supported answers improve by at least 4/40 over **both** controls.
- False refusals on answerable questions exceed base by at most 1/20.
- At least 38/40 episodes submit nonempty answers; at most 4/40 reach ten steps.
- Mean actions are at most six; latency is at most 1.5 times base on the same GPU.
- Execution-error episodes do not exceed base and are at most 1/40.

These are development selection criteria, not a statistical proof. Review paired outputs
against the source, disclose reviewer limitations, and report uncertainty. A faster model
that guesses or refuses too often fails. MuSiQue-style token-overlap reward remains only
a diagnostic; this change adds no new research reward.

## Commands

Audit the existing data:

```bash
python scripts/audit_qasper_sft_data.py \
  --data data/sft/qasper-v2/train.jsonl --data data/sft/qasper-v2/val.jsonl \
  --benchmark out/research/qasper-train-teacher-v2/benchmark.json \
  --output out/diagnostics/qasper-sft-behavior-audit-20260918.json
```

Recreate the development snapshot into a new directory (the dataset revision is pinned
by the converter; `--local-arrow` can instead use the cached official validation file):

```bash
python scripts/setup_qasper_code_exec.py \
  --split validation --num-questions 40 --min-insufficient 20 --seed 20260918 \
  --exclude-benchmark out/research/qasper-train-teacher-v2/benchmark.json \
  --exclude-benchmark out/research/qasper-test-abstention-v1/benchmark.json \
  --exclude-benchmark out/research/qasper-validation-eval-v1/questions.json \
  --output out/research/qasper-code-dev-v2-reproduced
```

Generate only the frozen teacher pilot, supplying the configured model explicitly:

```bash
python scripts/generate_qasper_teacher_batch.py \
  --benchmark out/research/qasper-sft-v3-pilot/benchmark.json \
  --corpus out/research/qasper-train-teacher-v2/corpus \
  --output out/research/qasper-sft-v3-pilot/trajectories.jsonl \
  --model claude-sonnet-5 --workers 2 --max-steps 10
```

Prepare review candidates (no training files without `--reviews`):

```bash
python scripts/export_qasper_sft_data.py \
  --trajectories out/research/qasper-sft-v3-pilot/trajectories.jsonl \
  --benchmark out/research/qasper-sft-v3-pilot/benchmark.json \
  --exclude-benchmark out/research/qasper-code-dev-v2/benchmark.json \
  --exclude-benchmark out/research/qasper-test-abstention-v1/benchmark.json \
  --exclude-benchmark out/research/qasper-validation-eval-v1/questions.json \
  --out-dir out/research/qasper-sft-v3-pilot/review
```

After replay and semantic review, repeat into a new output directory with
`--reviews PATH_TO_COMPLETED_REVIEW`. Only checked passes are exported; output manifests
record paper/question IDs, file hashes, and class counts for each split.
