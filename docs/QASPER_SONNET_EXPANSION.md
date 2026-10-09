# Sonnet code-execution demonstrations — September 19, 2026

The two-example overfit diagnostic established that the current Qwen3-8B/rank-4
adapter can learn and execute clean demonstrations. It did not establish
generalization. The next intervention is more diverse, reviewed demonstrations
under the same exact-span protocol, followed by SFT and a held-out comparison.

## Generation hypothesis and decision rule (before API calls)

Sonnet 5, given training-only QASPER annotations, can produce short, independently
executed searches followed by supported answers. Teacher-only references are never
student inputs. A reference answer or a valid span does not establish support.

Freeze 200 questions from 200 new official-training papers: 105 extractive,
35 abstractive, 20 boolean, and 40 unanswerable. Exclude all existing v5 reviewed
papers, historical SFT validation papers, the two fitted diagnostic papers, and
the earlier training pool's development/test/quarantine exclusions. Selection seed
is 20260919; exact IDs, source hashes, and exclusions are saved by
`scripts/prepare_qasper_teacher_expansion.py`.

Inspect the first 12 fixed questions (six extractive, two abstractive, two boolean,
two unanswerable). Continue to the remaining 188 only if at least 10/12 pass replay,
answer completeness, source support, and sensible stopping, with passing examples
of both answerability classes. Accept only real 2–6-action demonstrations without
execution errors. Reject teacher-label leakage or queries containing answer values
not yet visible in the question/tool history. Preserve failures; do not append gold
answers after a timeout. If the gate fails, diagnose before another paid batch.

Every candidate needs source review before SFT export. An assistant review with
annotations visible is not independent or blinded review. QASPER overlap reward
is logged as a diagnostic proxy, separate from the legacy environment reward.

## Fixed setup and spending

- Teacher request and returned identity: `claude-sonnet-5`, no model fallback.
- System prompt: shared `QASPER_SYSTEM_PROMPT` in `src/policies/code_execution.py`.
  SHA-256: `1d028e1cd23ae7ab1022d6a51accd9005f264e3794abe37292e8c137b8ec46b7`.
  Verified identical to the completed GPU diagnostic.
- Same initial observation, actual unmodified tool outputs, and raw assistant
  actions retained for student export. No repeated environment preamble.
- Max 10 actions, 4,096 output tokens per teacher response, four concurrent
  episodes, local persistent REPL, no vector-index download.
- Requested temperature 0; remove it if Sonnet rejects the parameter. No claim of
  seeded/deterministic teacher generation; Qwen training/eval seeds recorded later.
- Shared persistent **$10 teacher cap**, including the pilot and resumed requests.
  User reported adding credits but has not specified a higher cap. Overall next
  phase ceiling remains $50. No GPU is started by this generation script.
- Direct API [published prices](https://platform.claude.com/docs/en/about-claude/pricing),
  checked September 19: $2/M input and $10/M output. No caching/server tools enabled.
  Free token-count preflight reserves 120% of estimated input plus 1,024 tokens,
  and the entire requested output allowance before a paid request. Reconcile
  actual usage after each response. These are usage-derived estimates, not account
  balance queries or a provider-enforced account limit.
- Disable SDK retries. Failed episodes still count toward spend. Unknown charges
  retain their reservation and stop generation; they cannot silently reset on
  resume. A process lock prevents simultaneous use of the same ledger. Preserve
  completed/in-flight episode records when the batch stops.

The 44 earlier reviewed examples are preserved in their older document-citation
format. They need replay/review of precise spans before mixing with this batch;
merely changing their system prompt would create incorrect demonstrations.

## Reproduction

```bash
python scripts/prepare_qasper_teacher_expansion.py
python scripts/generate_qasper_teacher_batch.py \
  --benchmark out/research/qasper-sonnet-expansion-20260919/benchmark.json \
  --corpus out/research/qasper-rl-v1/train-v2/corpus \
  --output out/research/qasper-sonnet-expansion-20260919/trajectories.jsonl \
  --model claude-sonnet-5 --reference-guidance --budget-usd 10 \
  --workers 4 --limit 12
```

After recording the pilot review and gate decision, omit `--limit 12` to resume
against the same manifest and budget ledger. Existing records are skipped.
Do not treat the generated count as the accepted training count, or a successful
teacher batch as Qwen improvement. Training and held-out evaluation follow review.

## First pilot and revised check

The reference-guided pilot completed 12 episodes for an estimated **$0.379882**.
All generation responses reported Sonnet 5. The gate failed: eleven episodes have
identified disqualifying problems; the remaining MoE abstention is not accepted
without a broader source review. Mechanical validity was 12/12, demonstrating why
it cannot be used as the semantic gate.

Examples: the DepecheMood answer cited a slice ending before the lexicon's name;
the architecture answer cited an unrelated passage; queries for Honk/DeepSpeech
and LDC/NIST used private reference values before retrieving any source. Two
episodes contained prose that failed Python execution. Several exceeded six turns.
No first-pilot episode has been exported for training. Details and trajectory
hashes are in `out/research/qasper-sonnet-expansion-20260919/pilot-review.json`.

Revised hypothesis: removing private reference answers/passages from generation
will eliminate this source of search shortcuts. Explicit teacher-only instructions
to compute offsets in Python, inspect the exact cited slice, and emit short answers
should reduce span and format errors. Keep the student prompt, questions, corpus,
and Sonnet model unchanged. Unanswerability hints remain teacher-only. Apply the
same 10/12 gate to the same training-only questions; this is prompt development,
not an independent quality estimate. If it fails again, do not scale the batch.

Save the revised attempt separately as `unhinted-trajectories.jsonl`, omit
`--reference-guidance`, and explicitly share the first attempt's ledger with
`--budget-ledger out/research/qasper-sonnet-expansion-20260919/trajectories.budget.json`.
The **combined cap remains $10**, not $10 per attempt.

## Revised result and current state

The second 12-episode attempt cost **$0.513176**, bringing both attempts to
**$0.893058**. Across both attempts, all **121 generation responses** reported
`claude-sonnet-5`. The persistent ledger has no unsettled requests. No GPU was
started and Qwen was not retrained.

All 12 revised trajectories replayed with identical recorded observations and
completion flags. Review found **two passes, seven failures, and three pending**.
The seven known failures alone prevent the required 10/12 result, so bulk
generation was not started. The two passes are candidates, not a new train/val
split and not a claim that the dataset is sufficient.

The AMR pretrained-embedding question now gives the exact `Yes` answer with a
supporting source span. The NMT dataset question uses a real search, computes a
passage offset, inspects the complete dataset paragraph, and answers all parts.
Remaining failures include invented or incomplete spans, unrequested unsupported
details, execution errors, and five episodes exceeding the six-action data gate.
This is a teacher-data failure, not a new measurement of Qwen.

The next bounded intervention should validate that submitted spans were actually
inspected, and repair short otherwise-correct examples through real tool replay.
More credits alone do not fix these errors. Do not spend on the remaining 188
questions or another SFT run until the demonstration quality gate passes.

Artifacts:

- [Run summary](../out/research/qasper-sonnet-expansion-20260919/summary.json)
- [All revised transcripts](../out/research/qasper-sonnet-expansion-20260919/transcripts/index.md)
- [Revised review decisions](../out/research/qasper-sonnet-expansion-20260919/unhinted-review.json)
- [Two reviewed conversations](../out/research/qasper-sonnet-expansion-20260919/reviewed-conversations.jsonl)
- [Persistent spend ledger](../out/research/qasper-sonnet-expansion-20260919/trajectories.budget.json)

The implementation checks passed 36 targeted offline tests before generation,
including an actual local REPL episode with a fake API response. That test caught
the legacy Claude cleaner discarding `EVIDENCE`; the Sonnet teacher now uses the
same action parser as Qwen. After changing teacher instructions, all 29 affected
teacher/budget/data-review tests passed again. These are implementation checks,
not model-quality scores.
