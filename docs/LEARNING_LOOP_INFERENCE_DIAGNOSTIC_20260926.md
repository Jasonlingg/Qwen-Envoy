# Learning-loop inference diagnostic — registered 2026-09-26

This is a no-training diagnosis on the eight already-consumed development
questions from `data/research/learning_loop_code_pilot_v1.json`. It cannot
establish product performance or a held-out model improvement. Exclude these
questions, answers, and reviewed outputs from all future training.

## Frozen control

The control is `v5_3` from the [completed model sweep](LEARNING_LOOP_MODEL_SWEEP.md):
Qwen3-8B plus `jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5` checkpoint 50,
greedy non-thinking decoding, 1,024 new tokens per action, eight named questions,
six-paper corpus, label-free candidate IDs, 15 steps, same evidence verifier and
four recovery turns. Codex self-review scored 0 pass / 5 partial / 3 fail,
mean 0.625 on a 0–2 scale, with 8/8 submissions and no detected execution-error
steps. Exact model and corpus hashes are in the old run manifest.

## Arm A — evidence checklist only

Append `data/prompts/learning_loop_claim_check_v1.txt` to the old shared prompt
suffix. Keep checkpoint, greedy decoding, seed 42, questions, corpus, tools,
caps, and verifier unchanged. The added text asks for each requested part, its
own supporting passage, the benchmark associated with any numerical result,
and a complete but short submission.

**Hypothesis:** v5 already knows the tool and submission protocol; a claim and
source checklist will reduce unsupported claims and incomplete comparisons.
**Expected signal:** at least one full pass, mean review score >=0.875, no more
than one paired loss to the control, at least 7/8 submitted answers, no new
execution-error episodes, and mean latency <=2x the 15.6-second control.
**Decision:** keep this as a candidate only if all conditions hold. Any gain is
diagnostic because the set has been inspected and the reviewer is Codex.

## Arm B — sampled non-thinking decoding only

Use the original prompt and identical harness with Qwen3 non-thinking sampling
at temperature 0.7, top-p 0.8, seeds 42, 43, and 44. These are the Qwen3 model
card's suggested non-thinking parameters. No prompt or checkpoint changes.

**Hypothesis:** greedy decoding prematurely selects a plausible short answer
and may repeat a bad action; sampling might explore better evidence/answers.
**Expected signal:** mean blind-review score across the three runs exceeds the
greedy control by >=0.25 on the 0–2 scale, with no meaningful submission,
execution, or latency regression. Report per-seed spread and paired outcomes.
**Decision:** adopt sampling for a fresh evaluation only if that signal holds;
one favorable seed is insufficient. These development cases are not a final
benchmark.

## Review and cost

Freeze transcripts and manifests before scoring. Review anonymized answers
with materialized source spans using the same pass/partial/fail rubric; disclose
that Codex authored and has seen the questions and prior failures. Report
mechanical acceptance, semantic score, execution errors, latency, hardware,
seed, prompt hash, and checkpoint hash separately. Do not train or modify the
active MuSiQue reward. Cap incremental Runpod GPU compute at $3 for this test,
copy and checksum results locally, then stop the pod. A meaningful fresh
personal-vault baseline remains blocked by the current two-note demo snapshot.

## Result — 2026-09-26

The four new transcripts and manifests are in
`out/research/learning-loop-inference-diagnostic-20260926/`. The transcript
and model-identity SHA-256 checksums matched the remote originals before
shutdown. The anonymous review,
source quotes, row-level verdicts and notes are in `blind-review/`; Codex scored
all 40 rows before opening the system key. This is **Codex self-review of
Codex-authored, previously inspected development prompts**, not a human or
held-out estimate. The rereviewed greedy control received the same 0.625 mean
score as the prior review.

| Arm | Pass / partial / fail | Mean score (0–2) | Submitted | Error-step rate | Mean time/question |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy control `v5_3` | 0 / 5 / 3 | 0.625 | 8/8 | 0% | 15.58 s |
| Checklist, greedy 42 | 0 / 4 / 4 | 0.500 | 8/8 | 0% | 16.29 s |
| Sampled 42 | 0 / 7 / 1 | 0.875 | 8/8 | 11.76% | 15.35 s |
| Sampled 43 | 1 / 4 / 3 | 0.750 | 8/8 | 0% | 13.05 s |
| Sampled 44 | 0 / 5 / 3 | 0.625 | 7/8 | 31.43% | 33.99 s |

The checklist lost one paired question to the control and won none. It had no
execution regression, but missed the registered quality threshold and produced
no full pass. **Reject Arm A.**

The sampled mean was 0.750, only +0.125 over greedy, below the registered
+0.25 threshold. Seeds 42 and 44 each had an execution-error episode; seed 44
also omitted one answer and exceeded twice the control latency. The one full
pass across 24 sampled answers does not rescue the three-seed result. **Reject
Arm B.** Do not switch the default decoding based on the favorable seed.

The dominant answer defects remained incomplete requested parts, unsupported
numerical claims, citations to real passages that did not support the claim,
and failure to propose a fair local comparison. The checklist did not correct
them. In sampled seed 42, the generated code shadowed the `passage()` tool with
a dictionary and raised `TypeError`; seed 44 repeated a `KeyError: 'start'`
after assuming the wrong search-result shape, then failed to submit the last
answer. These observations motivate a later task-specific data or harness
diagnosis, but no training decision follows from this consumed set.

The frozen question and corpus hashes, checkpoint, tool settings, verifier,
and step cap matched the control across arms; `validation.json` records the
field-level check. Qwen3-8B was pinned to revision
`b968826d9c46dd6066d109eabc6255188de91218`; the v5 checkpoint-50 adapter
was pinned to revision `27b912a863ff914ad45baa03624d8911dc1e17fb`
and weight SHA-256
`7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3`.
The checklist prompt SHA-256 was
`ecf2672415be1db99965efeaead47ca944d344166270da72116ca23e71b3e046`;
the original suffix SHA-256 was
`dae40f66836d1b28919b9a066fc1b8b5b139fc9a508ae06f259036241ca3942e`.
Hardware was an NVIDIA RTX PRO 4500 Blackwell Server Edition with torch
2.8.0+cu128; `pip-freeze.txt` records the environment. The pod ran from
18:35:23 to 19:00:26 UTC, about $0.30 of compute at $0.72/hour, and was
verified stopped (`desiredStatus: EXITED`, `runtimeStatus: stopped`).

The next meaningful experiment needs a fresh, larger personal-vault snapshot,
named held-out questions, reviewed source-grounded outputs, and the same
base-versus-trained Qwen comparison. Neither a repeated run on these eight
questions nor another training job would establish either product gate. The
bounded before/after protocol is now in
[the targeted Qwen vault transfer test](TARGETED_QWEN_VAULT_TRANSFER_TEST.md).
