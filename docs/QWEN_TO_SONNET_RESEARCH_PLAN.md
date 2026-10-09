# Qwen-to-Sonnet research plan for the learning-loop agent

Research and transcript audit: 2026-09-26. This is a proposed experiment order,
not a training result or permission to train on the evaluated questions.

## What the current comparison actually says

The [ten-arm sweep](LEARNING_LOOP_MODEL_SWEEP.md) used eight Codex-authored,
already-inspected development questions over six historical papers. Blinded
Codex self-review gave Sonnet 5 three passes, three partials, two fails; Qwen3
v5 SFT zero passes, five partials, three fails; and base Qwen3 zero passes,
four partials, four fails. There is no independent human review or held-out
personal-vault score. Do not train on these eight questions, reference answers,
grader notes, source anchors, or corrected responses.

V5 submitted on 8/8 questions with zero detected execution-error steps. That is
evidence that it learned the *code-execution/submission protocol*, not that it
learned source-grounded research synthesis. Its remaining failures are mostly
after retrieval: an invented HotpotQA result on ReAct, omitting original RAG in
a RAG/Self-RAG comparison, missing CRAG's external-web fallback, citing the
Search-R1 mechanism without an ablation span, conflating a valid citation with
semantic support, failing to explain historical coverage limits, and giving an
arbitrary rather than quality/cost-based experiment stop rule. These are
plausible targets for better evidence and comparison demonstrations. They do
not show that more optimizer steps on the existing QASPER recipe would help.

### Task-alignment audit after the inference diagnostic

The checkpoint-50 training manifest identifies `data/sft/qasper-v5/train.jsonl`
(35 conversations, 106 assistant targets, **4,812 supervised tokens**) and a
nine-conversation validation file. Across all 44 conversations, every initial
question names one known QASPER paper, none requests a multi-paper comparison,
and none of the 44 final `SUBMIT:` actions contains an `EVIDENCE:` span. The
learning-loop evaluation instead gives several candidate papers and asks for
comparisons, semantic citation support, historical limits, and local decisions
with exact spans. V5's 8/8 submissions show that the test-time prompt can teach
the evidence *format*; its 0/8 full passes show that formatting did not teach
the missing synthesis. This is a data/task mismatch, not proof that 4,812
supervised tokens are universally insufficient.

The registered inference diagnostic has now rejected both the evidence
checklist (mean 0.500 versus greedy v5's 0.625) and three-seed sampling (mean
0.750, below its improvement threshold, with execution errors on two seeds).
Stop iterating on those eight consumed questions. The next model intervention
is one **task-aligned full-trajectory SFT continuation of v5**, conditional on
the already required reviewed personal-vault baseline. Use different papers
and questions; never export the eight evaluated questions or answers as student
training examples.

There is an existing source pool for data construction:
`out/research/ai-agents-development-v1-20260912/` contains 20 agent/research
papers disjoint from the six-paper learning eval, and
`data/research/development_questions_v1.json` contains 14 usable unreviewed
development prompts. They are **task seeds, not gold labels**. Source-check and
rewrite them for the current Python `search()`/`read()`/`extract()` loop; the
old `data/research/demonstrations_v1.json` uses the paused JSON-action protocol
and cannot be copied into SFT. The QASPER teacher generator and exporter also
hard-code a known single paper, so this batch needs a focused multi-paper
generator/exporter using the current evidence prompt and real REPL observations.

Start with a bounded batch that deliberately covers: two-source comparisons;
claim versus exact supporting span (including misleading real citations and
unsupported numerical claims); limits of a dated corpus; and a project
decision with a quality/cost stop rule. Every accepted trajectory should
execute successfully, inspect the cited text, submit concise `CITATIONS` and
`EVIDENCE`, cover every requested part, and pass source-support review. Keep
paper IDs and source hashes disjoint from the final transfer questions. Train
with the existing per-action, assistant-only loss and a continued v5 adapter;
retain a small amount of current-protocol single-paper rehearsal to protect
tool/submission reliability. The [frozen transfer test](TARGETED_QWEN_VAULT_TRANSFER_TEST.md)
is the before/after scoreboard; do not design another eval for this pilot.

Sonnet is an imperfect ceiling in this run. On questions 6 and 8 it drafted
substantive answers but repeatedly omitted the required `CITATIONS` and
`EVIDENCE` suffix. The one-line action was capped at 1,024 tokens; the
evidence verifier rejected and escalated both. Sonnet also failed to choose a
span supporting its numerical ReAct claim on question 1. Its 3/8 pass count
therefore mixes genuine answer quality with protocol/budget problems. A later
comparison should separately report semantic quality of a valid proposal,
mechanical acceptance, and end-to-end supported answers. Any protocol change
must be applied to all arms and evaluated as a new experiment.

## What external research supports, and what it does not

- [Qwen3's official model card](https://huggingface.co/Qwen/Qwen3-8B) describes
  distinct thinking and non-thinking modes. It recommends sampled decoding
  for non-thinking mode and warns against greedy decoding in thinking mode.
  The current policy hard-disables thinking and uses greedy decoding. A small
  decoding/thinking-budget ablation is justified, but the card does not prove
  it will fix this task. Prior project runs showed that uncapped thinking can
  consume the action budget before code appears.
- [RAFT](https://arxiv.org/abs/2403.10131) improved open-book domain QA by
  training on relevant passages mixed with distractors and quoting the useful
  text. [FRONT](https://arxiv.org/abs/2408.04568) trained fine-grained grounded
  citations; [ReClaim](https://arxiv.org/abs/2407.01796) alternated references
  and claims. Their settings are not our Python tool loop, but they directly
  motivate claim-to-quote supervision and distractor-rich examples rather
  than more paper-fact memorization.
- [AgentTuning](https://arxiv.org/abs/2310.12823) trained on high-quality
  interaction trajectories and mixed them with general instruction examples.
  This supports testing full tool trajectories instead of answer-only SFT;
  transfer to this vault must still be measured.
- [Search-R1](https://arxiv.org/abs/2503.09516) showed multi-turn search RL
  can improve *question-answering* with outcome rewards and retrieved-token
  masking. Our active MuSiQue reward in `src/env/reward.py` is answer overlap
  plus citation IDs, not a semantic-support judge. The sweep accepted answers
  with unsupported claims. Search-R1 therefore does **not** justify running
  GRPO on the current reward to improve research synthesis.
- [DPO](https://arxiv.org/abs/2305.18290) uses chosen/rejected response pairs
  without a separately trained reward model. A tool-specific study,
  [ToolPreference](https://arxiv.org/abs/2406.07115), uses step-wise
  preferences from successful and failed tool trajectories. These are
  candidates *after* we have independently reviewed pairs that truly differ
  in source support. Neither establishes that generic preference labels will
  solve our multi-paper task.
- [RAGBench](https://arxiv.org/abs/2407.11005) separates actionable retrieval
  and generation-quality dimensions. For this project, provenance validity,
  claim support, coverage of each requested part, calibrated uncertainty,
  and cost should be recorded separately. Exact spans alone are not entailment.

Qwen now offers a newer [post-trained Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).
Its published general-agent scores make it a useful *baseline candidate*,
not a proven winner on this task. It has a different architecture and chat
template: the existing Qwen3 adapter cannot be attached to it. Run a small
format/tool smoke test before any full comparison. Do not start from
`Qwen3.5-9B-Base` or raw pretrained weights for this narrow product skill;
that would discard substantial post-training and add a much larger task.

## Ranked intervention order

| Priority | Option | Why / exact test | Decision |
| --- | --- | --- | --- |
| 1 | Evaluation and inference diagnostics | Completed for the checklist and three-seed non-thinking sampling arms on the consumed eight cases; neither met its registered rule. Bounded thinking and a newer-model baseline remain optional later diagnostics, not prerequisites to task-aligned SFT. | Stop tuning against the consumed eight; preserve their transcripts as failure examples only. |
| 2 | Targeted full-trajectory SFT / teacher distillation | Once the personal-vault baseline is reviewed, train on *different* papers/notes with source-checked Python search/read/extract trajectories, multi-paper contrasts, distractors, numeric-result checks, unsupported-citation negatives, temporal limits, and local-experiment decisions. Retain concise actions and exact quote-to-claim links. | Start with a small data-quality and behavioral pilot. Scale only if a separate development split shows paired supported-answer gains over the same starting checkpoint and keeps protocol reliability. |
| 3 | Preference optimization | If SFT fixes the protocol but supported-vs-unsupported answer choice remains weak, collect human-reviewed chosen/rejected answers or step pairs on the same evidence. | Use only when pair labels are consistent and paper-disjoint development evaluation improves. |
| 4 | GRPO / other online RL | Consider only after reward agreement with human support judgments is demonstrated and rollout groups contain informative, non-uniform rewards. Mask retrieved tokens. | Stop if the reward can be increased by lexical overlap or valid-but-irrelevant citations, or if quality/latency regresses. |
| 5 | Continued pretraining or training from raw base | Low priority: the task is to use changing external evidence and obey a tool protocol, not to internalize a stable corpus. | Revisit only if a measured domain-language comprehension bottleneck remains after retrieval and task-aligned SFT. |

Long free-form chain-of-thought SFT is not a separate first step. Qwen3 is
already post-trained for reasoning; the current runtime expects executable
actions in non-thinking mode. Before adding reasoning text, test a *bounded*
plan such as requested subquestions and missing evidence, then train only on
that same format if it improves complete supported answers. Do not copy raw
Sonnet traces: the two escalations show that fluent but overlong trajectories
can be invalid in this protocol.

## Proposed next experiment and promotion rule

The repository currently contains only a two-note Obsidian demo snapshot, so
the required personal-vault baseline is still missing. Freeze a richer vault
snapshot with named questions and reviewed expected source passages. Keep a
paper/note-disjoint development set and an untouched held-out set; include
single-note, cross-note, conflicting/insufficient-evidence, recency, and
proposed-experiment questions. Have a human confirm references and review
blinded outputs. Compare one-pass RAG, base Qwen3, v5, a smoke-tested newer
post-trained Qwen if available, and Sonnet under documented token budgets.

**Hypothesis:** if v5's bottleneck is source-grounded synthesis rather than
tool mechanics, a concise claim/evidence scaffold or source-reviewed
multi-document trajectory SFT will turn partial answers into fully supported
answers without losing its submission reliability.

**Expected signal:** paired increases in complete supported answers and
question-part coverage, especially on cross-source and epistemic items;
unchanged or lower unsupported-claim rate; no unacceptable execution,
latency, or cost regression. Record checkpoint, question IDs and split,
corpus revision, decoding, reward version, hardware, and seed.

**Decision rule:** the checklist and sampling arms are closed. Prepare the
task-aligned SFT batch now, and train only when the reviewed vault baseline
confirms a recurring learnable failure. Promote a trained model only if it beats its own
base on the untouched held-out code-execution tasks, has acceptable
execution/latency/cost, and approaches Sonnet on the same supported-answer
rubric. A second training run that merely raises automatic overlap or 8-case
development score fails this rule. The weekly digest and MCP usability gate
remains separate and must also pass.

Starting point for SFT is a measured choice, not a doctrine: v5 is currently
the most reliable existing Qwen3 policy, so a continued adapter with rehearsal
examples is the first small pilot. If a newer post-trained Qwen wins the
zero-training baseline by a meaningful margin, train a fresh adapter on that
model instead. Preserve base and v5 as controls; never assume an adapter from
one architecture transfers to another.
