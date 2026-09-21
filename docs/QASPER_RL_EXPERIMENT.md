# Qwen3 QASPER RL experiment — September 18, 2026

The user authorized a bounded RL experiment with $70 remaining compute credit.
This is an annotation-supervised code-execution experiment, not a new teacher-data
generation job. The 44 reviewed SFT examples remain preserved and unused by this run.

## Hypothesis and decisions, declared before GPU results

The archived prefix-aligned Qwen3-8B SFT adapter can already execute document tools.
Sampling its investigations will produce enough variation in answer/evidence quality
for a frozen-reference GRPO update to improve supported answers on new papers.

First run a five-question base/SFT smoke under the new, common evidence prompt, then
sample four attempts per question on a training-only diagnostic subset. Check actual
PEFT reference restoration, nonzero finite gradients, initial importance ratios,
memory, elapsed time, completion, evidence coverage, and uniform-reward groups.
Do not expand if the protocol yields mostly invalid submissions or if five consecutive
training batches have no relative reward signal. Save the failures instead of hiding them.

The initial diagnostic has a $5 maximum allocation; start with a four-hour pod stop
timer (about $1.96 at the quoted $0.49/hour A40 rate, plus storage). Timers must stop
the pod through RunPod, not merely kill the training process. The total experiment
allocation is at most $50, keeping $20 for later evaluation, storage, and recovery.
This allocation is a ceiling, not a target to spend. No large run is automatic.

Operational update: the newly allocated A40 pod lacks the documented pod-scoped
runpodctl credential. A timer process was created, but its API preflight failed;
it is not a working billing cap. Until that is resolved, only supervised checks
are permitted, with the pod explicitly stopped through the MCP before this session
ends. Do not describe the timer as armed or leave a background training job billing.

For promotion, reuse the existing 40-question development IDs and compare base,
unchanged SFT, and RL under identical prompt, context, action budget, hardware and
greedy decoding. Existing published scores use a different prompt and are not controls.
Retain the previously declared development gate: at least four additional supported
answers over both controls, false refusals at most base+1, at least 38 submissions,
at most four ten-step episodes, mean at most six actions, at most one execution-error
episode, and latency at most 1.5x base. These are selection criteria, not significance
claims. Review paired answers against sources and report uncertainty. Reserve fresh
paper-disjoint data for confirmation; the consumed test set is not fresh confirmation.

## Data and reward

`scripts/prepare_qasper_rl.py` constructs 500 official training questions (400 answerable,
100 unanswerable), preserving all reference answers and mapped evidence. It excludes
reserved evaluation papers, unused v5 validation papers, known rejected review cases,
annotator disagreements, table/figure dependencies, ambiguous mappings, and missing
evidence for training. It records source revision, Arrow hash, exclusions and corpus hash.

`qasper-answer-evidence-v1` is separate from MuSiQue's unchanged `outcome-v1`:

- Only an explicit, nonempty submission after a successful document-tool action scores.
- Cited spans must be in bounds and agree with the cited document IDs.
- Answerable reward is answer F1 times (0.5 + 0.5 * evidence character-coverage F1),
  maximizing over complete individual annotations. Evidence must be present. Boolean
  answers require normalized Yes/No equality. The answer scorer uses multiset token F1.
- Unanswerable reward is one for exactly `Unanswerable` with empty citations/evidence,
  after investigation; false refusals, invalid submissions and timeouts score zero.
- Character coverage discourages one-character and whole-paper citation shortcuts.
  It is not the official paragraph metric or a semantic entailment verifier.
- No reward is given for printing words, number of tool calls, or merely valid code.

Gold annotations remain in the external grader. The model receives the question and
real tool observations, never the references or answerability label. The prompt adds
the existing passage tool and explicit EVIDENCE offsets; this is why controls are rerun.

## Implementation and limits

`scripts/train_qasper_grpo.py` uses the tested per-token PPO update, Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, the archived rank-4 SFT adapter, NF4,
8,192 context tokens, 1,024 action tokens, ten actions, temperature 1 for rollouts,
one update epoch per batch, learning rate 1e-6, beta 0.001 against a frozen copy of
the starting SFT adapter, and seed 42. Only generated action tokens carry policy loss.
The model is in non-thinking mode. Gradient checkpointing keeps training mode enabled
while dropout is disabled; Qwen's attention dropout must be zero. Action-only vocabulary
logits reduce peak memory. GPU smoke must validate these paths on actual PEFT weights.

Each run saves source and data hashes, package versions, checkpoint hash, hardware,
raw actions, cleaned executed actions, observations, scores, update diagnostics and
intermediate adapters. Wall-time limits stop scheduling new episodes; a separate
pod-side watchdog is required to stop billing even if generation or training hangs.

The personal-vault snapshot currently contains only Obsidian's Welcome note. It cannot
support a meaningful personal research baseline. This bounded QASPER diagnostic does
not establish product usefulness, paper discovery, or cross-paper synthesis. A populated
vault and reviewed questions remain necessary before a product-quality claim or a large
follow-up training commitment.

Research basis: [Search-R1](https://arxiv.org/html/2503.09516v3),
[ReSearch](https://arxiv.org/html/2503.19470v3),
[GRACE](https://arxiv.org/html/2601.04525v1), and
[Search-R1 training analysis](https://arxiv.org/html/2602.19526v1).

## Preparation result and launch status

- `out/research/qasper-rl-v1/train-v2/benchmark.json`: 500 questions on 390 target
  papers, with 400 answerable and 100 unanswerable questions. The earlier `train/`
  preparation is superseded because its review-decision filter missed fail/partial
  labels; do not use it. `train-v2` excludes those decisions.
- `out/research/qasper-rl-v1/dev/benchmark.json`: all 40 frozen development question
  IDs, 20/20 by answerability, with richer annotations. It includes the official
  validation corpus; this newly serialized corpus is recorded by hash and must be
  identical across the new base/SFT/RL comparisons. Train and dev corpora have no
  paper overlap. Two development questions have a reference without evidence;
  keep these questions and their other annotated references in the evaluation.
- Mechanical oracle-submission checks pass for all 500 training questions. This
  verifies schema/reward consistency, not label correctness or semantic support.
- 29 focused tests pass, including legacy GRPO regressions, action-logit gradient
  equivalence, dropout-disabled checkpointing mode, reward shortcuts, and existing
  SFT export/tokenization checks. `git diff --check` passes.
- RunPod pod `b3a5v7kh1q624a` (`envoy-qasper-rl-diagnostic-20260918`, A40 in CA-MTL-1,
  quoted $0.49/hour) was created for a supervised diagnostic and then **stopped**.
  No model inference or training ran. Automatic approval review rejected the SCP
  transfer twice, requiring explicit approval of the private adapter payload and
  destination. No archive was transferred. Local preparation is ready; GPU smoke
  and the training decision remain outstanding.
- Requested transfer: the existing private Qwen3-8B rank-4 adapter, relevant project
  code, and public QASPER data to that user-account RunPod pod. No `.env`, SSH keys,
  Git history, or personal-vault content belongs in the transfer. The adapter's
  provenance is HF revision `012d3c7aa13956702846892a1176ed2a170e923e` of
  `jasonlingg/qwen-envoy-qwen3-8b-qasper-sft`, previously evaluated on the user's A40.

### Restart attempt after transfer approval

The user approved starting the diagnostic. The original pod could not restart:
RunPod returned "There are not enough free GPUs on the host machine to start this
pod." Existing account pod `jtzpuo7bya1jxb` restarted at the same $0.49/hour A40
rate at 2026-09-19 00:17 UTC. Its container filesystem is fresh, so dependencies
must be installed again. Automatic approval review rejected the archive transfer
to this replacement destination and requested separate explicit authorization.
That authorization is pending; no private archive has been uploaded.

The five-question smoke selection is fixed in
`out/research/qasper-rl-v1/dev-smoke/benchmark.json`: the first three sufficient
and first two insufficient questions in frozen development order. Selection
preceded any new model outputs. Public dependency setup can proceed while the
transfer is pending; inference and training cannot.

The replacement's public dependency imports passed on real A40 hardware:
PyTorch 2.8.0+cu128, Transformers 4.57.6, PEFT 0.17.1, and bitsandbytes 0.47.0.
Its runpodctl authentication is also absent, so unattended billing shutdown
remains unavailable. The replacement was explicitly stopped through MCP while
transfer approval remains pending; status is EXITED. No model was loaded and
no inference or optimizer update ran. Stopping this container does not preserve
its root filesystem; the local input archive and original HF adapter remain safe.

### Approved launch, September 18 local / September 19 UTC

The user explicitly approved the replacement destination. Pod `jtzpuo7bya1jxb`
restarted at 01:23:32 UTC, and the 27 MB archive uploaded successfully. Its remote
SHA-256 matches `5897d08b6689ccac05cee5ecbcbbabda75067d2c06eb1ca0bb31c8e4c3b89827`.
The 17 optimizer/reward tests passed under the pinned GPU environment. At
01:25:52 UTC, tmux session `envoy-rl-smoke` launched the fixed five-question base
evaluation, followed by the unchanged SFT adapter, each with a 1,800-second process
limit. Results go to `runs/base-smoke-v1` and `runs/sft-smoke-v1` on the pod.
This supersedes the transfer-blocked status above. The missing billing-stop
credential still requires supervision and an explicit MCP pod stop.

The first GPU invocation exposed a Transformers 4.57.6 configuration merge:
Qwen's saved defaults replaced explicit globally default-valued fields in a
supplied `GenerationConfig`, including `do_sample=False`, temperature 1, and
top-p 1. The `*-smoke-v1` artifacts are invalid as greedy controls and must not
be used for a comparison. The diagnostic session was stopped; no RL ran.

The runner now passes `use_model_defaults=False` and records its complete
generation configuration. A tiny offline Qwen3 regression verifies both greedy
evaluation and unfiltered temperature-1 sampling against conflicting saved
defaults. All 18 focused tests pass on the pod. Corrected results are written
separately to `runs/base-smoke-v2` and `runs/sft-smoke-v2`.

Both corrected smoke runs completed. Base: 0/5 reward-bearing cases, three
submissions without successful investigation, two invalid/unfinished cases.
SFT: 5/5 protocol-valid submissions, two false refusals, one unsupported answer
on an unanswerable question, one answerable submission with zero answer/evidence
overlap, and one correct abstention. Mean reward is 0 versus 0.2; this tiny smoke
does not demonstrate useful answer-quality improvement.

Before sampling, freeze the first two sufficient and first insufficient questions
from prepared training order in `train-probe/benchmark.json`. Sample four attempts
per question at temperature 1 with the same evidence prompt. Require reward
variation on an answerable group before any optimizer experiment; variation only
on unanswerable cases is insufficient given the observed false-refusal problem.
This is a training-signal diagnostic, not another held-out evaluation.

## Completed diagnostic and decision

The 12-attempt probe completed at 02:01:58 UTC on September 19 (September 18
local time), in 1,248.6 seconds including model loading. Per-question rewards:

| Training question | Four sampled rewards |
| --- | --- |
| `qasper_train_d4d771bcb59bab4f3eb9026cda7d182eb582027d` (unanswerable) | 0, 1, 0, 0 |
| `qasper_train_62a3dc90ba427c5985789001a02825c9434ce67d` (datasets) | 0, 0, 0, 0.36106 |
| `qasper_train_d427e3d41c4c9391192e249493be23926fc5d2e9` (boolean) | 0, 0, 0, 0 |

There is answerable reward variation: the dataset answer identified the clinical
notes and cited the correct paragraph, but omitted Wikipedia articles, papers,
and textbooks. Its answer F1 was 0.46154 and evidence coverage F1 was 0.56459.
This is a partial success, not a fully correct research answer.

However, only 4/12 attempts were protocol-valid. Six exhausted the step budget,
one exhausted context, and one submitted inconsistent citations/evidence. Of the
four valid attempts, two falsely refused, one correctly abstained, and one gave
the partial dataset answer. **Do not scale this run:** 8/12 invalid or unfinished
attempts trigger the predeclared gate. No Qwen3-8B optimizer update ran, and no new
trained checkpoint was produced. The probe does not show that RL cannot work;
it shows that this initialization and protocol produce too few usable episodes.

The real PEFT 0.17.1 API was also checked on a tiny random CPU Qwen3: exception
restoration passed, the reference weights stayed unchanged, and a checkpointed
update changed policy weights with finite nonzero gradients and matching initial
ratios. This is stronger than a fake adapter-interface test, but **does not
validate the 8B NF4 backward path or GPU training memory**.

The corrected five-question SFT traces reveal evidence-use failures, not only
format failures. The model saw Europarl/Multi-UN before falsely refusing; it saw
the backward greedy boundary-search procedure but answered only "Boundary
Assembling"; and it repeatedly searched globally instead of reading a supplied
paper ID. SFT used 9.2 actions on average, with four of five episodes reaching
ten actions. These observations come from an unblinded assistant review against
saved source passages and QASPER labels, not independent human evaluation.

All artifacts are saved under `out/research/qasper-rl-v1/gpu-20260918/`, including
`comparison.json`, `sft-smoke-reference-review.json`, `probe-decision.json`, and
the full result archive. The archive SHA-256 was verified against the pod:
`38de09faa88c971d3e2ea0f67232e4e8eec7211ea05eedc34e534eaf5a061b22`.
Pod `jtzpuo7bya1jxb` is **EXITED**. Estimated GPU cost for this session is $0.33;
prior setup and storage charges are separate. The original adapter remains
unchanged and backed up on HF and locally.

Next: repair short investigations, evidence use, and final submissions before
scaling RL. Audit the existing reviewed examples against this exact prompt and
span protocol first; do not assume their old document-ID-only format is aligned.
Then declare a bounded repair experiment and rerun this same diagnostic. A
larger RL job, an improved-model claim, and a product-readiness claim remain gated.

### This was not an overfitting test

A follow-up membership audit found that **none of the three probe questions**
appears in the 189 SFT training examples or the 47 SFT validation examples used
by the archived adapter. Their source is QASPER's official training split; that
does not mean this adapter was trained on these particular questions. The
original JSONL hashes still match the recorded SFT inputs. The audit is saved at
`out/diagnostics/qwen3-overfit-membership-audit-20260918.json`.

The sampling probe performed no 8B optimizer updates and added evidence-span
requirements absent from that SFT protocol. It therefore cannot show that Qwen
is unable to memorize its own training examples. No completed controlled
Qwen3 tiny-set overfit-and-replay experiment was found in the inspected records.

The next training diagnostic should isolate this missing check: choose a handful
of clean, replayable demonstrations; keep the exact prompt, corpus, and output
format fixed; fit those examples; inspect loss and generation on their stored
histories; then run the same questions freely through the environment. If fixed
history fitting fails, inspect labels, gradient flow, optimizer settings, adapter
capacity, and save/reload identity. If fitting succeeds but free execution fails,
inspect where generated histories diverge. This debug set is explicitly trained
on and cannot be used to claim generalization. Declare a bounded budget and
success criteria before launching; no GPU was started for this audit.

**Subsequent diagnostic completed:** the user then authorized this missing check.
The same Qwen3-8B SFT adapter, trained on two clean three-action demonstrations,
passed the tiny-set memorization gate at 20 updates. After a verified save/reload,
autonomous answers on those exact two questions improved from 0/2 to 2/2, with
three actions each. This establishes memorization, not generalization. See
[the full diagnostic and its limitations](QWEN_LEARNING_DIAGNOSTIC.md).

## Targeted-SFT GRPO retry — pre-registered September 20, 2026

The earlier GRPO probe used the older SFT adapter and stopped before any optimizer
update because 8/12 sampled episodes were invalid or unfinished. The newer targeted
SFT checkpoint (`checkpoint-150`, SHA-256
`70143ec6933494efe60cab726e792fcc182965fd13a8561453202b41bba9bfb4`) was trained on
the exact evidence-span protocol. On its 25-question development diagnostic it had
zero execution-error episodes, and on the separate AI-paper retrieved-ID pilot it
submitted 9/10 answers. This justifies repeating the gated RL diagnostic; it does
not justify an unconditional full run.

**Hypothesis:** starting from the targeted SFT adapter will produce enough valid,
non-uniform answer/evidence rewards for group-relative updates to improve autonomous
QASPER behavior beyond unchanged SFT.

Use GRPO rather than actor-critic PPO for this retry. Both use a clipped per-token
policy objective here, but GRPO derives advantages from four sampled answers to the
same question and does not require a learned value model. That keeps the experiment
within one 24 GB GPU and reuses the tested update path.

The gates, fixed before new GPU output, are:

1. Run the current SFT adapter greedily on the existing five-question development
   smoke set. Require at least 4/5 protocol-valid submissions, at least 4/5 successful
   document investigations, and no execution-error episode.
2. Sample four trajectories for each of the first five questions in the frozen
   500-question training manifest. Require at least 16/20 protocol-valid submissions,
   reward variation in at least three of five groups, and variation in at least two
   answerable groups. Otherwise stop without an update.
3. Run one optimizer update. Require the step-zero importance ratio assertion to
   pass, finite nonzero gradient norm, finite KL, a changed policy adapter, an
   unchanged frozen reference adapter, and a reloadable saved checkpoint.
4. Only if all three gates pass, allow a bounded 20-update pilot. Stop after five
   consecutive all-uniform batches. Compare saved candidates against unchanged SFT
   on the frozen 40-question development set using greedy decoding. Promote only if
   supported answers improve by at least four, false refusals are at most SFT+1, and
   tool reliability does not regress.

Training reward is `qasper-answer-evidence-v1`, computed from external official
answers and mapped evidence spans. It is a proxy for answer/evidence overlap, not a
semantic support judge. Training-curve gains alone cannot pass the experiment. The
GPU diagnostic and pilot together have a $5 ceiling; stop the pod through RunPod
after artifacts are downloaded.

## Dr. GRPO five-update ablation — pre-registered September 20, 2026

The completed 20-update pilot peaked at checkpoint 5 on the frozen 40-question
development set, then regressed: unchanged SFT scored 0.45626, checkpoint 5 scored
0.52325, and checkpoint 20 scored 0.45353. Checkpoint 5 improved the automatic
supported-answer proxy from 10 to 12, correct abstentions from 12 to 14, and false
refusals from 4 to 3. It did not meet the stricter pre-registered promotion target of
four additional supported answers, so it remains a candidate rather than a promoted
model.

The run exposed a specific optimization risk. Thirteen of 40 prompt groups had
uniform rewards and therefore no policy-gradient signal. Eight of the 27 non-uniform
groups had a reward spread no greater than 0.15. Standard GRPO divided those small
centered differences by each group's standard deviation; for example, rewards
`[0, 0, 0.027, 0]` became advantages `[-0.5, -0.5, 1.5, -0.5]`. This can give a
tiny and potentially noisy lexical-overlap improvement the same update scale as a
clear success. Dr. GRPO removes that per-group standard-deviation normalization.
The existing implementation already uses a fixed token denominator rather than
per-response length normalization.

**Hypothesis:** centered, unscaled group rewards will preserve useful relative
feedback while preventing tiny reward differences from dominating updates. This
should make the first five updates at least as reliable as the previous checkpoint 5.

This is a one-variable ablation. Start again from the same targeted SFT checkpoint
and keep the base revision, frozen SFT KL reference, `qasper-answer-evidence-v1`
reward, train benchmark and corpus, batch size 2, group size 4, temperature 1,
learning rate `1e-6`, seed 42, context/action/step limits, and prompt unchanged.
Change only advantage computation from `(reward - group_mean) / group_std` to
`reward - group_mean`. Run exactly five updates and save every update.

Evaluate checkpoints 1 through 5 greedily on the same frozen 40-question development
set. It is now a development/model-selection set, not an untouched final test. Treat
the ablation as promising only if one checkpoint beats the previous checkpoint 5's
0.52325 mean reward in a paired comparison while keeping false refusals at most 3,
protocol-valid submissions at least 39/40, and execution-error episodes at most 3.
Automatic overlap remains a proxy: promotion still requires semantic review. Do not
extend beyond five updates in this experiment. Stop GPU billing after verified
artifacts are downloaded; the compute ceiling is $5.

## Dr. GRPO ablation result — completed September 21, 2026

The five-update run completed under the declared limits. All 40 training episodes submitted, all
five policy fingerprints differed from the starting adapter, and the frozen-reference fingerprint
remained unchanged. Peak allocated GPU memory was 19.33 GB. The update-level mean sampled rewards
were 0.1875, 0.0804, 0.0627, 0, and 0.2422; these values come from different sampled questions and
are not a learning curve.

Greedy evaluation on the same frozen 40-question development set produced:

| Checkpoint | Mean reward | False refusals | Valid submissions | Execution-error episodes |
| --- | ---: | ---: | ---: | ---: |
| Centered 1 | 0.4766 | 4 | 39/40 | 3 |
| **Centered 2** | **0.5485** | **3** | **39/40** | **3** |
| Centered 3 | 0.5063 | 5 | 39/40 | 3 |
| Centered 4 | 0.4786 | 3 | 40/40 | 3 |
| Centered 5 | 0.5020 | 4 | 39/40 | 3 |

Checkpoint 2 passed the pre-registered automatic gate and became the best automatic checkpoint in
this experiment series. It improved on unchanged SFT by 0.0922 mean reward, with a paired bootstrap
95% interval of [0.0146, 0.1914]. Its 0.0252 lead over the previous standard-GRPO checkpoint 5 has
a paired interval of [-0.0518, 0.1209], so the experiment does not establish superiority of the
centered optimizer variant over standard GRPO.

An unblinded review of the seven SFT-to-checkpoint-2 changes found four clear improvements, one
automatic improvement that remained semantically wrong, one evidence-offset change with the same
answer, and one real specificity regression. This is encouraging evidence that GRPO improved the
SFT worker on this development set. It is not a final generalization claim because this set was
used for checkpoint and method selection. The detailed analysis is in
[`qasper-grpo/README.md`](qasper-grpo/README.md).

The verified local archive is
`out/research/qasper-rl-v2/artifacts/envoy-drgrpo-centered-20260921.tar.gz` (SHA-256
`db7412a82b1a9513cf49de6344374f93aea274e13ce38abea57c66c36c166aed`). Estimated A40 compute was
$3.06, below the $5 ceiling. The RunPod pod is stopped.

## Paper-disjoint confirmation result — September 21, 2026

Centered checkpoint 2 was then compared with unchanged SFT on a previously frozen 40-question
confirmation set covering 39 unseen QASPER papers. The development improvement did not replicate:
SFT scored 0.3462 automatic reward and GRPO scored 0.3531, a +0.0069 paired difference with a 95%
bootstrap interval of [-0.0169, 0.0375]. Thirty-eight questions tied automatically.

The pre-registered identity-blind semantic review favored SFT by one question: SFT produced 13
passes, 2 partials, and 25 failures; GRPO produced 12, 3, and 25. The automatic GRPO win replaced a
false refusal with an unsupported answer and was not a semantic win. GRPO's only semantic change
was a pass-to-partial regression. The checkpoint fails the confirmation gates and is not promoted.
See [`QASPER_GRPO_CONFIRMATION.md`](QASPER_GRPO_CONFIRMATION.md) for the complete result and
artifacts.
