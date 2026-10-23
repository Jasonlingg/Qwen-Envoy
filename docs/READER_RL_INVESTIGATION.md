# Reader RL feasibility investigation — October 22, 2026

**RL is relevant, but no current reader reward passes the offline audit. Do not
launch training yet.** RL can generate its own action trajectories; it still
needs diverse questions, reliable outcome labels, adequate exploration and compute.
This investigation ran no model inference, GPU activation or optimizer update.
The `cce82c2` runtime remains frozen. Nothing establishes a new Qwen improvement.

The previously proposed 24-trajectory SFT continuation is on hold. Its size was
an engineering convenience, not an evidence-based learning requirement. A new
history check also found omissions in its exclusion registry, described below.
The examples remain useful as exposed protocol/reward diagnostics; their frozen
training-ready flags are historical and do not authorize training now.

## What the research supports

Primary papers read on October 22, 2026; results are those authors' experiments,
not measurements on our model. Question counts, unique papers, sampled attempts,
action examples and optimizer updates are different units.

| Source | Evidence relevant to us | Limits of transfer |
| --- | --- | --- |
| [Search-R1, COLM version](https://arxiv.org/html/2503.09516v3) | Uses question/answer data, multi-turn retrieval, outcome rewards and masks retrieved tokens from policy loss. Studies Qwen2.5 3B/7B; the published setup uses eight H100s. | Wikipedia search and short-answer correctness differ from our assigned scientific-paper excerpts, gap queries, Qwen3/V5 initialization and small-budget LoRA setup. Its resource use is not our required hardware minimum. |
| [R1-Searcher](https://arxiv.org/html/2503.05592v1), §§2–4 | RL without demonstration distillation or SFT cold start. Stage-one task counts total 350; stage-two counts total 8,148, before accounting for repeated rollouts. | The stages need curated QA data and rewards. They are not a universal data-size rule. The authors report reward-hacking cases, including answers containing the right words plus extraneous content. |
| [s3](https://arxiv.org/html/2505.14146v1), §§3–4,7 | Trains a Qwen2.5-7B searcher with a frozen generator, rewarding generation gain over a fixed RAG baseline. Its checkpoint uses 20 updates × batch 120 ≈2,400 examples, from a filtered 70,286-question pool. | It still requires QA supervision, generator/judge inference and task filtering. The 2,400 figure does not establish 2,400 distinct papers or independent trajectories. Optimizing a searcher is not the current passage-reader task. |
| [Search-R1 training analysis](https://arxiv.org/html/2602.19526v1), §§4–5 | Finds answer avoidance under F1 reward, interactions with action penalties, and optimizer-dependent stability. | One experimental setting does not establish that REINFORCE always beats GRPO. Its proposed no-answer penalties need care because our task includes legitimate gaps. |
| [DeepSeek-R1](https://www.nature.com/articles/s41586-025-09422-z) | Discusses the value of verifiable outcomes and limitations of model-assigned feedback for less-verifiable tasks. | Mathematical or executable correctness does not give us a semantic-support judge for paper claims. |

The s3 division of labor is conceptually close to the wider host/worker design.
Adopting its objective would require measuring the frozen host's downstream
answer improvement and paying for reward inference. It would change the
training task from passage extraction to retrieval selection; that is a separate
experiment, not a reason to unfreeze the current harness during this audit.

## Local RL evidence already exists

[September QASPER experiments](QASPER_RL_EXPERIMENT.md) used an earlier Qwen3 SFT
adapter and a different runtime. The first probe produced only 4/12 protocol-valid
attempts and stopped without an update. Later GRPO runs did update weights.
Centered checkpoint 2 improved development automatic reward by 0.0922, but its
[fresh 40-question confirmation](QASPER_GRPO_CONFIRMATION.md) did not replicate:

| Confirmation measure | Unchanged SFT | Centered GRPO |
| --- | ---: | ---: |
| Automatic overlap reward | 0.3462 | 0.3531 |
| Source-reviewed pass / partial / fail | 13 / 2 / 25 | 12 / 3 / 25 |

The review was identity-blind Codex-assisted, not independent human review.
An unsupported `Yes` earned an automatic improvement; another answer lost a
required selection step. This is evidence against repeating the same objective
and calling reward growth a research-quality gain. It does not rule out RL with
a better task/reward. These adapters are not the current V5 checkpoint.

## Actual offline reward audit

[Saved inputs, per-record scores and reproduction](../examples/reader-rl-investigation/README.md).
The audit has **150 diagnostic records**: 24 saved authored demonstrations,
3 actual exposed V5 outputs, 2 previously replayed corrections and 121 authored
counterfactual terminals (including 12 annotation-string controls). It does not
have 150 new Qwen rollouts. No proposed Python executes, even locally. Only the
12 training-paper pairs are used from the new pilot; development sources and
the reserved research_reader_v1 pack are not inspected for this audit.

Counterexample labels come from source-visible implementing-assistant review or
explicit counterexample construction. They are adversarial development cases,
not independent gold labels or a measured real-world error distribution.

| Diagnostic scorer | Unacceptable records receiving full reward | Acceptable records receiving zero |
| --- | ---: | ---: |
| Unchanged legacy QASPER overlap reward | 12 | 13 |
| Two-read/scoped-terminal gate + answer/evidence overlap | 24 | 0 |
| Same gate + exact annotated answer match | 36 | 14 |

There are 111 labeled unacceptable cases, 27 acceptable controls and 12
reference-string controls. These counts are not a fair ranking of training
algorithms: the candidate scorers intentionally isolate different weaknesses.
All fail the predeclared adversarial gate; none is deployed as a reward.

Concrete findings:

- The legacy scorer gives full reward to an exact reference and gold offsets
  after only an unrelated distractor read. It checks that *some* investigation
  occurred, not whether that citation was inspected. The scoped gate blocks this.
- Both scoped prototypes give full reward to `Information unavailable` / `research`
  and to a gap claiming impossibility with an unrelated potato-recipe query.
  Syntax and a reviewed missing-scope label do not establish useful uncertainty.
- Exact answer matching plus a real in-scope quote still gives full reward for
  twelve irrelevant-quote examples. Exact matching is not an entailment check.
- A reviewed answer binds VATEX to video captioning and WMT2014 to translation.
  Swapping those bindings receives exactly the same overlap reward, **0.347826**.
- The scoped overlap scorer handles the accepted translation output after applying
  the existing runtime's 1,200-character quote clipping. The initial private audit
  incorrectly treated its 1,600-character submitted spans as ineligible; corrected
  audit code validates the full read range first, then scores clipped packet spans.
  Runtime clipping behavior was not modified.

A functioning provenance guard is necessary but cannot repair these semantic
reward problems. The exact-match prototype also rejects valid paraphrases.
The old MuSiQue `outcome-v1` and QASPER `qasper-answer-evidence-v1` remain unchanged.

## Exclusion-registry correction and data hold

The RL/history investigation found that the previous inventory selected JSONL
training pools and evaluation-looking metadata paths, but missed training pools
serialized as `benchmark.json` or `questions.json`. Identifier-only inspection
of 13 omitted historical files adds **96 paper IDs and 659 question IDs** beyond
the frozen registry. Seven selected training papers, **14/24 trajectories**, have
prior prepared-pool or evaluation membership under the declared conservative rule.
The six development papers are unaffected by this supplemental scan.

This is not proof that all those questions supplied optimizer updates. For
example, an upstream annotation source pool is larger than its accepted training
export. But it contradicts the original claim that every selected paper was
absent from all prior prepared pools. The frozen 213-file registry and all old
artifacts remain preserved as the historical record. Training use is on hold.
A structured-benchmark coverage gate supersedes the old registry schema.

The old selection's strict rules left only **16 eligible train questions** after
its exclusions/rejections. Even that count predates the supplemental exclusions.
We cannot promise hundreds of fresh examples by rerunning the same selector.
A larger study needs an explicit data policy: for example, deliberate reuse of
verified training-only pools, or additional reviewed facts/scopes. Training-pool
reuse is legitimate when disclosed; it cannot masquerade as fresh evaluation.
Never relax evaluation-paper exclusions or open the reserved pack to get data.

## Trainer readiness

Static inspection, not trainer execution, finds:

- `scripts/train_qasper_grpo.py` constructs `DocumentExplorationEnv(use_docker=False)`.
  It uses the old `QASPER_SYSTEM_PROMPT` and no `QwenInvestigator`/`submit_gap` path.
  It must not run model-authored Python for this task. The legacy MuSiQue entry
  point's `use_docker=None` also does not guarantee Docker-only execution.
- Tested helpers provide action-token log probabilities, clipped updates,
  frozen-reference checks, one-epoch updates and centered-advantage support.
  This is reusable optimizer machinery, not a ready V2 reader training adapter.
- A future adapter must use the exact effective V2 prompt, real Docker observations
  and current terminal contract. Store original generated token IDs and their
  probabilities; never optimize fabricated, cleaned or retokenized action tokens
  as if they were the sampled tokens. Mask observations and previous history.
- It must separate infrastructure failures from zero-reward task outcomes, retain
  complete groups, measure uniform groups, preserve reference weights, stop on
  nonfinite updates, and verify save/reload identity. Hosted inference alone
  cannot update the policy's weights.
- Historical 19.33 GB peak memory came from a different adapter/run. It is not
  a current V5 capacity guarantee. Sequential rollout collection can reuse one
  loaded base plus adapters; actual GPU training capacity remains unmeasured.

No trainer, active reward, endpoint configuration, serving checkpoint or frozen
runtime file was changed during the investigation.

## Proposed next experiment, with explicit gates

This is a proposal, not an authorized or executable training launch. The earlier
24-example continuation is superseded as the immediate next action.

1. **Resolve data and reward first.** Freeze a reviewed, training-only task pool
   with explicit fact/alias/support keys, scoped missing labels and provenance.
   Include wrong numeric units, reversed relations, negation, irrelevant quotes,
   copied references, unsupported clauses and false or unhelpful gaps in reward
   validation. Obtain independent review where available; disclose its absence.
   A single short fact with a reviewed support key is a more tractable initial
   RL objective than unconstrained paper synthesis. Gap query usefulness still
   needs a tested semantic assessor or must remain explicitly unoptimized and
   independently evaluated; do not award it a usefulness claim from string checks.
2. **Declare the actual objective and split.** Treat the 150 cases as exposed reward
   development data. Lock separate reward-validation cases and model-evaluation
   papers before observing new results. Pick dataset size by a bounded learning
   curve and available reviewed diversity, not by assuming 24, 2,400 or 100,000
   is sufficient. The current six-question development set cannot support a broad
   promotion claim. Do not automatically collect or inspect a new test pack.
3. **Build the V2 training adapter and prove parity locally.** Docker-only; pinned
   image, V1 bytes unchanged; exact prompts, scoped inspection, atomic gaps,
   raw-token masks, response clipping and replay. Keep the model's actions
   multi-step and tool-mediated. No retrieval or prompt tuning in this comparison.
4. **Only then consider a budgeted exploration probe.** A proposed *diagnostic*
   ceiling is 16 named training questions × four sampled attempts = 64 episodes,
   four actions each: at most 256 generation requests, 262,144 generated tokens,
   and 2,097,152 total reserved input/output tokens at 8,192 per request. These
   are operational limits, not a sufficient training-set size. Actual IDs,
   checkpoint/adapter SHA, corpus, reward version, sampling distribution, software,
   hardware, per-request/context limits, active-time ceiling and dollar budget
   must be frozen before execution. Current paid/GPU allowance is zero.
5. **Require signal before updates.** Predeclare protocol-validity and answerable
   reward-variation thresholds; inspect paired reward/semantic disagreements.
   Uniform groups supply no relative task signal. The old experiment's variance
   is not evidence that current V5 has it. No optimizer choice cures a bad reward.
   Centered GRPO is a reproducible local baseline because its helpers exist;
   research does not establish it as the best algorithm. Do not start an optimizer
   sweep. One genuine GPU update would first have to show correct initial ratios,
   finite gradients/KL, changed policy, unchanged reference and verified reload.
6. **Require matched held-out gains.** Any later bounded training pilot compares
   candidate against unchanged V5 and base under matched prompts, tools, scopes,
   hardware and greedy decoding. Keep supported-answer quality and useful-gap
   quality separate; report false refusals, errors, tokens and latency, with
   source-visible identity-blind review and paired uncertainty. Higher training
   reward, more valid Python, or this offline audit cannot promote the model.

For a fixed critic-free rollout design, `U × B × G` counts training episodes and
`U × B × G × A` bounds action generations. It excludes evaluation and any generator
or judge calls. Those must be budgeted separately. Compare an SFT alternative by
teacher generation/review cost as well as training cost, and RL by rollout/reward
inference as well as backpropagation. No price or GPU-minimum claim is made here.
If a required hard financial limit cannot be enforced, skip paid calls.
