# V5 reader RL pilot: preparation and decision rule

> Subsequent October 24 update: the first integration attempt ran one gradient
> update but failed the optimizer-counter gate. Local repairs are tested; the
> corrected GPU run and paired evaluation remain pending. Both temporary resources
> are deleted. See [the actual report](PRIME_RL_INTEGRATION_20261024.md).

October 24, 2026. This is the concrete protocol for the user's approved RL
direction. **Preparation only: no new GPU, real-model inference, Qwen update or commit.**
The proposed paid allowance is not authorized. The current experiment assets are
[protocol.json](../examples/reader-rl-pilot-v1/protocol.json) and
[training-review.json](../examples/reader-rl-pilot-v1/training-review.json).

Hypothesis: ten native GRPO LoRA updates from the original pinned V5 can improve
complete, source-supported short-fact answers on separate papers. The expected
signal is more complete supported answers, without additional unsupported
answers or execution failures, under bounded inference resources. This is an
initial pilot, not a full QASPER benchmark or a claim about general research.
Evidence annotations choose the passage windows for this curriculum. Neither
training nor evaluation measures finding relevant papers or selecting passages.

## Data review and one launch hold

The implementing assistant re-read all 82 training-role source questions, accepted
answer variants and attributed supporting excerpts. This was a source-excerpt
review, not a full-paper audit or independent human validation. Its 82 row/card
bindings are preserved in the new review file; independent human review is null.

One paper is held out of **training** because its finite whitelist accepts the
incomplete answer `the public financial news dataset released by`. Removing the
`BIBREF4` marker left a phrase that does not identify the dataset. The original
excerpt names Reuters and Bloomberg. The whole training question is held until
an explicit future reference revision; no immutable source or reward key changed.
This held training paper does not become an evaluation example.

Result: **81 eligible training questions**, eight separate reward-validation
papers, and the existing 12 locked model-evaluation papers. The launch plan now
requires the attributed review, binds its hash, and excludes its holds. A plan
that silently reintroduces the held question is rejected. Old prepared plans
must be regenerated because implementation/review identity changed. The generic
taskset still inventories all 82 training questions; the bounded launch driver
applies the 81-question pilot selection.

Retained references remain deliberately narrow. Notes preserve limitations:
BBC/CNN are introduced as examples, Title-to-Story is called *a* baseline, and
the quoted 10% WER improvement is for the speaker-open setting. Excerpt review
does not establish exhaustive lists across a whole paper. Correct unlisted
paraphrases and shorter valid citations can still score zero. The fixed strict
reward is unchanged; guarded F1 is not enabled. A regression test demonstrates
that the old reward still accepts the incomplete alias while the pilot excludes
its question, keeping the intervention explicit.

No locked evaluation reference was decoded or reviewed during preparation.
Their bytes were hashed for existing integrity checks, and only their previously
locked IDs/source metadata were used for the protocol. They cannot supply new
training examples, tuning decisions, or replacement questions.

## Two experiments, with a fixed stopping point

1. **Integration check:** two updates; at most 256 generation requests, 2,097,152
   reserved tokens, and 1,800 seconds of workload. Require exact checkpoint and
   optimizer continuity, sampled-policy identity and native scoring gates. No
   quality evaluation. Its adapter is not the pilot's initialization.
2. **One training pilot and paired evaluation:** start again from original V5,
   finish exactly ten verified updates, select step ten, and compare it with
   unchanged V5 on all 12 locked papers. No checkpoint search or hyperparameter
   sweep. Training permits at most 1,024 generation requests, 8,388,608 reserved
   tokens and 3,600 workload seconds. Questions are shuffled with seed 42, sampled
   four times each, and consumed once without replacement. Uniform groups do not
   train. If the pool, budget or healthy mixed groups run out before ten updates,
   report incomplete and keep evaluation closed.

Native settings match the smoke: rank 4, alpha 8, dropout 0, LR 1e-6, centered
GRPO with native PPO clipping 0.8–1.2, no reference-KL term, sequence limit 8,192,
temperature/top-p 1, and the frozen four-step Docker investigator. Ten updates
is a bounded engineering choice, not a research-established optimum. The
[repeatable launcher runbook](PRIME_RL_REPEATABLE_RUN.md) pins the actual stack.

## Paired evaluation

Freeze the final checkpoint identity and this protocol before opening the locked
references. Review all 12 reference/source pairs for in-scope answerability
before reviewing model outcomes. Do not replace, repair or quietly drop an
ambiguous reference after seeing which arm benefits. Unresolved reference or
review ambiguity makes the quality gate inconclusive.

Run both models with the same native renderer/transport, prompt, source snapshots,
passage ranges, sampling settings, four steps and 1,024 tokens/action. Use seed
`42 + question_index` for both arms, the locked question order, and V5 first,
then RL. One episode per question per arm gives **24 episodes**, at most 288
generation requests, 2,359,296 reserved tokens and 1,800 workload seconds. There
are no retries or replacement episodes. Same-seed sampling reduces one source
of variation; it does not guarantee deterministic vLLM execution or establish
repeatability from one seed. Record hardware, all decoding settings and backend.

Preserve raw outputs and traces. A separate derived pack masks model identities
for review and preserves a private arm mapping until judgments are recorded.
Grade requested-answer completeness, correctness and each material claim against
the actual inspected/cited source. Preserve numbers, units, entity bindings and
negation. Quote matching establishes provenance, not semantic support. A false
refusal is not a supported answer on these answerable tasks. An incomplete but
otherwise supported answer and an unsupported factual assertion are distinct.
An unknown finite-answer spelling requires semantic review, not an automatic fail.

The initial **provisional positive signal** requires all of these:

- All 24 paired episodes and source reviews available; no unresolved judgments.
- At least **three net additional fully supported answers** out of 12, with at
  most one formerly supported answer becoming a failure.
- No increase in episodes with unsupported material claims, and no question
  newly acquiring an unsupported material claim. No increase in episodes with
  execution failures. All refusals, partial answers and errors remain visible.
- Mean reserved tokens and median episode latency at most 1.25× V5; p95 episode
  latency at most 1.5× V5. Report startup separately. Missing comparable timing
  or cost records makes the resource gate inconclusive; include startup in spend.

These are predeclared engineering tolerances, not a statistical significance
test. Twelve papers and one seed per arm give limited precision. Report every
paired win/loss/tie, uncertainty, false refusals, request/token use, actual GPU
lifetime and spend. Official Answer F1 and strict training reward are secondary;
neither can establish semantic improvement alone. Model-assisted judgments must
be labelled. Do not fill independent-human-review fields with model judgments
or promote a pilot signal into a validated general model-improvement claim.

## Proposed budget and current readiness

Proposal: one visible 80GB GPU, sequential inference/training, at most **two hours
of total owned GPU lifetime** and **$5 estimated total spend**, including startup
and storage. These are proposed ceilings, not a price quote or provider-enforced
dollar cap. Workload timeouts alone do not stop GPU billing. Before launch,
verify fresh prices/capacity/storage, obtain explicit budget authorization, and
install the separate resource-lifetime/cost watchdog with cleanup receipts. If
the projection does not fit, do not launch or silently raise the allowance.
The current launcher retains optimizer checkpoints: plan for at least **464 GiB
of initial free disk** beyond installed assets if preserving both the two-update
and ten-update runs. This is storage, not GPU memory. The training phase alone
requires 360 GiB free at its preflight. Include storage in the fresh cost check.

The training launch selection and paired evaluation runner are now implemented
and locally tested. [Evaluation runbook](READER_PAIRED_EVALUATION.md) describes
candidate freeze, gated reference review, shared capped collection, identity-masked
claim review and the predeclared reducer. Scripted replies passed through the
native renderer and real Docker for both arms; no held-out reference was opened
and no Qwen inference or learning was measured. The protocol's numerical settings
remain unchanged. Official F1/strict reward are secondary existing diagnostics,
not automatically computed by this collection/review command.

Consecutive GPU updates remain unverified. The evaluation controller now accepts
an explicit loopback SSH forward, preserving local Docker execution while native
inference runs remotely. Training's current sequential launcher still requires
Docker and CUDA on its executing machine; a new GPU pod's Docker capability is
not certified. That training layout and the independent provider-lifetime/cost
watchdog remain pre-launch requirements. No existing paid allowance is reused.

If the bounded pilot is incomplete or shows no convincing gain, retain V5 and
publish the working MCP/Studio and the observed limits. Further paid experiments
require a specific observed failure, proposed correction and a new budget.

Validation results for the new runner are recorded in
[the evaluation runbook](READER_PAIRED_EVALUATION.md) and `CODEX_HANDOFF.md`.
Native fixtures use authored replies and synthetic probabilities with real
Docker, not Qwen inference or training. New logs/plans:
`out/reader-paired-prep-20261024/`; previous preparation remains preserved.
