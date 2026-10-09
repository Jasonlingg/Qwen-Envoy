# Qwen learning diagnostic — September 18, 2026

The user authorized running the short-evidence and tiny-overfit diagnostics after
reviewing failed execution transcripts. This is a debugging experiment, not a
product evaluation or permission for a larger training job.

## Hypotheses and decision rules, before running

1. **Evidence use:** with the existing adapter, a short exact source passage may
   produce a correct answer where the accumulated search history failed. Compare
   base and SFT on short evidence and the same evidence appended after the saved
   history. Keep the terminal instruction and trailing evidence identical. Record
   full prompts. Success only in the short condition suggests history sensitivity;
   this small intervention does not identify the internal mechanism.
2. **Memorization:** the existing rank-4 adapter can learn two clean three-action
   demonstrations under the actual inference prompt and tokenizer. Both examples
   come from official QASPER training papers. No validation example enters updates.
   Measure masked next-action loss, greedy next-action match at all six fixed
   teacher histories, and independent full episodes on those same two questions.
3. **Execution gap:** fitting the six fixed next actions while failing autonomous
   episodes points toward divergence from demonstrated histories. Failure to fit
   within the bounded run means optimization/capacity/pipeline remain unresolved;
   it does not prove the model cannot learn the task.

The training questions concern the clinical-abbreviation corpus and the classifier
used for German poetry. Their source evidence directly supports the answers. Each
demonstration searches within the supplied paper, inspects the relevant paragraph
using an anchor visible in search results, and submits the complete answer with
exact offsets. The actual tools are replayed locally before use. This is an
assistant review, not independent human review.

## Fixed configuration

- Base: Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Start: the existing private prefix-aligned SFT adapter; hash recorded at run time.
- NF4 double quantization, BF16 compute, existing rank 4/alpha 8 adapter.
- Exact per-action chat-template prefix; only next-action tokens receive loss.
- Diagnostic training: constant LR 2e-4, no weight decay, dropout disabled,
  gradient checkpointing, equal weighting of the six action examples per update.
- At most 60 updates; inspect at updates 5, 10, 20, 40, 60. Early stop when every
  fixed action matches and mean action loss is below 0.05.
- Greedy generation, thinking disabled, top-k 0/top-p 1, fixed seed 42. Up to 512
  generated tokens per diagnostic action and ten actions for autonomous episodes.
- Validate finite gradients, changed adapter parameters, and save/reload equality.
- Save all outputs and the diagnostic adapter separately; preserve the starting one.

The short-evidence control includes the previously inspected Europarl/MultiUN
validation question. It is a diagnostic condition, not an unbiased test score,
and is never used for training. Gold passage selection is supplied by the evaluator
and must not be credited as autonomous retrieval.

## Budget and interpretation

Resume the previously authorized A40 pod only after local preparation. Maximum
supervised session: two hours at the currently reported $0.49/hour (under $1 GPU
compute, storage separate). Stop earlier if diagnostics finish or cannot proceed.
Stop the pod explicitly through RunPod after downloading results. A process timeout
is not a billing shutdown; no unattended billing is permitted.

Even perfect memorization is only a pipeline sanity check. Generalization and the
weekly research product remain unproven. A stronger held-out evaluation follows
only after these diagnostics establish which intervention merits testing.

## Completed result — September 18 local / September 19 UTC

The predeclared memorization gate passed at **20 updates**, so the run stopped early.
The adapter was saved, unloaded, reloaded, and then used for fresh autonomous
episodes. All 504 trainable adapter tensors changed; all saved/reloaded tensors
matched exactly.

| Measurement | Starting SFT | After 20 updates |
| --- | ---: | ---: |
| Mean next-action loss, six fixed teacher histories | 0.821410 | 0.00006195 |
| Mean next-token accuracy over those actions | 84.77% | 100% |
| Exact greedy next-action matches | 0/6 | 6/6 |
| Correct autonomous answers on the two fitted questions | 0/2 | 2/2 |
| Actions per autonomous episode | 10, 10 | 3, 3 |
| Mean autonomous episode time | 56.87 s | 17.61 s |

Both starting episodes falsely refused. Both reloaded episodes independently
searched within the specified paper, located and printed the relevant paragraph,
and submitted complete answers with the exact supporting spans. Their answer F1,
evidence F1, and reward were all 1.0. The assistant checked the answers against the
source text; this was not an independent or blinded semantic review.

At update 10 both final answers were already complete and correctly cited under
fixed teacher histories. The sole exact-match difference was an extra comma in
the clinical answer. The stricter predeclared gate passed at update 20.

### Evidence controls

Before any updates, base and starting SFT both answered Europarl/MultiUN correctly
when the exact paragraph and a final-submission instruction were supplied, in
both the short and long-history conditions. Token-overlap reward was 0.8 because
the models added "and"; the names and source spans were correct.

On the clinical question, short evidence produced a fabricated "MIMIC-III dataset"
answer from base and a false refusal from SFT. With the saved history plus that
same evidence, both models identified the physician logs but omitted the enriched
corpus resources. Therefore this small check does not support a general explanation
that shorter context fixes the problem. Models can cite the correct span and still
give an unsupported or incomplete answer.

These conditions supplied gold-selected evidence and an explicit instruction to
submit. They establish conditional answer behavior, not autonomous retrieval
success, and do not isolate the cause of the earlier unassisted failures.

### What is established and what remains open

The current Qwen3-8B/rank-4 LoRA combination can memorize these two short
code-execution demonstrations and execute them after reload. The previous claim
that it could not even overfit is now directly contradicted by a controlled test.
This is **training-set memorization**, not held-out improvement or product readiness.

The diagnostic reused the project's exact per-action tokenization helper and
execution harness, but used a small explicit optimizer loop with constant LR,
disabled dropout, and equal action weighting. It does not validate every detail
of the older TRL training path or identify which older setting/data issue caused
poor generalization. Long-sequence GPU training was not exercised: the longest
training sequence here was 1,304 tokens.

The next justified experiment is a bounded training run using reviewed, short,
protocol-aligned demonstrations across more training papers, followed by a common
held-out comparison. The exact same two questions must never be presented as
held-out evidence. No larger run was started automatically.

### Artifacts and cost

- [Readable results and before/after transcripts](../out/diagnostics/qwen-learning-20260918/RESULTS.md).
- [Reviewed training demonstrations](../out/diagnostics/qwen-learning-20260918/data/examples.md).
- Local archive: `out/diagnostics/qwen-learning-20260918/results.tar.gz`.
- Archive SHA-256: `e3a7e6073b0046262baaa61fc31c91c071066ab211736aa5cec819d3cfeae77e`.
- Final adapter SHA-256: `f2a14a2b5afca30c26a389e2d775c49bbc797483e384a417ba5c61dc7118dd2c`.
- Checkpoints 5, 10, 20, and `final` are all downloaded; original adapter preserved.
- Runtime of the diagnostic: 564.6 seconds. Peak allocated GPU memory: about 17.4 GiB.
- Pod `jtzpuo7bya1jxb`: resumed 02:43:15 UTC, confirmed **EXITED** at 02:58:19 UTC.
- Estimated session GPU cost: **$0.12** at $0.49/hour; storage is separate.

The container's earlier files were absent after restart, so the authorized local
backup and pinned dependencies were restored before running. No teacher API calls
were made. Results and all diagnostic weights were verified locally before stopping.
