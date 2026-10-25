# Repeatable native Prime-RL reader run

> Subsequent October 24 update: the first integration attempt ran one gradient
> update but failed the optimizer-counter gate. Local repairs are tested; the
> corrected GPU run and paired evaluation remain pending. Both temporary resources
> are deleted. See [the actual report](PRIME_RL_INTEGRATION_20261024.md).

October 24, 2026. The new sequential driver is implemented and locally checked.
**Consecutive GPU updates have not been run with this driver.** The separate
[October 23 smoke](PRIME_RL_GPU_SMOKE.md) established one real update, not model
improvement. This pass makes that path repeatable without adding a custom trainer.
No new inference, GPU rental, training, upload or Git commit was performed.
Two public Hugging Face metadata reads recorded the pinned model file identities;
no weights were downloaded during this pass.

Hypothesis: complete on-policy mixed-reward groups can survive consecutive native
updates, optimizer reloads and cold inference loads. Expected signal: each cycle
uses the previous checkpoint's adapter and optimizer state, sampled probabilities
agree with native scoring, and the resulting adapter changes and reloads exactly.
Stop on any identity, infrastructure, scoring, budget or reload failure. This is
an integration gate; improvement needs a separately declared evaluation after it.

The driver runs this sequence on one visible 80GB GPU:

```text
pinned V5 (cycle 1) / previous verified adapter (later cycles)
  → cold native vLLM process → successful LoRA load RPC + matching engine path
  → four complete Docker reader episodes on one training-role question
  → skip uniform-reward groups; abort on infrastructure failures
  → stop inference
  → native checkpoint load + exact adapter comparison + sampled-token scoring
  → native TrainSink/GRPO batch → native Prime-RL trainer: one update
  → fresh native DCP reload: optimizer steps/state, exported and startup tensors
  → portable adapter for the next cycle
```

The controller uses the existing Verifiers adapter, non-thinking default renderer,
four-step investigator, assigned source windows, frozen effective V2 prompt and
network-disabled Docker sandbox. Raw traces remain untrainable. Only a complete
healthy mixed group is copied for native batch compilation; that batch cannot
reach the trainer until the native scoring gate passes. Generic Verifiers
training remains blocked. No source discovery or reader behavior was changed.

The reward remains `reader-short-fact-reference-v1`: a finite accepted-answer
contract plus inspected, attributed support checks. It is narrow, can reject
correct paraphrases, and is not a semantic judge. Guarded F1 stays an offline
prototype. Independent human review remains null. A
[follow-up source review](READER_RL_PILOT.md) filters the 82-question training
pool to 81 eligible pilot questions. These are shuffled with seed 42 and consumed
without replacement; validation and the 12
locked evaluation papers are not sampled.

## Prepare locally

From the repository root, preparation only writes a new run directory:

```bash
python -m src.training.prime_run out/reader-prime-next \
  --assets /workspace/reader-assets \
  --updates 2 --requests 256 --tokens 2097152 --seconds 3600
```

`plan.json` records model/software identities, source/reward/prompt hashes,
implementation hashes, training question order and limits. Preparing at a
different execution path is required if moving to another machine. Finish code
changes before preparing: execution rejects implementation drift. The run
directory must be new; a started run cannot be restarted to reset its budget.
Its complete optimizer checkpoints remain available for explicit recovery work.

The isolated [CPU contract requirements](../requirements-prime-rl-contract.txt)
support local native configuration/batching tests. They are not the GPU install.
Use the exact stack and pinned official GPU image recorded in the smoke report
(Prime-RL `9e15490d`, renderer `5d870d8e`, VF 0.4.0, Transformers 5.6.2,
PyTorch 2.13.0+cu130, vLLM 0.31.0), with this repository and reader plugin installed.
The driver verifies native source fingerprints and rejects another stack.
It needs Docker access and the pinned sandbox image already installed.

Provision/download separately; the driver does neither. Assets must contain:

```text
reader-assets/
  base/b968826d9c46dd6066d109eabc6255188de91218/
    # complete Qwen/Qwen3-8B snapshot at that exact revision
  v5/
    adapter_config.json
    adapter_model.safetensors
    # original artifacts/full/checkpoint-50 at V5 revision 27b912a8...
```

Preflight streams every base shard and configuration through the pinned public
Hub Git/LFS identity catalog. V5 tensor SHA and adapter-config Git identity must
also match. A revision-shaped directory name alone is insufficient.
The catalog's source URLs are in
[prime-base-files.json](../examples/verifiers-port/prime-base-files.json).
Two updates require at least 104 GiB **free** beyond installed assets, because
the initial and per-update native optimizer-resumable checkpoints are retained.
An inference-only Hub adapter cannot resume the optimizer.

After separate compute authorization and preparation on that GPU machine:

```bash
python -m src.training.prime_run /absolute/path/to/new-run --execute-local-gpu
```

This command uses already-local hardware. It never starts, stops or deletes a
cloud pod. Generation is loopback-only, with proxy environment and redirects
disabled; response/request bodies are bounded. Each reservation is durably
written before forwarding, includes prompt plus maximum completion tokens, and
is never refunded. Limits span all questions and policy versions. Failure or an
unfinished reservation blocks further requests. The supervisor bounds subprocess
waits by one wall deadline and cleans up owned process groups and recorded Docker
containers on ordinary failures, timeout, Ctrl-C and SIGTERM. SIGKILL/host loss
cannot run cleanup; a separate infrastructure watchdog must handle that case.
These are workload limits, **not a provider dollar cap**. An existing cloud GPU
continues billing after the process exits until its owner stops it.

## Receipts and remaining verification

Each cycle saves native configs, full raw episodes, per-group rewards, policy-load
receipt, candidate/batch hash, sampler/scorer deltas, native metrics, DCP checkpoint,
optimizer digest and portable adapter. Compare the previous `verified.json` with
the next `prepared.json` to check optimizer continuity. `loaded.json` binds the
successful engine load and reported adapter path to the local adapter SHA. It
does not claim remote tensor attestation. `artifact-*.json` explicitly says that
filesystem acknowledgement did not load inference weights. The last export is
not cold-loaded unless another cycle is requested.

`result.json` distinguishes verified updates from attempted updates that might
have modified weights before an error. A missing verification receipt is not
proof that no update occurred. Original V5 and source/reference files are never
written. The first preflight failure still consumes the run directory, making
failure history explicit.

Local tests use authored responses and synthetic probabilities: they exercise
real Docker, native Verifiers, GRPO/TrainSink, packing and step-2 filesystem batch
transport. Separate checks cover concurrent reservations, accumulated budgets,
uncertain outcomes, redirects, oversized responses, process deadline/ownership,
configuration equivalence to the real smoke, and abort before the next update.
They do not execute the GPU checkpoint/scoring/trainer stages. Importing native
checkpoint code in the CPU-only environment currently stops at missing `torchdata`;
GPU dependencies were not mocked into a pass.

The isolated native suite passes **79 tests**. The first concurrent run with the
full repository suite had two failures and 22 fixture errors: Docker's existing
five-second image preflight reported the pinned image unavailable before any
generation. Direct image inspection and the later isolated suite pass; the
failure log is retained in `out/reader-prime-repeatable-20261024/native-tests.txt`.
No sandbox timeout or admission check was loosened. The first full suite passed
2,148 tests / 16 skips. The final full run, including the added optimizer-state
fixture, passes **2,149 tests / 16 skips**, with three existing SWIG warnings,
in 100.33 seconds. The log is `full-tests-final.txt` in the same output folder.
Focused Ruff is clean; all 29 runtime files and frozen RL source/reference hashes
are unchanged. Prior tracked user edits are preserved and the Git index is empty.

Next: a separately authorized two-update GPU run must pass all these gates.
Then choose a training budget and a fixed comparison before opening the locked
evaluation. This implementation, the earlier single update, and reward increases
alone do not prove that fine-tuned Qwen improved.

## Research basis and finite next decision

[Search-R1](https://arxiv.org/abs/2503.09516) studies RL with multi-turn retrieval,
outcome rewards and retrieved-token masking. It supports the broad training
pattern; its results do not establish gains for our task, reward or small pool.
Prime-RL's [official Wiki Search example at our pinned revision](https://github.com/PrimeIntellect-ai/prime-rl/blob/9e15490dcab1d506ce3462e69b8aa5dde45c6b3a/examples/basic/wiki-search/README.md)
demonstrates a Qwen tool-use task, LoRA, baseline/training/post-training evaluation,
and sampling for nonzero advantages. It uses native function calls, an LLM judge
and multiple GPUs; our Docker Python loop, strict reward and sequential single-GPU
handoff are project-specific adaptations. The original
[QASPER paper](https://aclanthology.org/2021.naacl-main.365/) supplies the research-
paper QA setting and answer/evidence annotations, not validation of our adapted
reward. No source proves our current recipe is optimal.

Plan two further experiments for an initial decision, conditional on success:

1. **Integration check:** two updates. Verify that the second batch uses the first
   update's adapter, and optimizer state survives. Passing ends the currently
   identified training-plumbing gate; it is not an answer-quality experiment.
2. **Bounded training pilot with a paired evaluation:** compare unchanged V5 with
   one RL candidate on the same paper-disjoint questions, prompt, corpus and
   decoding settings. Assess source-supported answers and unsupported claims,
   alongside execution failures, latency and cost. Fix the budget, update count,
   reference-review procedure and numerical success rule before launch. These
   choices are not finalized yet. Twelve locked evaluation papers provide a
   small pilot signal, not a broad benchmark or production-quality claim.

These are two planned experiments, not a guarantee that two runs produce an
improvement, nor a count of individual generation requests or GPU allocations.
Further paid runs require a specific observed failure, proposed correction and
budget. If the pilot gives no convincing gain, retain V5 and report that result;
do not treat increasing training reward as a reason to keep spending.
