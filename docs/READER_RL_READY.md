# V2 reader RL preparation — October 22, 2026

> **Historical preparation report — custom training path withdrawn.** The user
> subsequently chose the smaller Verifiers path and requested unused code removal.
> The custom learner, optimizer, generation code and CLI described below have been
> removed from the working tree. This report and its test outputs are historical,
> not current training readiness. See [the active Verifiers report](VERIFIERS_PORT.md).
> The prior source is recoverable locally in
> `out/reader-simplification-20261022/archive/`; the old GPU plan is not executable.

The user authorized completing the work required for RL readiness, and explicitly
forbade commits until requested. This pass leaves all changes uncommitted.
Prior reports, the old held export, protected optimizer/reward files, the frozen
V2 runtime and original user modifications remain unchanged.

## Hypotheses and gates before new experiments

1. The exact production policy formatter can drive local HF generation while the
   unchanged QwenInvestigator executes actions only in Docker. Real source
   observations, retries, terminal conversion and clipping should match saved
   fixed-action controls exactly (except UUIDs). Preserve original generated token
   IDs and sampler log-probabilities, including cleaned-away text and retry calls.
   No loss on observations or past turns; no scoring-context truncation.
2. A narrow short-fact curriculum with source-reviewed aliases and support keys
   can provide verifiable reference rewards without pretending to grade general
   synthesis or gap-query usefulness. Use official QASPER train only, exclude all
   previous evaluation/validation papers and exposed reward diagnostics, disclose
   any previous training-pool reuse, and lock paper-disjoint reward-validation
   and future model-evaluation roles before model outcomes. Dataset size is a
   bounded pilot pool, not a claim about what is sufficient to generalize.
3. Sampling and scoring distributions must match: fresh generation settings,
   positive temperature, top-p 1, top-k 0, no repetition penalties, thinking off,
   dropout off. Capture sampled-token probabilities from actual generation scores;
   check them against uncached teacher-forced scoring before updates. Complete
   groups only; infrastructure/budget failures cannot become zero-reward examples.
4. Centered GRPO is the initial reproducible baseline. Reject mismatched outer
   batch lengths, mixed questions, missing/invalid token probabilities, nonfinite
   rewards/gradients, model/reference drift or incomplete groups before an update.
   Uniform groups do not justify training. A real smoke must show policy change,
   unchanged reference and byte-identical adapter weights after save/reload.
5. Report local engineering readiness, source-reviewed reward validation, real
   Qwen exploration signal and GPU update proof separately. Local scripted replay
   and tiny CPU-model gradient tests never establish Qwen improvement or real GPU
   readiness. A future model promotion needs matched supported-answer evaluation.

All local work is authorized. No commit, paid teacher call or GPU launch occurs
without a concrete bounded configuration and budget. A budget question is pending
while local work continues. The reserved research_reader_v1 pack stays closed.

## What is implemented

The new `src/training/` path connects the frozen V2 investigator to a Hugging Face
learner. The local controller executes model-authored Python in the existing
network-disabled Docker sandbox. A separate GPU process holds Qwen3-8B, the V5
policy adapter, a frozen reference adapter, and the optimizer. Only authenticated
loopback RPC over an SSH tunnel connects the two; the GPU learner does not execute
document code. This avoids requiring Docker inside a training pod.

Each sampled action retains its original context/token IDs and actual sampled
log-probabilities. Cleanup of a model response never changes its training target.
Only generated action tokens contribute to the loss. The learner owns those
tokens, rejects stale revisions and incomplete groups, checks sampler/scorer
agreement before updating, and stops after a failed update. Verified checkpoints
are saved, reloaded, hashed and downloaded to the controller; an earlier verified
checkpoint remains retrievable after a later failure.

The active MuSiQue reward, old training scripts, frozen runtime and V1 prompt
are unchanged. The two existing files changed by this pass extend the historical
data inventory to `data/rl`, case/context identifiers, source review records and
conservative question-to-paper reservations. Legacy inventories fail closed.

## Data and objective

`data/rl/reader-facts-v1/` contains 82 official QASPER training questions on
distinct papers, eight separate official-validation papers for reward-contract
checks, and 12 additional official-validation papers locked for future model
evaluation. Source revision, snapshots, exact questions, role assignments,
references and selection history are hash-bound. The manifest discloses previous
training-pool reuse. These are locally adapted QASPER tasks, not a new dataset or
an official QASPER leaderboard evaluation.

An implementing-assistant source review rejected 28 candidate labels. A subsequent
conservative historical question-to-paper join excluded 11 more training papers;
their prior roles include training curation and unknown-purpose artifacts, so
these are not all demonstrated evaluation leaks. No Qwen model used the draft
93-question pool. Original and canonical question IDs are both resolved to papers.

The question adds a short-fact answer instruction, and passage windows use QASPER
annotations. Reward requires a verified source inspection, a supported citation
and membership in a finite set of reviewed answer variants. It does not evaluate
retrieval quality, general synthesis, useful gap reports or arbitrary paraphrases.
Some correct unlisted answers or shorter valid citations can receive zero.
Review is model-assisted, not independent human review. The 12 model-evaluation
references are locked and have not been manually reviewed or used for tuning.

## Actual local results

| Check | Result | What it establishes |
| --- | --- | --- |
| Full suite in isolated pinned environment | 2,159 passed; 25 warnings; no skips | Local regression coverage |
| Focused RL/reward/inventory suite | 124 passed | New learning path and failure checks |
| Original saved V5 actions replayed in Docker | 3/3 packets match except UUID; all nine policy messages match; six inspections | Formatter and sandbox compatibility |
| Authored Docker controls | 101/101 completed; eligible current subset 90/90 scores one, with 180 inspections and 270 formatter turns | The adapted reference tasks are mechanically executable |
| Reward-contract controls | 84/84 expected outcomes on eight papers: 28 positives, 56 negatives | Finite-reference contract, not blind semantic calibration |
| Tiny CPU GPT2 and Qwen3 models with LoRA | Sampling/scoring agreement, policy update, frozen reference, save/reload and RPC download pass | Optimizer plumbing on actual tiny models |
| Actual Qwen3-8B GPU calls or updates in this pass | Zero | GPU readiness remains unverified |

Scripted replay does not measure how Qwen would react to new feedback. Authored
controls have no sampled probabilities and cannot enter an optimizer update. The
reward positives are reviewed reference variants from the same implementing
assistant, not an independent answer-quality assessment. None of these results
shows that fine-tuned Qwen improved.

Saved evidence is indexed in `examples/reader-rl-ready/`. Full Docker receipts and
earlier draft attempts remain in `out/reader-rl-ready-20261022/`. The original 2,061
files were hashed before this pass so the final preservation report can distinguish
the two intended inventory changes from the user's existing modifications.

## Frozen first GPU experiment

The first run starts from the pinned V5 adapter on the pinned Qwen3-8B base. The
historical baseline identity records greedy decoding; the experiment explicitly
overrides it with temperature 1, top-p 1, top-k 0, thinking off and seed 42.
Sixteen named training questions each receive four rollouts, each with at most
four investigator steps. The hard generation budget is 256 requests, 2,097,152
reserved input-plus-output tokens, 262,144 output tokens and 1,800 process seconds.
These are resource ceilings, not a dollar estimate.

Training proceeds only with at least four complete groups, 90% valid protocol
terminals, 25% nonuniform groups and four positive episodes. The first four
nonuniform complete groups are eligible for one centered-GRPO update at learning
rate 1e-6, KL 0.001 and clipping 0.2. Uniform or invalid probes stop without an
update. Before any update, the GPU must meet the pinned sampler/scorer tolerance.
Policy-only change, unchanged reference and exact adapter reload must pass.
An updated checkpoint still needs matched source-supported evaluation before use.

The real 8B CUDA/NF4 memory, numerical parity, rollout signal and update are
pending. No paid calls, Runpod resources or persistent volumes were created.
The pending budget choice must be resolved before provisioning. Deployment must
verify a provider-side termination deadline and a worst-case resource quote
including storage; the learner watchdog alone does not stop cloud billing.

## Running the prepared path

Use an isolated environment with the versions in `requirements-reader-rl.txt` and
the appropriate CUDA PyTorch already supplied by a verified training image. The
local verification environment inherits CPU PyTorch 2.10.0 without changing the
default Python packages. It does not validate the prospective GPU stack.

The following commands use the saved local receipts. `serve` is an outline for
an authorized, bounded GPU deployment with the pinned weights already downloaded;
the authorization receipt must bind the plan, resource, quote and termination
read-back. Both machines need the repository and identical dataset/code hashes.
The local controller also needs the pinned Docker image and cached tokenizer.

```sh
python scripts/reader_rl.py validate-references \
  --bundle data/rl/reader-facts-v1 --output out/new-reference-check

python scripts/reader_rl.py freeze-plan \
  --bundle data/rl/reader-facts-v1 \
  --validation out/new-reference-check \
  --parity out/reader-rl-ready-20261022/runtime-parity \
  --output out/new-probe-plan.json

# Set the same private ENVOY_READER_RL_SECRET in both environments.
# GPU host, after a verified deployment and budget authorization:
python scripts/reader_rl.py serve --plan out/new-probe-plan.json \
  --bundle data/rl/reader-facts-v1 --authorization out/paid-authorization.json \
  --adapter /path/to/pinned-v5-adapter --output out/gpu-learner --enable-updates

# Local machine: forward 8008 to the learner over the verified SSH connection.
python scripts/reader_rl.py probe --plan out/new-probe-plan.json \
  --bundle data/rl/reader-facts-v1 --output out/gpu-probe --update-after-signal
```

The saved plan and receipts in `examples/reader-rl-ready/` remain the auditable
record of this preparation. New runs require fresh output directories; source or
data changes invalidate the old plan. Retrieve and verify artifacts before the
provider termination deadline, then remove only the resources created for that run.
