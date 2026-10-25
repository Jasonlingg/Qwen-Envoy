# Verifiers compatibility trial — October 22, 2026

> **Historical report; current status updated October 24.** The single-update
> [native GPU smoke](PRIME_RL_GPU_SMOKE.md) subsequently passed. The
> [repeatable sequential launcher](PRIME_RL_REPEATABLE_RUN.md) is implemented and
> locally tested; consecutive GPU updates and model-quality improvement remain
> unverified. Counts and statements about native training below describe the
> earlier compatibility stage, not the current implementation total. Generic
> Verifiers training remains blocked outside the bounded driver.

## Current path after simplification

The user requested removing unused implementation instead of maintaining both
backends. After native prime-rl contract verification, the active addition is
**497 implementation lines plus 847 test lines: 1,344 Python lines total**, down
from 3,540 before cleanup (62% less). The latest check adds 45 runtime lines and
321 test lines; it adds no custom trainer or optimizer.
Counts include the installable plugin and exclude documentation, JSON data and
ignored local backups. Small existing data-inventory edits are separate.

Removed the unused custom GRPO optimizer, learner/RPC server, training CLI,
paid-run plan builder, Hugging Face generation/probability capture and one-time
dataset construction code, plus their implementation-specific tests/dependency
file. Retained the reviewed frozen dataset, its integrity/reward checks, the
80-line Docker runner, and the small Verifiers adapter. The runner has no
PyTorch or Transformers imports. The three saved Docker controls and all 84
saved reward-contract controls still have regression coverage.

The custom GPU plan is withdrawn. **Native prime-rl training remains unverified;
no replacement optimizer is claimed to be running.** Current traces remain
evaluation-only. Earlier local tests and source snapshots are historical evidence,
with source backups under `out/reader-simplification-20261022/archive/`.

Post-cleanup verification: **18 focused tests pass** in a fresh VF 0.4 environment
without PyTorch or PEFT installed; Transformers is a transitive framework
dependency but is not imported by this adapter. The existing full suite passes
**2,122 tests, with one optional VF-module skip and three existing warnings**.
The lower count reflects removal of tests for the deleted custom trainer, not
new failures. All 84 saved reward controls and the Docker replay checks pass.
The existing two data-inventory files retain 25 net added lines separately from
the then-785 lines in new files: 810 net Python lines after cleanup, or 1,003
including the subsequent 193-line token-contract test module. No runtime or
trainer implementation was added by that module.

## Original compatibility trial

The user authorized trying a Verifiers adapter while retaining all existing
uncommitted work and making no commits. This trial adds an optional, installable
taskset/harness/environment plugin. It does not replace the frozen investigator,
start a GPU, use paid inference, or establish model improvement. The subsequent
cleanup removed the unused custom backend as described above.

## Hypothesis and decision rule

Hypothesis: Verifiers can own interception, trace construction and task scoring
while the existing investigator owns its exact Python interaction loop and pinned
Docker sandbox. Expected signal: identical saved-control model requests and final
packets, including actual source inspections, with explicit failure attribution.
Adopt the environment adapter only if those checks pass without changing the
prompt, tools, reward, corpus or investigator. A native training port additionally
needs exact renderer/token/probability parity and a bounded real V5 GPU update;
this trial cannot satisfy that second gate.

## What changed

- `src/training/verifiers_adapter.py`: 0.4-specific Task, Taskset, Harness and Env
  classes. Calls the existing `collect()` instead of implementing another agent
  loop. The native VF interception server captures every model request and raw
  response. Task scoring calls the unchanged narrow short-fact reward.
- `environments/envoy_reader/`: a small installable `envoy-reader` package that
  exposes the `envoy_reader` plugin through Verifiers' discovery interface.
- `requirements-verifiers.txt`: optional dependency/plugin installation.
- `tests/test_verifiers_reader.py`: actual native Agent/interception integration,
  Docker controls and focused lifecycle/failure tests.
- `examples/verifiers-port/`: public authored control fixtures and compact saved
  evidence. No sampled training probabilities are invented.

Only the 82 prepared official-training questions are exposed by the taskset.
Private reference keys stay outside the model's messages. The adapter rejects a
changed task/system prompt and a changed source manifest. See the existing
`data/rl/reader-facts-v1/README.md` for QASPER attribution, annotation-assisted
scopes, historical training-pool reuse and reward limitations. This trial neither
reviews nor evaluates the 12 locked future model-evaluation papers.

The VF subprocess runtime runs only trusted orchestration. `collect()` still
executes all model-authored Python in the existing Docker sandbox, with the pinned
image, network disabled and read-only corpus mount. These boundaries are checked
in the saved sandbox receipts. The plugin defaults to that local controller,
rather than Verifiers' generic cloud runtime. Its model endpoint must be explicit
HTTP loopback. No Prime API key or cloud sandbox is needed for the replay trial.

The forwarding adapter allows at most 12 requests per investigator, uses a
30-second HTTP timeout, rejects redirects and bounds returned response bodies to
2 MiB. These are compatibility limits, not a GPU billing cap. On cancellation it
waits for the worker's bounded investigator and finally-cleanup; cancelling a
Python worker thread alone would not stop its Docker work.

## Results

- **13 adapter tests passed** on real Verifiers 0.4.0 in a clean Python 3.11
  environment. The native plugin loaders resolve the taskset, harness and Env.
- **Three authored saved controls replayed exactly**: nine complete production
  payloads match, and packets match except their run UUIDs. Each control performs
  two actual Docker source inspections.
- **One additional fenced-Python variant passes**: raw fences remain in the model
  history/native trace; the cleaned Python executes in Docker with the same final
  packet. The first test draft incorrectly expected stripped conversation history;
  the test was corrected after inspecting the unchanged production policy.
- **A provider failure after one successful inspection** yields an unsuccessful
  trace, no assigned numeric task reward and retained earlier observations. The
  container is cleaned up, and neither provider nor rollout retries are hidden.
- Prompt substitution, unapproved interception endpoints, unsupported training
  clients and cancellation cleanup have explicit regression coverage.
- **Existing full suite: 2,159 passed, one optional-module skip, 25 warnings.** The
  skip is this new module in the old VF 0.1.12/pinned learner environment; its
  13 tests run separately in the 0.4 environment. Existing VF 0.1.12 wrapper tests
  continue to pass. These are deliberately separate dependency stacks.
- **75 focused source/reward/investigator tests passed** in the clean VF 0.4
  environment as well, checking the shared code under its new dependency stack.

All control actions are authored or replayed, not freshly sampled Qwen decisions.
There were zero paid/model inference calls, zero GPU launches and zero optimizer
updates. Fixed-action replay cannot measure how Qwen reacts to new feedback.
Finite-reference reward checks do not establish general semantic answer quality.

## Reproduce locally

Start from this repository checkout with Docker running and the pinned sandbox
image already built. The public control fixtures are included; the trial does not
depend on private `out/` receipts from the earlier RL preparation.

```sh
python3.11 -m venv out/verifiers-clean
out/verifiers-clean/bin/python -m pip install -e . -r requirements-verifiers.txt \
  pytest==8.4.2 pytest-asyncio==1.3.0
PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  out/verifiers-clean/bin/python -m pytest -q \
  tests/test_verifiers_reader.py tests/test_reader_rl_data.py \
  --basetemp out/verifiers-replay-receipts
```

The pinned sandbox image is
`sha256:79bd8bf84bd4b7f2a32a0596cb1fd90164716c1cfca879f78829411193646382`.
This is a local image content ID, not a portable registry pull reference. Rebuild
the frozen sandbox and validate its identity before trusting replay on a new host;
a different build will fail closed. The current source checks all 29 frozen
runtime files. The tested host is Apple M3/macOS with CPU PyTorch, not a validated
Linux/CUDA environment. Dependency snapshots are in the evidence bundle.

The first attempted environment inherited system packages and hit an existing
`connect-python`/`connectrpc` namespace conflict. The clean environment avoids
that conflict. Global packages and the existing RL verification environment were
not modified. Do not install VF 0.4 over the existing 0.1.12 environment: its root
API was replaced by `verifiers.v1`. The prototype pins a published release, not
moving upstream main or a development build. One dotted-module discovery attempt
also failed; the adapter now uses Verifiers' normal installed top-level plugin ID.

## Remaining training gate

This adapter defaults to VF's EvalClient, marks every trace `trainable=False`,
and rejects TrainClient unless the explicit local `token_contract_only` check is
enabled. That check also leaves every original trace untrainable. EvalClient provides no exact training
token IDs or sampled probabilities. No optimizer should consume these replay
traces. In particular, VF's convenience aggregate `trace.reward` can be zero when
named rewards are absent: failure consumers must check `trace.ok` and reward
availability rather than treating that aggregate as a task-failure label.

The earlier token-contract check below covers the native client and graph in
isolation. The [native prime-rl check](PRIME_RL_LOCAL_CHECK.md) now verifies the
actual investigator/interception path through native group scoring, batch packing
and trainer-side transport decoding, using authored fixtures. Remaining GPU gates
include exact V5 warm start, reference policy, resource fit, one bounded update,
checkpoint reload and retrieval through prime-rl. Its default algorithms/settings
must not be silently presented as the already frozen custom centered-GRPO recipe.
That is a new experiment plan. The real GPU trial still needs a hard approved
budget and verified provider termination; no GPU launch is included here.

The original trial retained the custom learner for comparison. The user later
authorized removing that unused path; only a local backup remains. The existing
frozen production runtime, historical committed trainers and user modifications
remain intact. Nothing was staged or committed.

## Native token preflight — October 23, 2026

Hypothesis: the native TrainClient can preserve the frozen V5 prompts and raw
generated tokens while masking all observations and template tokens out of the
loss. Expected signal: exact token equality, aligned finite sampling log-probabilities,
and no sampled positions outside completions. Proceed to the next integration
gate only if every check passes; keep the production training guard in place.

The optimized `Qwen3RendererConfig(enable_thinking=False)` matches **3/9** saved
prefixes: first turns only. It retains empty historical thinking blocks which
the frozen Hugging Face template removes. Use the existing
`DefaultRendererConfig(enable_thinking=False)` instead: **9/9** prefixes match,
including hashes checked against the earlier Transformers 4.57.6 token receipts.
The optional requirements now pin renderers 0.1.12 and Transformers 4.57.6 as well
as Verifiers 0.4.0. Pin `renderer_model_name` to the locally cached tokenizer
snapshot at Qwen/Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`; an unpinned model alias is insufficient.

`tests/test_verifiers_tokens.py` drives the real native TrainClient, renderer and
trace graph through an in-memory HTTP transport. A one-layer, random-initialized
Qwen3 model on CPU samples four tokens per turn for three turns, with actual
temperature-1/top-p-1 sampling log-probabilities. This checks transport and mask
alignment, not V5 capabilities. It loads no pretrained model weights, executes no
generated text and runs no optimizer. Separate synthetic fixtures check raw Python
fences and rejection of missing, non-finite, misidentified and sentinel
log-probabilities; their fake probabilities are never training data.

Because the frozen template changes historical assistant prefixes, the native
graph produces three separate branches. Each preserves its exact inference prompt
and completion, with only its own raw completion marked sampled. Historical
actions, source observations, simulated error feedback and template scaffolding
are masked out. Raw token IDs are retained directly from the engine response;
decoded/re-tokenized text is not substituted for them. This avoids writing a
custom renderer or token/loss-mask implementation.

This is **not** an end-to-end prime-rl training run, actual vLLM validation, or a
real V5 rollout. Error feedback in the earlier token test is simulated, while the
later native prime-rl check uses actual Docker/controller feedback. The adapter
still marks all original traces untrainable and rejects TrainClient by default.
Native branch export and advantage/reward attribution now pass local authored
controls; reference-policy scoring, exact V5 LoRA loading, updates and reload
remain unverified. No paid call, GPU launch or new model
improvement is established. The separate GPU experiment requires a hard approved
budget and verified shutdown first.

Reproduce token checks in the optional environment with CPU PyTorch 2.10.0 added
(the evaluation adapter itself still needs no PyTorch):

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q \
  tests/test_verifiers_tokens.py
```

The pinned tokenizer must already be cached. Missing optional dependencies or
tokenizer files skip these tests; a skipped check does not pass the training gate.
Results and exact prefix hashes are in
[`token-preflight.json`](../examples/verifiers-port/token-preflight.json).
Detailed local receipts, including intermediate fixture failures, remain under
`out/verifiers-token-preflight-20261022/`.

Current verification: **27 focused tests passed** (nine token-contract tests and
18 existing adapter/data tests). A no-PyTorch environment also passed the three
saved-prefix cases. Shared runtime checks passed 55 tests; both opt-in Docker
cases skipped there passed in a subsequent five-test Docker selection. The full
legacy suite passed **2,109 tests**, with 15 optional skips and three existing SWIG
warnings. Thirteen skips require the Docker-image environment variable, and the
other two are the VF 0.4-only modules tested separately above. A subsequent
**14-test Docker selection passed**, covering all 13 Docker skips plus one already
passing companion case. No required check remains skipped across these runs.
