# Native prime-rl local integration check — October 23, 2026

The native data handoff passes locally: frozen investigator → actual Docker →
Verifiers Env/interception/TrainClient → native GRPO advantages → TrainSink →
batch packing → filesystem transport → trainer-side batch decoding.
This is a contract check with authored actions and synthetic engine probabilities,
not a V5 rollout, actual vLLM engine, optimizer update or model-improvement result.
All original local fixture episodes remain untrainable. No paid call or GPU launch
was used for this local stage. The subsequent user-authorized GPU smoke is recorded
separately under `out/prime-rl-smoke-20261023/`; no commit has been made.

## Hypothesis and decision rule

Hypothesis: the existing Docker investigator can supply native prime-rl samples
without replacing its Python loop, tools, prompts, source scopes or reward.
Expected signal: exact V5 prefixes, raw response tokens/probabilities, action-only
loss masks, correct group advantages and unchanged batch transport. Reject
incomplete, failed, mismatched or non-finite groups. Pass this local stage only if
all checks pass; retain the production training block until real GPU proof.

## Reproducible stack

- prime-rl **0.10.0**, Git revision
  `9e15490dcab1d506ce3462e69b8aa5dde45c6b3a`; matching workspace configs package.
- Python 3.12.6, Verifiers 0.4.0, renderers revision
  `5d870d8eaa65114f5bef8a4cffc146fbcb418ec4` (0.1.13.dev5), Transformers **5.6.2**,
  CPU PyTorch 2.10.0; Apple M3/macOS. This is separate from the original VF 0.4
  environment with Transformers 4.57.6 and the legacy VF 0.1.12 environment.
- Cached Qwen/Qwen3-8B tokenizer revision
  `b968826d9c46dd6066d109eabc6255188de91218`; no pretrained weights loaded.
- `DefaultRendererConfig(enable_thinking=False)`. The prior optimized-renderer
  mismatch still reproduces: fallback **9/9** exact frozen prefixes; optimized
  Qwen3 renderer **3/9**. No new renderer implementation.
- Actual controller payloads remain unchanged. Native sampling context explicitly
  disables top-k and min-p truncation; temperature/top-p are 1, max action tokens
  1,024, thinking disabled, no retries. The earlier tiny random CPU sampler test
  also passes under this new stack; it samples 12 actual tokens, never executes them.
- All 29 production runtime files, V1 prompt bytes, pinned Docker image and frozen
  QASPER short-fact data/role/reference files remain unchanged. The test uses one
  already-public training-role control, not the locked future evaluation questions.

## Changes

`ReaderHarnessConfig.token_contract_only` is an explicit local validation option.
It allows the native TrainClient with the pinned local tokenizer and fallback
renderer, while keeping `trace.agent.trainable=False`, `training_eligible=False`
and `weights_verified=False`. The default still rejects TrainClient. The trusted
controller runs locally; permitted document actions execute in the pinned Docker
sandbox, with networking disabled and a read-only corpus. Unsupported Python is
rejected by the unchanged controller before execution.

`complete_reader_group()` is a small admission callback for native TrainSink.
It requires the expected group size, identical task/group/run/policy identities,
healthy episodes and a finite named short-fact reward. Compose it with the native
train source's `on_result` callback in a future training driver. The generic
prime-rl CLI does not automatically install this project-specific callback.

The native GRPO scorer filters failed episodes before calculating its baseline;
the admission callback prevents surviving members of a failed group from being
exported. No custom optimizer, loss, learner RPC or trainer was added.
This pass adds **45 runtime lines and 321 test lines**. Total current new Python
files: 497 implementation + 847 tests = 1,344 lines, versus 3,540 before cleanup.

## Actual checks

- **Six full native Env/Docker fixtures:** correct answer, wrong answer, fenced
  Python, recovery after a controller-rejected action, provider failure and invalid
  token metadata. Correct/fenced/recovery rewards are 1; wrong answer is 0. Both
  infrastructure failures retain the first inspected source, receive no numeric
  task reward and make no hidden retry. These are authored controls, not Qwen decisions.
- Every actual production request matches the native engine's prompt token IDs;
  unchanged good/bad control payload fields equal the saved V5 requests.
  Raw fenced text and EOS token IDs remain in captured completions.
  Source observations, rejected-action feedback and historical/template tokens
  stay outside the loss mask.
- The two-episode mixed group yields **six samples, three packed microbatches,
  250 action tokens**, with advantages +0.5/-0.5 and exactly aligned masks,
  probabilities and temperature. Native filesystem sender/receiver and msgpack
  round-trip preserve the packed batch exactly.
- Original untrainable episodes, incomplete/failed groups and uniform-reward
  groups emit no training payload. Separate mutation checks reject wrong task,
  group, policy, missing run/reward, non-finite reward and untrainable members.
- **52 focused tests pass** in the native prime-rl environment, including all
  previous Verifiers/token/data checks. **2,122 full-suite tests pass** in the
  legacy environment with real Docker enabled; three optional module skips are
  the modules tested separately in the native environment. Three existing SWIG
  warnings remain. Dependency validation and focused Ruff checks pass.

The low-level `trace_to_samples` compiler assumes eligibility was checked upstream;
it does not itself inspect `trace.agent.trainable`. These checks use TrainSink's
actual eligibility filtering, not direct compiler output as a readiness signal.
Only explicit in-memory copies of authored fixtures are made trainable to exercise
the compiler. Their -1 log-probabilities are synthetic and are never eligible
on-policy samples. No optimizer is created or fed these copies.

## Remaining boundary

Importing the actual trainer loss on this CPU-only Mac fails at the GPU-side
`ring_flash_attn` dependency. This was recorded, not patched or mocked into a
passing trainer. The package's Linux GPU extras require a different CUDA/PyTorch
stack, so native trainer forward/backward, sampler/scorer numeric agreement,
reference-policy identity, real V5 LoRA warm start, memory fit, updates and
save/reload are not established by these local checks. The subsequent authorized
[Linux GPU smoke](PRIME_RL_GPU_SMOKE.md) now records actual V5 generation, the
native update/reload and verified cleanup separately.

The native trainer's default IPO loss is not the historical centered-GRPO recipe.
This check explicitly selects native GRPO **credit assignment** with no length
penalty and token aggregation; it does not execute or select a trainer loss.
The GPU smoke explicitly pins native PPO clipped surrogate and wires the complete
group callback. Its one-update result does not establish a useful training recipe
or validate the default production training path.

## Reproduce

Docker and the existing pinned sandbox image must already be available. Cache the
pinned tokenizer first; tests run with Hugging Face offline and never download weights.

```sh
uv venv --python 3.12 out/prime-contract
uv pip install --python out/prime-contract/bin/python --no-sources \
  -r requirements-prime-rl-contract.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-project --python out/prime-contract/bin/python python -m pytest -q \
  tests/test_prime_rl_reader.py tests/test_verifiers_tokens.py \
  tests/test_verifiers_reader.py tests/test_reader_rl_data.py
```

The initial install failed because the shallow checkout omitted optional git
submodules; `--no-sources` bypasses unrelated workspace sources.
A subsequent direct Git install tried fetching unrelated submodules and failed
without git-lfs. The requirements use immutable GitHub source archives for Prime-RL and configs,
and an exact Git revision for its renderer client. The latter is required to decode
the native GPU endpoint's packed float32 probabilities; the published 0.1.12 client
rejects that response. The four packed-response checks cover valid values, short
arrays, NaN, and the missing-probability sentinel. The updated native suite passes
56 tests. No custom probability decoder was introduced.
Initial fixture assumptions about compiler eligibility and division-by-zero
execution were corrected against native/frozen behavior. All intermediate logs
are retained under `out/prime-rl-local-20261023/`. Public summaries and provenance
hashes are under `examples/verifiers-port/prime-rl-local/`. Review is by the
implementing assistant; independent human review is null.
