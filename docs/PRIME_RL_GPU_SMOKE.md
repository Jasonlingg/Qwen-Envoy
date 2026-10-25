# Native Prime-RL GPU smoke — October 23, 2026

The bounded Runpod smoke passed: real V5 generation → frozen Docker investigator
→ native Verifiers/GRPO batch → one native Prime-RL update → fresh native checkpoint
reload → portable adapter loaded into a cold vLLM process. No model-quality
improvement has been measured. No Git commit was created.

The verified adapter is saved privately at
[jasonlingg/qwen-envoy-v5-reader-rl-smoke-20261023](https://huggingface.co/jasonlingg/qwen-envoy-v5-reader-rl-smoke-20261023),
Hub commit `ef297bcc832dc949483fb1c99f83c7f2f3a4e89d`. Downloading that exact commit
reproduced adapter SHA-256
`7b83c2b4a5bf3eff1edf79dbc8bf1fe4a692ec1692609296512cbf154dc6d384`.
The original V5 repository was not modified.

## Actual results

| Check | Result |
| --- | --- |
| Real training-role questions / episodes | 1 / 4 |
| Complete group rewards | `[0, 0, 1, 0]` |
| Successful generation calls / all dispatched calls | 12 / 19 |
| Earlier discarded infrastructure calls | 6 packed-response parser failures + 1 transport timeout |
| Native training branches / generated loss tokens | 12 / 553 |
| Sampler/scorer absolute log-probability delta | mean 0.00237037; maximum 0.18167973 |
| Preset scoring gates | mean ≤0.03 and maximum ≤0.25; passed without adjustment |
| Native gradient updates | Exactly 1 |
| Native update time / peak VRAM | 23.334 seconds / 35.342 GiB |
| Loss / gradient norm | 0.00159714 / 0.09608159; finite |
| Adapter matrices changed / reloaded exactly | 504 / 504 |
| Trainer startup policy equals initial V5 | Passed |
| Original V5 file SHA unchanged | Passed |
| Portable adapter cold-loaded in vLLM | Passed; no post-update generation evaluation |
| Test-resource cleanup | Pod and temporary volume each freshly confirmed 404 |

The four answers concern the same QASPER training-role paper. The positive sample
matched the reviewed finite answer/support key. Zero-reward reasons were an
unlisted answer variant, an additional citation without reviewed support, and
support lost after quote clipping. These rewards are not a 25% semantic-accuracy
measurement. They can reject a correct paraphrase, and quote provenance alone is
not semantic entailment. Reference review was implementing-assistant/model-assisted;
independent human review is null.

## Experiment and reproducibility

Hypothesis: pinned V5 can produce a complete mixed-reward group whose sampled-token
probabilities agree with native teacher-forced scoring, then survive one finite
native update and exact reload. Every gate was required; incomplete or failed
rollouts were discarded, never converted into zero-reward training examples.
No reward, source scope, prompt, sandbox or scoring tolerance was changed.

The first four named training-role questions were fixed before generation. The
first question produced a usable group, so the other three were not sampled:
`qasper_train_95bb3ea4ebc3f2174846e8d422abc076e1407d6a--rl-short-fact`.
The named 82-question training pool, source-reviewed keys, QASPER revision
`13b496d2a5359329b110e3419628de3cf791843b`, and paper-disjoint locked roles are
unchanged. No locked future evaluation outcomes were obtained.

- Base: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`.
- V5: `jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5`, revision
  `27b912a863ff914ad45baa03624d8911dc1e17fb`, `artifacts/full/checkpoint-50`.
- Prime-RL: `9e15490dcab1d506ce3462e69b8aa5dde45c6b3a` (0.10.0); seven executing
  native model/trainer/transport/server source files matched the pinned checkout.
- Renderer: its pinned upstream submodule revision
  `5d870d8eaa65114f5bef8a4cffc146fbcb418ec4` (0.1.13.dev5).
- Verifiers 0.4.0, Transformers 5.6.2, PyTorch 2.13.0+cu130, vLLM 0.31.0.
- One A100 SXM4 80GB, US-WA-1. Inference was stopped before training and reload.
- Seed 42; successful rollout seeds 49–52 after seven discarded attempts.
  Temperature/top-p 1, top-k disabled, min-p 0, thinking false, max 1,024 tokens
  per action, four investigator steps, native 8,192-token sequence limit.
- Native centered GRPO advantages, token aggregation, native PPO clipped surrogate
  0.8–1.2, AdamW LR 1e-6, zero weight decay, constant scheduler, no reference-KL
  term for this single-update smoke. Native optimization/reduction float32 defaults
  were preserved. LoRA rank 4, alpha 8, dropout 0; original seven projection targets.
- Model-authored Python ran only in the pinned network-disabled Docker sandbox.
  The trusted framework/controller ran on the host. All eleven attempted episode
  containers were removed, and all 29 frozen runtime files remain unchanged.

The official Prime-RL image was pinned to
`ghcr.io/primeintellect-ai/prime-rl@sha256:bd810178d034ea134663be6d21b6269af02e88b4a54517d481a1874b139af52f`.
Its bundled development Verifiers package was aligned to the locally tested 0.4.0.
The released renderer 0.1.12 could not decode the native endpoint's packed float32
probabilities; using Prime-RL's exact renderer submodule fixed this without a
custom decoder. The initial failed attempts are archived and counted against
the same request ceiling. `runtime-final.json` records the executed stack; the
model-download receipt predates this dependency alignment.

Thin smoke glue imported V5 into native adapter parameters, used native DCP
checkpointing, acknowledged filesystem adapter exports while inference was stopped,
and invoked the unchanged native trainer. This single-GPU artifact acknowledgement
is not a live inference weight update. The separate cold-load check verifies the
portable export loads into vLLM. No custom optimizer or training loop was restored.

Portable packaging adds the standard `base_model.model.` key prefix and a public,
pinned base-model identity. Local comparisons proved every tensor value unchanged
from the native export. The Hub contains the inference adapter, not the full
optimizer-resumable 31GiB DCP checkpoint; that checkpoint was reloaded before the
temporary volume was removed. Raw episodes and original traces remain untrainable;
the complete native candidate batch was admitted only for this bounded smoke.
The default production training block remains unchanged.

## Budget and cleanup

Authorized limits were $20, 90 minutes, 64 generation dispatches, 524,288 reserved
tokens, and one gradient update. Actual dispatch count was 19, with no extra model
request for the final cold-load check. Independent local and on-pod deadline guards
were active; the guard credential was passed over SSH stdin, held in memory, and
its temporary file immediately removed. No Hub write credential was sent to the GPU.

The single GPU was quoted and read back at **$1.79/hour**. Creation through confirmed
deletion took **41.06 minutes**, giving a conservative **$1.225 GPU estimate** that
includes image pulling. Temporary storage adds about a cent for this attempt.
Runpod's billing API had not posted an itemized charge; this is not a final invoice.
Earlier cancelled/failed placement attempts created no GPU, and their temporary
volumes were removed; their brief storage charges were not independently itemized.

Pod `osmoo0a2gzjqd9` and volume `q3egk3botx` were deleted and each confirmed absent
by a fresh API read. The local watchdog independently confirmed the pod's absence.
No smoke-test compute or storage is left billing.

## Verification and limits

- Full repository suite: **2,109 passed, 16 skipped**, three existing SWIG warnings.
- Native integration suite with matching renderer: **56 passed**.
- Final renderer/token regression rerun: **13 passed**.
- Ruff, compile checks, Git whitespace check, artifact hashes, local tensor
  comparisons, Hub download hash and frozen-runtime/container cleanup checks passed.

[Saved receipts and exact configs](../examples/verifiers-port/prime-rl-smoke/README.md)
include actual trainer metrics. Full raw episodes, engine responses, packed batch,
scripts and logs are local in `out/prime-rl-smoke-20261023/`.

This proves the bounded native training plumbing works on real V5 rollouts.
It does not prove RL improves Qwen, validate general research synthesis, or establish
that the 82-question pilot pool is sufficient. Longer training, a reviewed semantic
reward/evaluation decision, and a separately authorized held-out comparison remain
separate work. The MCP's research quality is not established by this smoke.
