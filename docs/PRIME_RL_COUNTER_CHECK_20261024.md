# October 24 corrected-counter GPU check

The corrected two-update Qwen integration did **not complete**. A separate tiny
CUDA checkpoint diagnostic passed on the actual A100/PyTorch stack. These are
different results; this attempt produced no Qwen training update or new adapter.

The user authorized the two-update check with “ok go.” The existing allowance was
$5 estimated and at most two hours of owned-resource lifetime, with 256 generation
requests, 2,097,152 reserved tokens and 1,800 work seconds. The hypothesis remained
that two consecutive native updates could preserve optimizer state and reload
exactly. Any infrastructure, identity, scoring, reload or budget failure stopped
the integration. The ten-update pilot was not launched.

## Actual integration outcome

The fresh local/GPU plans bound the repaired implementation and unchanged numerical
protocol. The exact image manifest, base weights, V5 adapter, seven native source
files, package versions and A100 80GB passed preflight. During inference startup,
the deployment monitor's SSH read of `result.json` exited 255. Its `finally` block
terminated the owned native parent, which wrote its stopped result and cleaned up.
The wrapper did not retain SSH stderr, so the underlying disconnect reason is
unknown. Successful subsequent SSH diagnostics do not establish that reason.

- Qwen generation requests / reserved tokens: **0 / 0**.
- Qwen attempted / verified updates: **0 / 0**.
- No batch, scorer result, exported adapter, ten-update pilot or paired evaluation.
- No locked reference review, promotion, Hugging Face upload or Git commit.

The one actual Qwen update from the earlier failed attempt remains historical;
it is not an update from this attempt. See
[the earlier report](PRIME_RL_INTEGRATION_20261024.md).

## Counter diagnostic

Before cleanup, a tiny CUDA linear model exercised the repaired initializer,
PyTorch `get_state_dict` / distributed-checkpoint save/load / `set_state_dict`,
and two real AdamW updates. It used the same GPU stack as the intended Qwen run:
NVIDIA A100-SXM4-80GB, PyTorch 2.13.0+cu130.

The counter was **0 → 1 → 2**. All three checkpoints reloaded with identical
optimizer fingerprints. At every step, weights matched an independently initialized
ordinary AdamW control exactly. This establishes the narrow CUDA diagnostic,
**not** full Qwen/DTensor/native-Prime-RL checkpoint integration or answer improvement.
The [script and result](../examples/verifiers-port/prime-counter-check-20261024/)
are retained with the stopped-run preflight and result.

## Monitor repair and cleanup

The deployment-only driver now retries the same read-only JSON status request at
most three times, with a 15-second timeout per attempt and 1/2-second backoffs.
Only SSH exit 255 and transport timeouts are retried. Each failure retains bounded
stderr. Remote command failures and malformed JSON still stop immediately.
Generation, training, uploads and mutations are not retried; saved plans, job
nonces and budgets are not reset. Four authored transport tests pass. This repair
has not been tested on another live training run.

The repaired operational driver and tests are under
`out/reader-prime-counter-check-20261024/`; a reviewable patch is in the public
receipts. This is launch tooling, not a new trainer or a reader-harness change.
A future authorized attempt must use fresh plans and this repaired monitor.

The independent watchdog successfully deleted the **live** Pod
`qptbwk1qk67kx5` and temporary 600GB volume `0rdp3k2d1m`. Fresh MCP reads returned
404 for both. This time automatic cleanup worked; older user resources were
untouched. Owned lifetime was about **10.4 minutes**. At the quoted $1.79/hour GPU
rate and conservative $1.90/hour combined bound, the estimate is **$0.33** before
tax. Filtered provider billing returned no posted rows yet, so actual billed spend
is pending, not zero. Standard volume pricing was checked against the
[official storage documentation](https://docs.runpod.io/storage/network-volumes).

Validation: **139 focused native tests passed**; **2,204 full-suite tests passed**,
16 optional skips. The native run had three expected CPU DCP warnings; the full
suite had three existing SWIG warnings. Four separate authored SSH-read tests
passed. Existing source/reward/data/prompt/runtime hashes and user edits were
preserved. The remaining gate is a completed two-update Qwen integration using
the repaired monitor, before projecting or starting the ten-update pilot.
