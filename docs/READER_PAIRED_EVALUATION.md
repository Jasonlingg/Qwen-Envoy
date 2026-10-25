# Paired V5 / reader-RL evaluation runner

> Subsequent October 24 update: the first integration attempt ran one gradient
> update but failed the optimizer-counter gate. Local repairs are tested; the
> corrected GPU run and paired evaluation remain pending. Both temporary resources
> are deleted. See [the actual report](PRIME_RL_INTEGRATION_20261024.md).

October 24, 2026. Local implementation and scripted contract checks only. No new
GPU, paid generation, optimizer update, locked-reference review, upload or commit.
The numerical experiment protocol in `examples/reader-rl-pilot-v1/protocol.json`
remains unchanged. This runner implements its collection and claim-review path;
it does not supply semantic judgments or evidence that RL improved Qwen.

## What is connected

`src/training/reader_evaluation.py` reuses the frozen investigator, Docker sandbox,
Verifiers 0.4 native TrainClient and pinned non-thinking renderer. Here TrainClient
is a token transport: evaluation tasks have no reward function, traces stay
untrainable, and no TrainSink or optimizer is called. The ordinary training
resolver explicitly rejects an evaluation-session task.

A session can only be frozen from the complete predeclared ten-update pilot,
starting at original V5. All ten verified checkpoints, optimizer step counters,
reload/export and sampled-policy continuity receipts must be present. The export
copies both adapters and training receipts with hashes. It can move between the
GPU host and local controller without rewriting absolute paths inside a launch
plan. Source data, protocol, effective prompt, implementation and tokenizer bytes
are checked; a changed checkpoint, reordered training plan or changed seed fails.

The reference export is a separate explicit action, available only after candidate
freeze. Before generation, every reference/source pair needs an attributed review
of in-scope answerability and ambiguity. Unfilled or ambiguous rows stop the run;
no replacement questions. Preparation tests never decode the locked references.

Collection enforces V5 first, then step ten; frozen question order; identical
`42 + question_index` seeds; no retries. Both arms share the existing persistent,
nonrefundable **288-request / 2,359,296-reserved-token / 1,800-second** budget. A
partial arm cannot restart or be overwritten. Infrastructure failure preserves
its record and stops. The CLI handles termination and cleans up only container
IDs recorded by that arm. GPU/resource lifetime requires the separate watchdog.

Verifiers omits task state from JSON. The saved episode therefore includes an
explicit `docker_episode` field containing the original packet for portable
review. Raw episodes, engine payloads/responses, per-question timings and token
reservations remain available. The derived review pack removes arm/model metadata,
randomizes review IDs/order and keeps a separate private arm map. Do not show the
private map or raw identity-bearing records to the reviewer before grading.

Reviewers must inventory material factual claims, judge each against its inspected
citations, and explain requested-fact completeness, numbers, units, conditions and
negation. A real quote establishes provenance. It does not establish support.
Empty, uncertain, conflicting, unattributed or source-edited reviews cannot pass.
Model-assisted review must be labelled; independent human review remains null
unless an actual human supplies it. Raw telemetry and review-visible source fields
are bound at export so grading cannot silently rewrite them.

The reducer applies the original +3 net supported answers / at most one regression,
zero newly unsupported questions, failure-count and token/latency thresholds. It
reports per-question outcomes and nearest-rank p95 latency. Missing billed-cost,
startup and GPU-lifetime receipts makes the result inconclusive. This is still a
12-paper, one-seed pilot, not a significance test. The existing official QASPER
scorer and strict reward remain secondary diagnostics; this command does not
compute or substitute those scores for claim review.

## Local Docker and remote inference

```text
Mac / trusted controller                    GPU machine
  frozen questions + tokenizer                pinned base + frozen adapters
  VF native renderer                          native Prime-RL / vLLM engine
  durable request/token budget -- SSH -->      127.0.0.1:8000
  frozen investigator
          |
          v
  Docker: generated Python
  network disabled; read-only corpus
```

Only `http://127.0.0.1:PORT` is accepted as an engine. HTTP redirects, credentials,
proxy environment settings, public hostnames and URL paths are rejected. Generation and model-metadata response
sizes are bounded. Bind an SSH forward explicitly to loopback, for example:

```sh
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -L 127.0.0.1:18000:127.0.0.1:8000 USER@GPU_HOST
```

This is a command template, not a connection made during preparation. Use an
already authorized host/key and include the provider's SSH port when needed.
No GPU API key belongs in model-authored Python or the Docker container.

On the GPU host the native load command hashes the session's adapter, uses the
native LoRA loader, checks the engine's `envoy` alias/root, and records hardware,
package versions and load time. The controller checks that arm-specific receipt
and alias/root before each question. Root/RPC checks are not tensor-level engine
attestation. Engine startup time and paid lifetime must be recorded separately.
Use the exact smoke inference settings/native config from the completed training
run, a fresh engine for each arm, and the same GPU/package stack for both.

**Training deployment is still a launch gate.** `prime_run` currently runs its
collector, CUDA scoring and trainer on one machine and explicitly requires Docker
there. The earlier one-update smoke used local Docker plus tunneled inference,
but that does not certify consecutive updates with the new launcher. This pass
connects the split placement for evaluation, not remote training orchestration.
A newly rented GPU container has not been shown to support Docker. Before any
rental, establish a Docker-capable GPU layout or explicitly implement the split
training collector. Never replace Docker with generated Python on the host.
The provider lifetime/cost watchdog also remains to be installed and verified.

## Commands for the later authorized experiment

Use the isolated pinned Prime-RL environment. None of these generation commands
was run against a real model in this preparation pass.

1. On the machine with the completed training run:

   ```sh
   python -m src.training.reader_evaluation freeze /path/eval-session --training /path/training-run
   ```

2. Copy the frozen session to the Docker controller. Keep its initial files
   unchanged. Open references there, then complete every reference-review row
   with real attribution **before seeing model outputs**:

   ```sh
   python -m src.training.reader_evaluation open-references /path/eval-session
   ```

3. For each arm, in order `unchanged_v5`, `final_rl_step_10`: start a fresh native
   engine on the GPU using the completed run's pinned inference config. The
   existing `prime_native TRAINING_RUN inference 10` stage starts that engine;
   manage its process and billing with the separately authorized watchdog. On
   the GPU host, with the same frozen session:

   ```sh
   python -m src.training.reader_evaluation load-arm /path/eval-session \
     --arm unchanged_v5 --engine http://127.0.0.1:8000
   ```

   Copy that arm's `backend.json` to the controller. Through the SSH forward:

   ```sh
   python -m src.training.reader_evaluation collect-arm /path/eval-session \
     --arm unchanged_v5 --engine http://127.0.0.1:18000 \
     --tokenizer /path/b968826d9c46dd6066d109eabc6255188de91218 \
     --backend-receipt /path/eval-session/unchanged_v5/backend.json
   ```

   Stop the owned engine, cold-start it for the second arm, and repeat load and
   collection using `final_rl_step_10`. Do not restart a failed/partial arm.

4. Export review, complete its claim-level judgments with actual attribution,
   and apply the gate:

   ```sh
   python -m src.training.reader_evaluation export-review /path/eval-session
   python -m src.training.reader_evaluation summarize /path/eval-session
   ```

   Supply `lifecycle.json` with session SHA, `cost_kind: provider_billed`, actual
   spend, total GPU-lifetime seconds and `startup_seconds_by_arm` for both arms;
   `billing_receipt_sha256` binds the separately saved `billing-receipt.json`.
   Preserve/redact provider receipts appropriately. Missing records cannot be
   replaced by guessed spend. Proposed $5/two-hour ceilings remain unauthorized
   and are not provider-enforced caps.

## Validation and remaining evidence

- Isolated pinned native suite: **115 passed** (52.85 seconds).
- Full default suite: **2,182 passed, 16 optional skips**, three existing SWIG
  warnings (101.66 seconds). Ruff and Git whitespace checks pass.
- The native tests include two scripted arms through actual Docker, shared budget,
  native rendering, saved-record/review export, and native admin loading against
  loopback HTTP with an explicitly mocked GPU identity. They make no real model,
  GPU or learning claim. The initial native run exposed the namespace-path bug;
  the complete isolated rerun passes after its fix. No boundary was relaxed.
- All 29 frozen runtime files and original source/reference hashes remain intact.
  Numerical protocol SHA remains `1d0038b7c5d2f84327ac1bdcf55d939bcbdbfcbf1d5a991d07ef81ffb05aa8d9`.
  Earlier user edits are preserved. No files were staged or committed.
- Two fresh plans validate 81 selected training questions with the original
  integration/training caps. Plans and logs: `out/reader-paired-prep-20261024/`.
- Actual paired Qwen results: **not run**. Locked reference review: **not opened**.
  Consecutive GPU training, GPU Docker placement and provider watchdog: **pending**.
  No new paid calls, GPU resource, checkpoint update or upload.
