# Generated-token logprob diagnostic (2026-10-03)

This is an inference diagnostic of the pinned Qwen3-8B base and v5 adapter,
not a new benchmark score, model-training result, or calibrated confidence test.
The September 30 40-question comparison already exists. The new measurement
asks whether vLLM can return usable generated-token logprobs alongside the
unchanged code-tool trajectories, so that future abstention work has inspectable
telemetry rather than a guessed confidence number.

## Frozen smoke set and decision

- Use the frozen `dev` split, with seed 42, 15 maximum program rounds,
  greedy decoding, thinking disabled, and 1,024 maximum output tokens per turn.
- The two IDs were chosen before this run as the first development question in
  each QASPER-derived answerability stratum:
  `qasper_validation_003d6f9722ddc2ee13e879fefafc315fb8e87cb9`
  (`insufficient`) and
  `qasper_validation_0d34c0812f1e69ea33f76ca8c24c23b0415ebc8d`
  (`sufficient`). Both models receive exactly these questions and settings.
- Hypothesis: the endpoint returns finite sampled-token logprobs for both
  checkpoints without changing the action protocol. Expected signal: each arm
  records at least one generated turn with `valid_logprob_count > 0`, a complete
  trace, and no adapter-load or sandbox error.
- If either arm lacks telemetry, fails to load, or cannot run the sandbox,
  stop after the smoke test and diagnose offline. If both arms pass, run the
  entire 40-question `dev` split once per arm with the same serving configuration
  to collect a descriptive distribution. Do not choose questions based on smoke
  outcomes or change decoding between stages.
- Interpret logprobs only as likelihoods of tokens the model actually emitted.
  They are not probabilities that an answer is correct, that a paper contains an
  answer, or that its cited evidence supports a claim. The QASPER-derived labels
  and saved model-assisted reviews are not independent human ground truth.

Use one A40-class pod, budget at most 45 minutes of running time and $1 of
compute at the observed $0.49/hour rate. If setup or inference runs long, stop
at the time cap even if the full split is incomplete. Record the actual GPU,
runtime/version, model and adapter revisions, server logprobs mode, pod ID,
runtime, output paths, and observed cost. Run the benchmark client and its
labeled Docker sandbox locally through an SSH tunnel to the pod's vLLM server;
the logprob capture code is in the current uncommitted worktree.

Copy outputs locally and stop the pod immediately after the run, including on
failure. Verify the stopped state with Runpod. Terminate the new pod after
artifacts are safe locally to avoid continuing storage charges.

## Observed run and protocol deviation

The two-question smoke test passed for both arms. Base submitted 2/2 answers
with valid sampled-token logprobs on all 18 generated turns (566/566 reported
tokens). V5 submitted 2/2 with the same coverage on all 6 turns (259/259
tokens). This establishes that the pinned adapter loads and vLLM can return
generated-token logprobs through the existing code-tool harness. It does **not**
establish confidence calibration or answer quality.

The planned 40-question expansion is **incomplete as a paired evaluation**:

| Observation | Base Qwen3-8B | v5 adapter |
| --- | ---: | ---: |
| Rows recorded | 40 | 40 |
| Submitted answers | 36 | 11 |
| Endpoint errors | 4 context-limit HTTP 400s | 1 disconnect and 28 connection refusals after pod stop |
| Generated turns with logprobs | 368/368 | 36/36 |
| Valid / reported token logprobs | 11,521/11,521 | 1,575/1,575 |

Only 10 question IDs have submissions from both models. The run manifest says
`complete` because the runner wrote all 80 rows, including error rows; it does
not certify that both models completed the split. Do not compute or present a
40-question model win/loss from this run. The 40 questions are previously used
development questions, the answerability labels have not been independently
human-validated, and no blinded supported-answer review was performed on these
outputs. The logprobs refer only to generated tokens, including tool code; they
are neither calibrated correctness probabilities nor evidence-support scores.

The remote server was vLLM 0.30.0 on a Runpod Secure Cloud A40 48 GB in
CA-MTL-1, pod `hvlssbkhwovar8`, using an 8,192-token serving context. Both
arms used `logprobs=true`, `top_logprobs=0`, raw-logprob mode, greedy decoding,
thinking disabled, and a 1,024-token output limit per turn. The base revision,
adapter revision and checked adapter SHA-256, local runner configuration,
benchmark/corpus hashes, seed, and code hashes are in the saved manifests. The
local Docker sandbox image ID is also recorded there.

The 45-minute stop limit was **missed**. The pod started at approximately
21:08:58 UTC and was confirmed `EXITED` at approximately 22:22 UTC, about 73
minutes later. This also interrupted the v5 arm. At the displayed $0.49/hour
compute price, 73 minutes corresponds to roughly $0.60 of GPU time before any
storage or billing adjustments; the final bill had not posted at the time of
this report. Runpod confirmed that the stopped pod has no mounts or network
volumes. The local SSH tunnel has been closed. No further GPU run was started.

Artifacts: `out/envoybench/logprob-probe-20261003/smoke/` and
`out/envoybench/logprob-probe-20261003/dev-full/`. These are local ignored
outputs; preserve them before cleaning the worktree. A future full comparison
would need a new run with explicit context management and a wall-clock stop
mechanism that cannot be delayed by benchmark or orchestration waits.
