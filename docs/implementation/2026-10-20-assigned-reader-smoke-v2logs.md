# Assigned-reader live smoke after the log repair — October 20, 2026

The pinned Qwen v5 adapter ran both frozen questions on an RTX A5000 and the
endpoint is verified parked again. The startup repair worked for this attempt,
but **neither answer passed the frozen development gate**: one fail and one
partial result. This is an end-to-end component test, not evidence of fine-tuning
improvement or advantage over a larger assistant.

## Authorization and execution

The user explicitly authorized GPU use for one additional attempt. The
[approval](../../examples/assigned-reader-smoke-v2logs-approval.json) was committed
as `4644f9d` before activation. It binds the same plan hash, source, questions,
effective prompt, Docker sandbox, decoding settings and limits. Both prior
attempt records remain untouched; the previous global marker was archived
byte-for-byte and a new exclusive marker identifies this attempt. Its allowance
is now exhausted. No automatic replacement or additional paid run was performed.

The unchanged hypothesis was whether v5 could inspect its assigned paper, return
a supported batch-size fact and report a qualified A5000/batch-8 evidence gap.
The frozen rule requires both cases to meet source/claim requirements with no
execution errors, repeated-call rejections or budget violations. Mechanical
span validity alone cannot pass a case.

Preflight confirmed zero workers/jobs and the one-A5000 pin. A fresh catalog read
quoted $0.69/hour serverless, with high availability under minimum CUDA 13.0.
The runtime verified base Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218` and the v5 adapter revision
`27b912a863ff914ad45baa03624d8911dc1e17fb`, subfolder
`artifacts/full/checkpoint-50`, SHA-256
`7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3`.
The identity log reports RTX A5000, PyTorch 2.13.0+cu130 and vLLM 0.31.0.
Temperature 0, top-p 1, seed 42 and thinking disabled were retained.

Unlike the previous attempts, the canonical v2 worker ID successfully retrieved
live container logs and the pinned identity. Some log reads timed out and bounded
retries succeeded. This validates the live startup path for this attempt; it does
not establish the causes of either historical failure.

Model-authored document actions ran inside the pinned Docker sandbox. The limits
remained six actions/requests per case, twelve requests total, 7,168 input and
1,024 output tokens per request, one GPU, no inference resubmissions, and a
900-second watchdog armed before activation. Actual use was **7 completed model
requests**, 16,824 reported input tokens and 290 output tokens. No request failed;
the watchdog did not fire. The lifecycle records ~180.075 seconds before cleanup,
which is not invoiced GPU time. Actual billed spend is unknown; these limits are
not a provider dollar ceiling and startup/idle/storage can bill.

## Actual question and action results

| Question | Requests | Reader time | Mechanical result | Source-visible review |
| --- | ---: | ---: | --- | --- |
| Batch size used for walltime measurements | 3 | 77.18 s | Cited spans valid | **Fail:** denied information explicitly present in its citation |
| Speedup established for A5000 at batch 8 | 4 | 18.56 s | Accepted abstention; one controller rejection | **Partial, not pass:** no invented speedup, but missing gap report and next query |

In the batch-size case, action 1 inspected the abstract; action 2 searched for
`batch size walltime` and displayed the batch-size-1/TPU-v4 measurement statement;
action 3 nevertheless claimed the inspected evidence did not establish batch
size. Its second exact quote itself says “a batch size of 1 on a single TPU-v4”.
Provenance passed while the claim contradicted the source. Retrieval supplied the
answer; the final interpretation failed.

In the A5000 case, action 1 read the first abstract window. Action 2 assigned
`search_within(...)` to `hits` without displaying it; the controller rejected it
before execution. Action 3 read the next window, which duplicated abstract text;
action 4 abstained. No numeric target-setup speedup was fabricated, but no accepted
targeted search, `report_gap` or useful next query reached the host. These reads
cannot establish absence from the full paper or literature.

There were **zero Python execution errors, one controller action rejection, zero
repeated-call rejections and zero budget breaches**. Empty execution-error lists
must not be described as error-free tool use. The original runner did not expose
a separate controller-rejection counter; a local reporting-only repair now does,
in commit `552cb45`, with a regression using the observed undisplayed-assignment pattern. The original
live `results.json`, packets and request ledgers are preserved unchanged. The
separate [claim/action review](../../examples/assigned-reader-smoke-v2logs/claim-review.json)
records the derived rejection at step 2.

Root and independent source-visible reviews agree on fail/partial. These are
**model-assisted judgments, not independent human review**. The frozen human
review template remains blank. Both questions are familiar assistant-authored
development cases, not held out; there was no base-Qwen or larger-assistant
control and no training reward was computed.

## Cleanup and evidence

All seven submitted jobs are recorded completed. Cleanup read-back and subsequent
independent MCP reads confirm min/max 0/0, zero active workers, all health worker
counts zero and no queued/in-progress jobs. A first post-cleanup health rollup
transiently reported running=1 while the active-worker list was empty; later reads
all report zero. The [final state](../../examples/assigned-reader-smoke-v2logs/final-runpod-state.json)
uses those later reads. No further activation occurred.

The [runtime record](../../examples/assigned-reader-smoke-v2logs/runtime.json)
retains exact execution-file hashes, approval and frozen limits. The
[validation artifact](../../examples/assigned-reader-smoke-v2logs/validation.json)
records tests, review, artifact hashes and preservation checks. All ten protected
files and both prior published attempt sets remain unchanged.

Preflight focused tests: **65 passed** in 4.42 seconds. After the reporting fix,
focused runner/preparation/approval tests: **27 passed** in 1.58 seconds; the full
offline suite: **1,638 passed**, no skips, three existing SWIG warnings in 68.91
seconds. Ruff and whitespace checks passed. Independent review found no issue in
the approval/lifecycle boundaries or reporting fix and agreed with the claim
review. These local tests did not make additional paid calls.

The result identifies concrete follow-up work: improve evidence interpretation
and gap-report compliance before a broader comparison. It does not justify
promoting this worker or starting another training run.
