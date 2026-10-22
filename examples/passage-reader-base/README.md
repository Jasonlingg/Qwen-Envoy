# Base Qwen passage-reader attempt

One newly authorized base Qwen3-8B attempt on the same three exposed development
cases as the [saved V5 repaired rerun](../passage-reader-rerun-1/README.md).
The [predeclared scope](../../docs/implementation/2026-10-21-base-passage-reader.md)
keeps prompt, source windows, tools, four-action limit and decoding unchanged.
Only the model route changes to `envoy-base-only`, without applying V5.

First requests were checked against V5's frozen requests: only `model` differs.
The effective prompt SHA and product implementation hashes match. Source,
question and review-input hashes match; references stay outside model inputs.
Scripted local rehearsal is synthetic and does not establish model quality.

Prepared inputs and the plan-bound single-attempt allowance are frozen here
before activation. Existing attempts remain consumed. Results and source-based
claim review will be appended after the attempt; human review remains blank.
This is a sequential, three-case development diagnostic, not a held-out
model-improvement result, statistical comparison or reason to start training.
