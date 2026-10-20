# Assigned-reader two-case smoke preparation — October 20, 2026

Prepared locally, not run. No inference requests, paid model calls, Runpod mutations,
or GPU activation occurred in this step. The CLI supports only `prepare` and `check`.
The last verified endpoint state from the preceding task was parked (min/max workers
0/0, zero workers and no pending jobs); this preparation did not query its current state.

Implementation: `272685b`. The frozen [bundle](../../examples/assigned-reader-smoke-prepared/plan.json)
records the exact working-tree hashes, including preserved uncommitted configuration.
The commit alone does not identify that working tree. Its source snapshot is a local
cache dependency, so checking the bundle on another machine requires that same cache.

## Two development cases

Both use the controller-assigned, version-pinned paper *Fast Inference from Transformers
via Speculative Decoding*, arXiv `2211.17192v2`, with the assigned V2 reader protocol.

1. **Known answer:** “What batch size was used for the walltime measurements?”
   The cached paragraph reports batch size 1 on a single TPU-v4. Require a correct
   claim, source inspection, named paper citation and exact supporting spans.
2. **Target-setup gap:** “From the available extracted text, what walltime speedup,
   if any, is established specifically for speculative decoding on an NVIDIA RTX
   A5000 at batch size 8?” Require a qualified unresolved gap and useful next query;
   do not transfer the TPU-v4/batch-1 measurement to this setup. Cited partial
   findings are acceptable when their measured setup is clear. Missing evidence
   does not imply zero/negative speedup or absence throughout the literature.

These are assistant-authored diagnostics on a familiar paper, not held-out benchmark
questions. The HTML extraction can omit tables and figures. String no-match counts
are not proof that the complete paper lacks a result.

The frozen [references](../../examples/assigned-reader-smoke-prepared/references.json)
contain an exact cached paragraph and claim-review requirements. They are separate
from [model inputs](../../examples/assigned-reader-smoke-prepared/model_inputs.json).
The [review template](../../examples/assigned-reader-smoke-prepared/claim-review-template.json)
is blank: no model answer or human answer review exists yet. Exact quotation matching
establishes provenance; each eventual answer also needs a small claim-level review.

## Frozen limits for later execution

| Limit | Value |
| --- | ---: |
| Cases / attempts per case | 2 / 1 |
| Actions and model requests per case | 6 / 6 |
| Total model requests, including formatting retries | 12 |
| Input / output token maximum per request | 7,168 / 1,024 |
| Reserved input / output token maximum across requests | 86,016 / 12,288 |
| Whole-session watchdog from activation | 900 seconds |
| Provider execution timeout per job | 120 seconds |
| Inference resubmissions | 0 |
| Maximum active GPU workers | 1 |

These are frozen requirements for the future live executor. This preparation does
not start that executor or demonstrate the watchdog. Existing durable queue-ledger
limits were tested locally with fake responses: each case rejects request seven,
including after reopening its ledger. A later live run must use exactly the two
ledgers, refuse replacement sessions after an uncertain submission, apply the action
and request caps, and kill/reap its child process at the watchdog deadline.

The bundle pins Qwen3-8B, the v5 checkpoint-50 adapter and revisions/checksums,
Docker image, source/corpus, effective assigned-reader prompt, decoding settings
(temperature 0, top-p 1, seed 42, thinking off), and implementation files. With the
cached pinned tokenizer, initial prompts contain **1,938 and 1,964 tokens**. Later
observations still require per-request token enforcement.

Live activation remains a separate approval step. Before activation, recheck the
current GPU rate and pinned worker identity, explain cold-start/idle billing, start
the watchdog with activation, and park/verify the endpoint afterward. Request/token
caps and client termination do not provide a provider-side dollar ceiling or prove
remote cancellation. No price or actual spend is claimed here.

## Reproduce the offline preparation

Use a fresh output directory; preparation never overwrites an existing bundle.
The tokenizer must already be cached at the pinned revision. No provider credential
is required.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/prepare_assigned_reader_smoke.py prepare \
  --library-root out/qwen-live-cheaper-20261019/native-v3/servers/library \
  --sandbox-image sha256:79bd8bf84bd4b7f2a32a0596cb1fd90164716c1cfca879f78829411193646382 \
  --output out/assigned-reader-two-case-prepared-20261020

python scripts/prepare_assigned_reader_smoke.py check \
  --output examples/assigned-reader-smoke-prepared
```

The checker rejects changed artifacts, runtime pins/limits, implementation bytes,
source context or token counts. The manifest checksum detects changes; it is not a
signature or an approval mechanism.

## Verification and interpretation

Focused tests: 36 passed across preparation, queue transport and the existing known-paper
diagnostic. Ruff and CLI help passed. The real cached-tokenizer preparation and a
separate check of the copied public bundle passed. All ten protected file hashes
match the preceding task. Independent preparation review found no actionable issues;
this was an engineering/source-reference review, not an independent human answer review.

Full-suite results are recorded in the accompanying
[validation artifact](../../examples/assigned-reader-smoke-prepared/validation.json).
The earlier narrow v5 diagnostic did not use the new assigned/gap protocol and is
not a pass for these cases. No current-protocol live answer, latency/cost improvement,
base-versus-trained improvement, or comparison with a larger assistant has been established.

See the [assigned-reader implementation report](2026-10-20-assigned-reader-gaps.md)
for the controller handoff, sandbox grammar and gap behavior this future smoke tests.
