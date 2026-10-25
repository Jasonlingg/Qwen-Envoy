# Reader pilot v2: training aliases and a bounded quality comparison

Prepared October 25, 2026. **No paid authorization or execution.** This is the
prospective successor to the unchanged v1 protocol, after the successful native
two-update integration and the saved training-answer audit. It reuses the native
Prime-RL trainer, optimizer, loss, renderer and Docker investigator.

Hypothesis: ten native GRPO updates from original pinned V5, using a less brittle
training whitelist, improve complete source-supported short-fact answers on the
same 12 locked evaluation papers. Expected signal: at least three net additional
supported answers, with at most one supported-answer regression, no new unsupported
claims or execution-failure episodes, and the original resource tolerances.
These remain provisional model-assisted judgments unless independently reviewed.
Reward increases alone do not establish better answers.

## Training-only reward revision

`reward-revision.json` defines `reader-short-fact-reference-v2`. It adds six exact
phrases across four training questions from the attributed October 24 review.
The seventh proposed phrase belonged to the ambiguous representation question;
that entire question is held instead. The existing incomplete financial-dataset
question remains held. **80 questions are eligible**, consumed once in seed-42
order with four trajectories per question. The eight reward-validation papers
and 12 locked evaluation papers are unchanged and cannot receive new aliases.

The revision binds the original bundle/review/card hashes and audit artifacts.
Only finite accepted-answer phrases can change. It retains exact matching,
source scope, inspection receipts, reviewed support spans, clipping and extra
citation checks. No F1 or LLM-based reward is enabled. This is model-assisted
training-data curation, not independently established semantic truth.

The original `data/rl/reader-facts-v1/` keys and all v1 receipts remain unchanged.
Default tasksets and integration plans still use v1. Selecting this protocol
explicitly binds the v2 revision through the controller, Verifiers taskset,
reward and GPU plan. Revised plans fail if the protocol, reward identity, source
review, eligible questions, caps or implementation drift. Earlier source-bound
execution plans are historical; regenerate plans against the current source.

## Proposed budget and stopping rule

Proposal: **$8 estimated allowance, at most 13,500 seconds (3h45) of owned-resource
lifetime**, including setup, source transfer, training, comparison, export and
cleanup. This raises the earlier $5/two-hour proposal and requires its own
explicit authorization. The prepared protocol still says `paid_authorization:false`.

- Training: exactly ten verified updates, 1,024 requests, 8,388,608 reserved tokens,
  at most 10,800 seconds (three hours) of native work. Restart from original V5;
  neither diagnostic integration adapter initializes this run.
- Evaluation: 24 episodes (12 questions × V5 and final step ten), 288 requests,
  2,359,296 reserved tokens, at most 1,800 seconds, no retries or replacement questions.
- Setup/export/cleanup: reserve 900 seconds outside those work phases.
- Rental: one 80GB GPU, sequential inference and training; temporary 600GB STANDARD
  volume; at least 360GiB free checkpoint disk at preflight. Keep at least 10% of
  the estimated dollar allowance outside the planned time×rate bound. Combined
  quoted hourly rate must be at most $1.92; recheck all GPU/storage/root-disk charges.

The live catalog read returned A100 SXM (`NVIDIA A100-SXM4-80GB`), 80GB, Secure
price **$1.79/hour**, POD availability MEDIUM with CUDA floor 13.0; US-WA-1 was LOW.
US-WA-1 supports STANDARD network volumes. The
[official storage price](https://docs.runpod.io/storage/network-volumes) is
$0.07/GB/month for the first TB. `price-read.json` preserves the tool replies;
capacity is not reserved. The combined planning bound remains $1.90/hour, subject
to a final root-disk and capacity check at launch.

The observed-pattern estimate is 3.09h / $5.88 at that planning rate. A 25% time
margin would need 3.87h and **does not fit** the proposed 3h45 limit. This is an
explicit stop, not a completion guarantee. If time, tokens, requests, disk, healthy
mixed groups or the single question pass run out, retain V5 and report incomplete;
do not open evaluation for a partial training candidate. A completed candidate
may still fail to finish evaluation within the guard; report that as incomplete.

The independent watchdog requires a matching paid-authorization receipt to enable
this larger allowance. Legacy invocations retain their old $5/two-hour limits.
It deletes only resources identified by this launch's create receipts and verifies
absence. It depends on the local machine and provider control plane remaining
available; it is not a provider-enforced dollar ceiling or tax-inclusive guarantee.
The authorization template is deliberately false and cannot enable this budget.

## Preparation and evaluation

CPU-only preparation, using the isolated native environment:

```bash
python -m src.training.prime_run out/reader-pilot-v2-controller \
  --assets /absolute/local/assets \
  --updates 10 --requests 1024 --tokens 8388608 --seconds 10800 \
  --pilot-protocol examples/reader-rl-pilot-v2/protocol.json
```

The external GPU plan uses the same protocol/revision and the controller plan hash;
paths may differ, while contracts and source identities must match. Generated
Python remains in local Docker with networking disabled. No host REPL fallback.
Before dispatch, require fresh provider reads, the explicit authorization receipt,
and the independent watchdog heartbeat. The local prepare command does not rent
hardware, start an endpoint, or grant permission to the provider.

After ten fully verified updates, `reader_evaluation.freeze()` follows the protocol
bound into the training plan. It freezes final candidate identity before any
locked reference export. All 12 references must be reviewed before outcomes;
ambiguity makes the quality gate inconclusive without replacing questions. The
paired runner uses the same frozen prompt/tools/scope/decoding for both arms and
creates an identity-masked claim review. Independent-human-review fields stay
null when review is model-assisted. Keep raw traces and report refusals, incomplete
answers, unsupported assertions, execution errors, latency, tokens and lifecycle
cost. Publish the result whichever direction it goes; no automatic promotion,
checkpoint search or additional paid retry.

## Offline verification

The saved-output replay covers all 44 original healthy, untrainable episodes.
On all saved episodes, positives change 4→7 and mixed groups 2/11→4/11. Removing
the newly held question leaves 40 episodes / ten groups: positives 3→6 and mixed
groups 1/10→3/10. This rescoring does not measure learning or changed policy
behavior. `fixed-output-replay.json` retains every result and the hold flags.

Validation: 183 focused native tests passed, including revised taskset/reward
transport and portable candidate freezing under both protocols. The new adapter
test was rechecked in the isolated pinned environment after correcting its
optional-dependency handling. The full default suite passed **2,224 tests with
17 optional skips**, including the new skip for incompatible Verifiers in the
main environment. Three expected CPU checkpoint warnings occurred in native
tests; the default suite has three existing SWIG deprecation warnings. Ruff,
diff checks, frozen runtime/source hashes and the saved case-study checker passed.
`verification.json` records preservation and preparation checks; `SHA256SUMS.json`
binds the prospective artifacts and current source without rewriting v1 receipts.

No GPU, new inference, optimizer update, evaluation reference review, commit,
upload or promotion occurred during this preparation. The earlier two-update
checkpoint remains diagnostic. Actual model quality and pilot completion remain
unmeasured.
