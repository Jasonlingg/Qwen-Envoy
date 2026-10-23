# Offline reader RL investigation

**RL is plausible; no tested reward is ready for training. No new Qwen result.**
[Research, prior experiments, trainer audit and proposed gates](../../docs/READER_RL_INVESTIGATION.md).
The current 24-example SFT export is also [on hold](../../data/sft/passage-reader-pilot-v1/HOLD.json)
because its original exclusion scan omitted structured historical task pools.

This is an adversarial development audit, not an RL training run or a model
benchmark. The 150 records consist of:

- 24 already saved, reviewer-authored demonstrations from twelve training papers;
- 3 already exposed actual V5 outputs and 2 previously replayed corrections;
- 121 authored counterfactual terminals, including 12 reference-string controls.

There are 27 acceptable controls, 111 unacceptable examples, and 12 annotation
controls. Judgments are source-visible implementing-assistant review or explicit
counterexample design, not independent human labels. Full-reward attacks are
existence proofs about these scorers, not a population error rate.

| Scorer | Unacceptable full-reward records | Acceptable zero-reward controls |
| --- | ---: | ---: |
| Unchanged QASPER answer/evidence overlap | 12 | 13 |
| Scoped/two-read eligibility + overlap | 24 | 0 |
| Scoped/two-read eligibility + exact answer | 36 | 14 |

The scoped candidates are diagnostic scaffolds, not deployed reward functions or
complete runtime validators. They use external reviewed scope labels and require
two distinct reads for these controls. The existing runtime can accept a supported
answer after one read; this audit makes no general equivalence claim for all
possible runtime trajectories. Submitted ranges are checked before packet quote
clipping, matching that aspect of the runtime.

An unrelated distractor read followed by gold offsets can get full legacy reward.
Protocol gating fixes that, but syntactically valid generic/false-impossibility
gaps still get full scoped reward. Exact matching can reward a correct answer
with an irrelevant real quote and can reject legitimate paraphrases. The VATEX /
WMT task-binding reversal ties the reviewed answer at **0.347826** overlap reward.
All three candidates fail the gate; none is wired into training.

Only source provenance and saved formatting inputs are mechanically checked.
No counterfactual terminal was executed or sent to Qwen. The adapter replays saved
payload formatting locally with fixed replies; it does not start Docker, a host
Python REPL, a model endpoint or an optimizer. A test explicitly forbids socket
connections, subprocess launches and REPL startup during the actual audit.
No six-paper development source or reserved research_reader_v1 pack was opened
for this audit. The prior exposed three-question regression remains regression-only.

Focused tests: **58 passed**. Final full suite: **2,093 passed** in 133.22s,
no skips and three existing SWIG warnings. [Commands/counts](checks.json),
[full log](full-suite.log), [preservation](preservation.json). One preceding full
attempt had a loopback fixture timing-comparability failure; its
[log](full-suite-first-after-history.log) and [diagnosis](full-suite-failure-note.json)
are retained. The isolated file then passed all three cases before the full rerun.
All 29 frozen runtime files and protected original dirty/training files remain
unchanged. Four intended prior files changed for inventory tests/gating and hold
notices; all other 2,039 baseline file hashes are unchanged.

## Files and reproduction

- [Inputs and counterexample rationales](inputs.json), [all scores](results.json),
  [decision/counts](summary.json), [bound audit files/code](manifest.json).
- [Historical trainer inspection](trainer-inspection.json): old prompt,
  `use_docker=False`, no current investigator or terminal gap support. It was
  parsed as AST data; no training entry point ran.
- [Supplemental historical identifier audit](history-coverage.json): 13 omitted
  benchmark/question files, 96 additional papers, 659 additional question IDs;
  seven training papers / fourteen trajectories affected, zero development papers.
  Prepared-pool membership is not proof of optimizer use. The original artifact
  files are unchanged, and current preparation/export rejects their old registry.
- [Primary research references and local history hashes](research.json).
- [Corrected schema-v2 inventory](corrected-inventory.json),
  [strict source/identifier check](inventory-check.json): 230 files, 1,063 excluded
  papers and 2,041 question IDs. This new snapshot also includes the already
  created pilot/audit artifacts; it does not retroactively make the old data fresh.
- [Recomputed saved scores](saved-score-check.json), [complete bundle hashes](validation.json).

The 16-question eligible count in the old selector is a feasibility count before
supplemental exclusions, not a recommended RL training size. Larger training pools
need a deliberate reuse/new-data policy; prior evaluation papers stay excluded.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/audit_reader_rl.py --output out/reader-rl-audit-repeat
```

Saved reward inputs/scores can be checked without cloud access. Recomputing the
historical coverage audit needs older local `out/` files not all redistributed;
missing history is reported as unavailable, never treated as zero prior exposure.
This directory contains no training export. Current GPU/paid allowance is zero.
