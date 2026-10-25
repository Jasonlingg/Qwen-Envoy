# Reader RL readiness evidence — October 22, 2026

> **Historical, withdrawn custom-training path.** [STATUS.json](STATUS.json)
> supersedes the earlier readiness flags for current use. The custom trainer and
> CLI have been removed at the user's request; the saved plan is not executable.
> Use [the smaller Verifiers path](../../docs/VERIFIERS_PORT.md). Old source hashes
> below describe the historical implementation, not the current working tree.

The local implementation passes its checks. **The actual Qwen3-8B GPU learner is
not yet verified.** No cloud resources, paid model calls or new commits were made.
See [the implementation report](../../docs/READER_RL_READY.md) for the architecture,
commands, reward limitations and first experiment's decision rule.

Start with `readiness.json`. `plan.json` pins 16 training questions, model and
code identities, decoding, the source-reviewed reward, signal gates and hard
generation/time/update limits. The historical baseline's temperature-zero metadata
does not specify training sampling; the plan's `sampling` object specifies
temperature one. The paid authorization field remains null. This is a preparation
record, not evidence of an 8B rollout, successful training or improved answers.

- `runtime-parity-summary.json`: three original saved-action Docker replays.
- `active-pool-control-verification.json`: 90 eligible current-pool controls from
  101 authored Docker episodes; 11 excluded draft-pool controls are regression only.
- `reference-validation/`: 84 finite-reference contract controls on eight papers.
- `focused-tests.log`, `full-suite.log`: actual local pytest outputs, including
  warning details; all new optional training tests ran in the pinned environment.
- `fresh-cleanup-verification.json`: all 104 owned replay/control containers are
  absent on a fresh Docker inspection; unrelated containers were not changed.
- `history-upgrade-check.json`, `raw-question-history-check.json`: the historical
  closure audit. An intermediate history artifact can describe the initial draft;
  the active dataset manifest and `active-pool-control-verification.json` bind the
  final 82/8/12 pool.
- `preservation-verification.json`: original dirty files and all other old files
  preserved except the two intentional data-inventory changes; unchanged HEAD
  and empty index.
- `validation-manifest.json`: hashes of this smaller review bundle, repository
  implementation/tests/report and dataset manifest, plus detailed local receipts.

The detailed private local outputs remain at
`out/reader-rl-ready-20261022/`, including raw policy messages, source inspections,
container receipts, unsuccessful preparation attempts and earlier draft plans.
The receipt index records their hashes; those large files are not all copied
here. Earlier draft plans are stale and must not authorize execution. The dataset
itself is in `data/rl/reader-facts-v1/` with source attribution, hashes and role locks.

Reference review is by the implementing assistant, not independent human review.
Fixed-action replay cannot test responses to new feedback. Exact source provenance
does not by itself establish semantic support. Future model promotion requires
matched source-supported evaluation; this bundle makes no improvement claim.
