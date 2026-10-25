# October 25 unused-code and commit review

The user authorized committing the accumulated work after checking for unused
code. This is a local reference/lint/reproducibility review, not an independent
security review or a new model evaluation. No paid work was run.

## Findings

- Reviewed the 11 new `src/training/` modules, four new CLI scripts, installable
  Verifiers plugin, their callers/tests, and existing MCP/sandbox configuration
  changes. Ruff found no unused imports or local variables in the new code.
- An AST reference scan flagged `ReaderEnv`. It is exported by
  `environments/envoy_reader/src/envoy_reader/__init__.py` and resolved through
  Verifiers plugin discovery, including the actual native collector. It is used
  and must not be deleted based on the absence of direct constructor calls.
- The F1 and semantic-reward candidates are not the active GRPO reward. They have
  documented offline CLI callers, saved negative controls and regression tests.
  Retain them to reproduce why those reward candidates were not adopted; do not
  mistake their presence for an enabled LLM judge or F1 training reward.
- The withdrawn custom learner/trainer implementation and its dedicated tests
  were already removed in the earlier simplification. The current path delegates
  training to Prime-RL. No additional orphaned runtime code was established, so
  this review does not delete functioning plugin or diagnostic code.
- Checked 493 pending files for credential filenames, private-key headers and
  common API-key patterns: no matches, binary model weights, symlinks or files
  larger than 5 MB were found. This limited scan is not proof that all possible
  secret formats are absent. Ignored `out/` directories, virtual environments,
  cloud connection files and adapters are excluded from commits.
- Preserve the three original untracked October 16 plan/spec files as user inputs.
  Preserve the protected legacy training/reward files. Commit the related
  implementation, source-bound data, tests, small evidence records and docs.

## Verification

The native focused suite passed **160 tests**, with three expected CPU distributed
checkpoint warnings. The main full suite passed **2,224 tests with 17 optional
skips**, with three existing SWIG warnings. Ruff and Git whitespace checks passed.
The archived `ssh-read-retry.patch` retains its required space-prefixed blank
context lines and recorded hash; a path-specific Git whitespace attribute avoids
misclassifying those patch delimiters as source-code trailing whitespace.

Frozen runtime hashes and the last pilot's implementation-bound controller plan
still validate. The revised task pool contains 80 eligible training questions;
the 12 evaluation questions remain locked. The saved case-study checker returns
`verified`, meaning artifact integrity and recount of prior judgments, not fresh
semantic review. No runtime source was changed by this review.

Commit groups: explicit sandbox image configuration; reader training/data/tests
and reproducibility evidence; architecture/status/README/résumé documentation.
No push, new training run or cloud deployment is included.
