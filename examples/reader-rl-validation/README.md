# Local reader RL validation

This is an exposed reward-development experiment, not new Qwen evaluation or
training. [Scope, actual results and remaining gates](../../docs/READER_RL_VALIDATION.md).

- `reviewed-keys.json`: 14 implementing-assistant source reviews, 20 claim/support
  keys and 28 finite complete-answer variants. All diagnostic-only; human review null.
- `results.json`: per-record candidate and old-proxy rewards for all 150 saved
  cases. Missing-scope cases have `reward: null` and must be excluded from this objective.
- `reviewed-variant-checks.json`: 28 local terminal-scoring checks; no Python execution.
- `summary.json`: 0/74 known answerable negatives rewarded; 13/14 old positive
  controls accepted, with one source-support disagreement retained; 28/28 reviewed
  variants accepted. All 12 unapproved reference-string controls score zero.
- `preflight.json`: still blocked; frozen runtime identities, original data hold,
  old trainer incompatibilities, zero paid budget. This is not a launch guard.
- `manifest.json`: scoring inputs, source and core-output hashes.
- `validation.json`: bundle, report, code and test bindings; `checks.json`,
  `preservation.json`, `recompute-check.json` and raw logs record verification.
  Focused 60 passed; final full suite 2,116 passed, no skips, three SWIG warnings.
  A focused reporting failure from a subprocess-based platform probe is retained;
  reporting was corrected to avoid subprocesses. The first full run also passed.

The finite answer list makes rejection of unlisted outputs largely true by
construction. It can reject correct paraphrases and shorter sufficient citations.
Semantic support comes from attributed source review, not quote matching or a
parser. There is no independent reward calibration, gap-quality objective,
training-ready dataset, new model rollout, generalization or Qwen gain here.

Recompute to a **new** output directory:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/validate_reader_rl.py \
  --reviewed-keys examples/reader-rl-validation/reviewed-keys.json \
  --output out/reader-rl-validation-reproduction
```

Exit **2 is expected**: the report explicitly says training is not ready. No GPU,
API key, model download, Docker execution or training is requested by this command.
It checks the saved 150-record bundle before scoring. Existing outputs are never
overwritten. Scores/summaries are deterministic; host metadata may differ.

The original perplexity control still says "independent test sets"; its cited
excerpt does not establish independence. Two narrower reviewed variants pass.
The old source/control artifacts were not rewritten to hide the disagreement.
