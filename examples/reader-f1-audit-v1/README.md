# Offline reader F1 audit

See [the hypothesis, definitions, result and decision](../../docs/READER_F1_AUDIT.md).
`results.json` records all 48 authored controls and four saved real training-role
Qwen episodes, per-annotation evidence availability, per-case metrics and group
means. The authored records assume inspection for fixture construction; they are
not model rollouts or independent human review. The four real episodes are
reverified against their original source packets before scoring.

The script records its own SHA, the frozen source manifest/revision, the case hash
and hash-pinned upstream evaluator. Output is exclusive, preventing accidental
replacement of a previous result. Reproduce from the repository root:

```sh
python scripts/audit_reader_f1.py --output out/reader-f1-audit.json \
  --saved-rollouts out/prime-rl-smoke-20261023/rollouts
```

The full saved smoke rollouts are ignored local artifacts. Omit the optional
argument to reproduce the 48 authored controls using repository data alone.

No API/model requests, GPU launches, training, active reward changes, or commits.
Only reward-validation papers and already exposed training episodes are scored;
locked model-evaluation references stay unused. Paragraph-overlap mapping and
character coverage are explicitly nonofficial diagnostic adaptations. Real quotes
and high lexical scores do not establish semantic support or model improvement.

Focused verification: 39 passed. Ruff passes. Frozen runtime: 29/29 unchanged;
frozen source/reference manifest validates unchanged. Full suite: **2,120 passed,
16 skipped**, three existing SWIG deprecation warnings, 152.21 seconds.
