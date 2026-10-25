# Gated Answer F1 reward test — candidate rejected

[Protocol, results and decision](../../docs/READER_F1_AUDIT.md#follow-up-gated-answer-f1-candidate--protocol-before-running).
`results.json` contains 65 authored controls on the existing eight reward-validation
papers plus four previously saved real Qwen episodes from one training question.
Fixtures assume inspection for construction; saved episodes use their original
controller inspection receipts and are reverified against their original packets.
No locked model-evaluation references are scored. Human review remains null.

The candidate preserves the old inspection, clipping and attributed-reference
support gate and replaces only the answer whitelist with upstream Answer F1.
The original answer is retained and scored, never replaced in model traces.
Full reference controls score 1; proposed paraphrases average .598; all 32
invalid-evidence controls score zero. The wrong numeric answer `15.00 sentences`
receives the same score of 1 as reference `1500 sentences`, failing the declared
gate. Negated references average .827. The candidate remains disabled; these are
lexical metrics, not semantic grading or an evaluation of trained-model gains.

Reproduce from the repository root into a new output path:

```sh
python scripts/audit_reader_f1_reward.py --output out/reader-f1-reward.json \
  --saved-rollouts out/prime-rl-smoke-20261023/rollouts
```

The ignored local rollouts are optional; omitting the argument reproduces all
65 authored controls. Original records, source/evaluator/implementation hashes,
per-case results and gate decisions are saved. No retries, requests, network
calls, Docker executions, GPU launches, training updates, or Git commits occur.

Focused tests: **47 passed**. Full suite: **2,128 passed, 16 skipped**, three
existing SWIG deprecation warnings, 96.17 seconds. Ruff and whitespace checks
pass. All 29 frozen runtime files and frozen source/reference hashes validate
unchanged. Of 2,076 tracked files hashed before this test, only the intended
`CODEX_HANDOFF.md` update changed; existing user changes were preserved. No staging
or commits. These passing regression tests do not override the failed reward gate.
