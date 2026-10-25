# Native Prime-RL GPU smoke — October 23, 2026

One real V5 reader group passed the unchanged Docker harness, native sampler/scorer
check, one native optimizer update, exact checkpoint reload, and cold vLLM loading.
This is engineering proof, not evidence of improved answer quality.

See [the report](../../../docs/PRIME_RL_GPU_SMOKE.md). `result.json` records the
actual counts, timings, artifact identity and private Hub commit. The trainer and
inference configs are fully resolved. All 29 frozen runtime files are unchanged.
No weights or full native checkpoints are stored in Git. Complete raw episodes,
engine responses, packed batch, scripts and logs remain locally under
`out/prime-rl-smoke-20261023/`; these small receipts do not replace those raw traces.

Review is by the implementing assistant. Independent human review is null.
