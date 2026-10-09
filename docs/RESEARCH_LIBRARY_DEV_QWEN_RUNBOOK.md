# Public-paper Qwen development run

The first live check uses **four development questions**, not the twelve
held-out questions. It exercises the same `QwenInvestigator` code-execution
worker as the paper-library MCP tool: Qwen searches and inspects a frozen
20-paper library, then proposes an answer with exact source spans. This is a
failure diagnosis before training, not a score proving model improvement.

## Ready without GPU

The current dry run is
`out/research/research-library-dev-qwen-dryrun-v2-20260928/`. Its manifest
pins question IDs `D01`–`D04`, benchmark SHA-256
`797e5bc40ac1559fc957a19edef2a94a3597992399501f006808af5a134f3154`,
corpus hash
`9a9c1d750daf5647898eb18681a5f75775ba76047713316dcc5583f6ed201598`,
the public-paper prompt, tool code hashes, decoding, seed 42, and 15 steps.
The dry run made **zero model calls**. Reproduce it with a fresh output path:

```bash
python scripts/run_research_library_qwen_dev.py --dry-run \
  --output out/research/research-library-dev-qwen-dryrun-new
```

The source-check sheet and `reference-review-template.json` in that output
are pending. A person must compare each reference answer and answerability
judgment with the frozen paper text, check the required paper IDs, and decide
whether each anchor supports the claim. If a reference changes, regenerate
the dry run so the review record matches the new benchmark SHA. Marking a
form `approved` without doing that inspection is not a valid review. The
runner rejects a pending or mismatched record **before** using Docker or a
model.

## Live development baseline after reference review

Serve pinned base Qwen3-8B and the existing v5 adapter on a GPU, with an
OpenAI-compatible endpoint and a built `rlm-sandbox` Docker image on the
machine running the evaluation. Run one model arm at a time, using a fresh
output directory and the same snapshot, prompt, tool version, decoding, seed,
step cap, hardware class, and reviewed reference record. For example:

The current Mac has neither the sandbox Docker image nor cached Qwen weights,
so the live runner needs a prepared GPU host (or a local Docker setup plus a
remote model endpoint); the dry run above works here now.

```bash
python scripts/run_research_library_qwen_dev.py \
  --output out/research/research-library-dev-base-live \
  --endpoint http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-8B \
  --checkpoint Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218 \
  --hardware 'GPU model and runtime' --run-label base \
  --review-record path/to/approved-reference-review.json
```

Use the served adapter name and pinned adapter revision for the v5 arm. The
CLI's checkpoint string is an **operator attestation**; verify the actual
served weights before trusting the comparison. Each run saves a packet for
each completed question, a manifest, model-call count, exact-span checks,
and a mechanical handoff. A valid quote is still not a semantic support
verdict. Blindly review the saved answers and compare paired failures before
deciding whether a bounded, base-initialized training run is justified.

The paper-library [MCP server](RESEARCH_LIBRARY_MCP.md) can expose the same
worker to a larger assistant. The older `scripts/run_eval.py` has a different
prompt and discovery rule; its scores must not be presented as this product
worker's before/after result.
