# Envoy agent guide

## Maintained scope

Qwen Envoy is a public technical case study: a Qwen3-8B code-execution worker, Docker paper tools, saved QASPER evaluation, and QASPER Agent Studio. An optional research MCP links paper evidence to experiment records. Read [the technical case study](docs/TECHNICAL_CASE_STUDY.md), [the findings](docs/QASPER_AGENT_STUDY.md), and [the project guide](PROJECT_GUIDE.md) before changing these paths. Historical prototypes and one-off experiments are preserved at the `archive-full` Git tag.

The QASPER study uses selected questions from Dasigi et al.'s dataset; it is not an original dataset or an official full-test leaderboard result. Saved replay verifies hashes and arithmetic, not semantic support. A quote match proves provenance, not claim correctness. Do not fill an independent human review with model judgments. The v5 dataset includes AI repairs, and the RL pilot did not establish answer-quality improvement.

## Engineering boundaries

- Keep the agent multi-step and tool-mediated. Model-authored Python must run inside Docker, not on the host.
- Preserve the recorded v1 execution and evaluation interfaces when inspecting old runs. New execution settings should be opt-in.
- Treat [`src/env/reward.py`](src/env/reward.py) as the active historical MuSiQue reward, not a reward for research synthesis.
- Before changing training or rewards, read [the improvement brief](docs/GPT_ASTRA_IMPROVEMENT_BRIEF.md) and state the hypothesis, expected signal, and decision rule.
- Make experiments reproducible: record checkpoint, question split and IDs, corpus revision, decoding settings, reward version, hardware, and seed.
- Prefer isolated changes and focused verification. Do not launch paid inference or training without a bounded plan and authorization.
- Preserve uncommitted user files. Do not infer product success from a working MCP connection, valid JSON, or quotation spans.

A product-quality claim needs useful research workflow evidence and a held-out, human-reviewed supported-answer improvement over base Qwen without unacceptable execution, latency, or cost regressions. Neither has been established by the saved study.
