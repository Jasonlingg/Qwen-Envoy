# Native prime-rl local contract receipts

Read [the integration report](../../../docs/PRIME_RL_LOCAL_CHECK.md) for reproduction
and limitations. This tests native prime-rl 0.10.0 with the unchanged Docker
investigator. It is not an RL training result or a real pretrained V5 rollout.

- `summary.json`: stack, scoped task identity, per-scenario/action results, native
  sample counts, test results and remaining GPU gates.
- `upstream-source.json`: hashes of the actual installed native modules exercised.
- `dependencies.json`: exact package versions in the isolated Python 3.12 environment.
- `validation.json`: final source/artifact hashes and preservation/cleanup checks.
- Logs: focused native tests and full legacy regression checks.

Engine responses are authored and their probabilities synthetic. Only explicitly
marked in-memory copies enter the native compiler; no optimizer consumes them.
Original episodes remain untrainable. The separate tiny random CPU-model sampler
test checks actual token probability capture, without pretrained weights or updates.

The single reused public training-role control adapts QASPER (Dasigi et al., 2021;
CC BY 4.0), source revision `13b496d2a5359329b110e3419628de3cf791843b`.
Annotation-assisted passage scopes and finite-reference rewards are component
checks, not semantic quality or retrieval benchmarks. No locked evaluation
question was used. Independent human review is null.
