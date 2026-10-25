# Verifiers adapter trial

> The user subsequently requested code simplification. The custom training path
> has been removed; the active Verifiers path is 1,344 Python lines including tests
> after native prime-rl verification (497 implementation, 847 tests).
> The original trial logs/manifests below describe that earlier source snapshot.
> See the current report and `simplification.json` for post-cleanup validation.

An optional Prime Intellect Verifiers **0.4.0 evaluation integration**, using the
existing frozen Qwen Envoy investigator and actual Docker source inspections.
Read [the trial report](../../docs/VERIFIERS_PORT.md) for installation, reproduction,
the exact scope and the remaining native-training gate.

`controls/` contains three **authored reference controls**, copied from the earlier
local readiness work. Each contains original policy payloads, fixed raw replies
and the expected packet. They are not newly sampled Qwen trajectories and cannot
be used as on-policy training data. The local fixture returns no invented token
IDs, log-probabilities or model-reported usage.

The controls adapt QASPER (Dasigi et al., 2021; CC BY 4.0), at source revision
`13b496d2a5359329b110e3419628de3cf791843b`. Passage scopes use evidence annotations;
questions add a short-fact answer instruction. Original source snapshots and the
role/reward history are in `data/rl/reader-facts-v1/`. This is not an original
question benchmark or an official QASPER leaderboard result.

The saved results cover three unchanged controls, one fenced-Python variation and
one provider failure. Full receipts and failed intermediate setup/test attempts
remain under `out/verifiers-port-20261022/`. The validation manifest indexes the
new files and detailed local receipt hashes. Native eval traces contain raw model
messages but no trainable sampled token streams; all are explicitly untrainable.

No real Qwen inference, paid model calls, GPU launches, optimizer updates or
commits occurred. Review is by the implementing assistant; independent human
review is null. Passing these checks establishes interface compatibility, not
model improvement or readiness of the native prime-rl trainer.

`token-preflight.json` records the later local native TrainClient/graph checks.
The fallback renderer matches all nine saved V5 prefixes; the optimized Qwen3
renderer matches only the three first turns. Twelve real tokens from a tiny random
CPU model test capture/masks; synthetic malformed/fenced fixtures test failure
handling and raw text preservation. None is a pretrained V5 rollout or an RL
update. The production adapter still rejects training clients. See the report's
October 23 section for reproduction and the remaining integration gate.

`prime-rl-local/` records the later full native Env/Docker data handoff into
prime-rl 0.10.0 group advantages, batch packing and trainer-side transport
decoding. The opt-in `token_contract_only` mode keeps original traces untrainable;
only explicitly synthetic in-memory copies exercise native training compilation.
No optimizer consumes them. See
[the local integration report](../../docs/PRIME_RL_LOCAL_CHECK.md) for the passed
checks and remaining Linux/GPU boundary. Older manifests describe their named
historical snapshots; the new directory has the current source hashes.
