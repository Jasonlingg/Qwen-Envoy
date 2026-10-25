# Reader RL pilot preparation

This folder contains the pre-run protocol and follow-up implementing-assistant
training-label review. No new inference or training result is stored here.

- [Protocol](protocol.json): model/source identities, 81 eligible training IDs,
  12 locked evaluation IDs, two-update check, ten-update pilot, paired review
  rules, fixed stopping criteria and an **unauthorized** $5 budget proposal.
- [Review](training-review.json): all 82 training cards, 81 retained for the pilot
  and one hold for an incomplete accepted alias. Source excerpts were reviewed;
  full-paper completeness and independent human validation are not established.
- [Explanation and readiness](../../docs/READER_RL_PILOT.md).

Original frozen data/references/reward remain unchanged. The held training paper
does not enter evaluation. Locked evaluation reference contents remain unopened.
The protocol remains preparation, not a guarantee of learning. The paired runner
and review gates are now implemented; see the [evaluation runbook](../../docs/READER_PAIRED_EVALUATION.md).
Consecutive GPU training, its Docker placement, provider watchdog and fresh paid
authorization remain launch gates. Locked references were not opened during wiring.

Recheck data integrity and launch selection from the repository root:

```python
from pathlib import Path
from src.training.reader_data import load_bundle, reviewed_training_ids

bundle = Path("data/rl/reader-facts-v1")
_, roles, _, cards = load_bundle(bundle)
assert len(roles["train"]) == 82
assert len(reviewed_training_ids(bundle, roles, cards)) == 81
```

`SHA256SUMS.json` binds these pre-run documents. Any intentional protocol change
must be recorded before training or evaluation; an edit after model outcomes
cannot preserve the claim of preregistration.

A subsequent [real integration attempt](../../docs/PRIME_RL_INTEGRATION_20261024.md)
stopped at its optimizer-counter gate after one actual gradient update. The
current implementation manifest records the post-failure local initialization/
watchdog repair; it is not the original pre-run implementation freeze. Numerical
protocol unchanged; no held-out reference/outcome was opened. The earlier
manifest/source snapshot is retained with the failed-run local archive.
