---
record_id: research-retrieval-correction-2026-09-02
kind: correction
captured_at: 2026-09-02
supersedes: research-retrieval-hypothesis-2026-08-20
status: reviewed
---

# Retrieval worker correction, September 2

I reviewed a small local pilot after writing the August hypothesis. In that pilot, one-pass retrieval gave more complete supported responses on the reviewed questions than the iterative worker. The pilot used a selected sample of research notes and few questions, so it is a development signal, not a general result about all libraries or all Qwen checkpoints. The exact counts and full human review sheet are not preserved in this sample vault; I should not invent a win rate from this note.

I now prefer to start with the cheap retrieval baseline and call the iterative worker only where it adds evidence that the baseline missed or resolves a conflict. This corrects the earlier default-win expectation. To justify paying for a worker, I need a paired test on held-out questions with the same snapshot and source-support rubric, while recording latency and inference cost.

## Linked history

- [[Learning/Research/2026-08-20-retrieval-hypothesis|Earlier hypothesis]]
- [[Learning/Research/2026-09-12-worker-decision|Architecture decision]]
