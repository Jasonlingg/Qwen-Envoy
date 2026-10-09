---
record_id: research-worker-decision-2026-09-12
kind: decision
captured_at: 2026-09-12
derived_from: research-retrieval-correction-2026-09-02
status: planned
---

# Worker decision, September 12

For the next prototype, I will use lexical search to select source candidates and ask Qwen to investigate only questions that need multiple notes, dates, or uncertainty handling. Qwen's job is to return a small, exact evidence packet; the main assistant explains it to me. The worker should not write a lasting personal memory without my approval.

This is an architectural decision, not evidence that Qwen already beats lexical search. The September 2 correction is why I changed the default from “always use a multi-step worker” to “use one when it earns its extra cost.” I will revisit the decision if a held-out comparison shows that Qwen fails to add source coverage or introduces unsupported claims. This sample vault contains no later outcome for that comparison.
