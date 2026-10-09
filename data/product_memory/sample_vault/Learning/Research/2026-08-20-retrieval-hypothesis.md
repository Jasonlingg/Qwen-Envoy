---
record_id: research-retrieval-hypothesis-2026-08-20
kind: hypothesis
captured_at: 2026-08-20
status: superseded
---

# Retrieval worker hypothesis, August 20

For a sample research library, I proposed that a multi-step Qwen worker would probably beat a single BM25 lookup because it could search again after reading a promising passage. That was a hypothesis about a local workflow, not a published finding or a measured result. I had not compared the two systems on the same questions, source snapshot, or review rubric when I wrote it.

The useful part of the idea was the proposed test: freeze the documents and questions, hold the answer budget fixed, inspect exact quoted spans, and compare source-supported answers along with extra time and cost. I did not want to count a real quotation as proof that the answer claim was supported. The September 2 correction supersedes my expectation that the iterative worker would win by default.
