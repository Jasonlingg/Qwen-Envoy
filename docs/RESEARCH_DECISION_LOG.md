# Research-to-decision log

This is a record of **which outside work affected a decision or experiment** in
QASPER Agent Studio, and what our own evidence said afterward. It is selective,
not a bibliography. Dates below are the earliest supporting repository record
unless marked retrospective; a document's date alone does not prove when a
paper was first read. The [prior-art review](../benchmarks/envoybench/PRIOR_ART.md)
lists related systems without claiming that we invented paper-agent evaluation.

| When and source | Decision or test in this repository | Outcome and boundary |
| --- | --- | --- |
| **September 14 — [QASPER](https://aclanthology.org/2021.naacl-main.365/)** (Dasigi et al.). [Dataset decision](RESEARCH_DATASET_DECISION.md), [conversion pilot](QASPER_PILOT.md); commit `f933cd7`. | Use QASPER's existing research-paper questions, answers, answerability labels, and annotated evidence for a **known-paper** task. The pilot pinned the dataset revision, preserved splits, and excluded figure/table or unmappable evidence rather than inventing text spans. | QASPER supplies the questions; Envoy adds a bounded Python-tool environment. The final selected 40 questions are not original questions, an open-web discovery test, or a full QASPER leaderboard result. On **October 9**, commit `e99b165` added the [pinned official Answer F1 scorer](../release/qasper-agent-study/README.md#scoring-boundary) for the unchanged saved outputs. |
| **September 14 — [s3](https://arxiv.org/abs/2505.14146)**. [Dataset strategy](RESEARCH_DATASET_DECISION.md); commit `f933cd7`. | Treat a smaller evidence-gathering worker feeding a larger answerer as a plausible architecture. | This was a design precedent, not a transferred result. The current public Studio evaluates the investigator; it does not demonstrate the complete two-model product or s3's RL method. |
| **September 18 plan, September 21 result — [Search-R1](https://arxiv.org/abs/2503.09516), [ReSearch](https://arxiv.org/abs/2503.19470), and [GRACE](https://arxiv.org/abs/2601.04525)**. [RL plan](QASPER_RL_EXPERIMENT.md), [confirmation](QASPER_GRPO_CONFIRMATION.md); confirmation commit `d36d711`. | Use search-agent RL as the research basis for a bounded QASPER GRPO experiment, then test on separate unseen papers before promoting the checkpoint. | The development improvement did not replicate in the confirmation, so the RL checkpoint was **not promoted** as an improvement. The plan document was added to Git later; its September 18 date is a retrospective record, not independent proof of when each paper was read. |
| **September 20 experiment, documented in October — [GAVEL](https://arxiv.org/abs/2609.19315)**. [Verifier experiment](GAVEL_EVIDENCE_VERIFIER.md); code committed `1976690` on October 8. | Borrow only GAVEL's distinction between mechanically detectable action mistakes and semantic planning. Track observed document IDs, repeated calls, and premature abstention; give bounded feedback or escalate. | The 25-question ablation kept supported-answer passes at **11/25** with and without one-turn feedback, while mean steps rose from 2.24 to 2.64. We rejected it as an answer-quality improvement. The document was committed retrospectively; GAVEL does not prove our verifier improves research answers. |
| **September 28 design, committed September 29 — [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) (engineering article), [ALCE](https://arxiv.org/abs/2305.14627), and [FActScore](https://arxiv.org/abs/2305.14251)**. [Architecture proposal](RESEARCH_LIBRARY_HARNESS_ARCHITECTURE_20260928.md); commit `d1297ba`. | Propose a narrow read-only tool broker and sandbox for the investigator, then give the writer a checked evidence bundle. Keep exact source provenance separate from the harder judgment that a passage actually supports a claim. | This is partly a **proposal**, not a shipped end-to-end two-model workflow. Structural citation checks exist; independent semantic support review remains unfinished. The code-execution article argues for tool composition and context efficiency, not error-free generated Python. |
| **October 8 — [AstaBench](https://allenai.org/blog/astabench), [Inspect View](https://inspect.aisi.org.uk/log-viewer.html), and other prior art**. [Positioning review](../benchmarks/envoybench/PRIOR_ART.md); committed with Studio in `7c03930`. | Frame the release around pinned runs, per-question traces, resource/error diagnostics, and separate execution versus answer review. | These are precedents for the **auditable presentation**, not proof that a trace viewer or scientific-agent benchmark is novel. Because the review and Studio entered Git together, this record does not establish that every viewer feature came from those sources. |

Two later reads should **not** be credited as causes of the original system:

- The executable-Python direction was recorded on **September 15** in
  [the code-execution plan](CODE_EXECUTION_SECOND_BRAIN.md) (commit `54cc438`).
  [CodeAct](https://arxiv.org/abs/2402.01030) appears in a later
  [research audit](QWEN_VAULT_TRAINING_RESEARCH_20260928.md) as supporting
  precedent. We cannot say CodeAct caused the September 15 choice.
- [Harness-Aware Distillation](https://arxiv.org/abs/2610.02858) was discussed
  after the Qwen training and the September 30 comparison. It suggests a
  **future training hypothesis**: teach a small model to act on harness feedback.
  We have not implemented that distillation method. The October 10
  [syntax-feedback change](../src/env/document_env.py) (commit `ecd921a`)
  came from inspecting Nemotron's saved malformed actions; it has local tests
  but no post-change model score.

The October 9 [Nemotron reference run](../release/qasper-agent-study/EXPERIMENT.md)
was a predeclared, budgeted comparison prompted by the project's NVIDIA/Nebius
demo needs. It was not a replication of any paper in this log.

The [findings report](QASPER_AGENT_STUDY.md) is the outcome check on these
choices. V5 raised official QASPER Answer F1 from **19.92% to 30.69%** on the
selected 40 questions, but answerable-question F1 fell and provisional
supported-answer passes stayed **3/20** for both base and v5. That result
supports a narrower engineering claim about protocol execution and refusals,
not a literature-backed claim that v5 became a better substantive paper reader.
