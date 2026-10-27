# Qwen Envoy experiment atlas

This is a **local reading map of the experiments already saved in this repository**.
Start with a question below, open the linked run record, then inspect the raw
artifact before making a new decision. It needs no MCP server, model endpoint,
GPU, or new training run. A link labelled “led to” records the project's research
sequence; it does not prove that one paper or experiment caused another result.

The entries distinguish measured observations from model-assisted or
implementing-assistant interpretation. QASPER questions and annotations are from
Dasigi et al. (2021), CC BY 4.0. The selected question sets here are development
evidence, not an official full-test leaderboard or independent human review.
The [QASPER case study](TECHNICAL_CASE_STUDY.md) and each linked artifact give
the protocol and detailed limits.

## The connected research paths

```text
Can Qwen investigate papers reliably?
  → v5 supervised fine-tuning
  → paired 40-question code-agent comparison
  → fixed BM25 baseline and matched reader comparison
  → retrieval experiments and live hybrid-input check
  → reviewed correction replay (no new model inference)

Can reward-based training fix the remaining reader errors?
  → reward counterexample audit
  → bounded Prime-RL/Verifiers pilot
  → three optimizer updates, interrupted before held-out evaluation

Can two cheap readers share one model?
  → two-reader serving profile
  → overlap observed; latency benefit unresolved

Does structured research memory improve decisions?
  → five-question web/Markdown versus MCP pilot
  → no decision or recovery advantage observed
```

These arrows are **workflow chronology and follow-up questions**, not causal
claims or a combined benchmark. The arms, datasets, hardware, and scoring rules
differ across rows. Do not compare their headline numbers as one leaderboard.

## Experiment records

| Question and actual attempt | Saved observation | Decision and next question | Primary evidence |
| --- | --- | --- | --- |
| **Did v5 SFT produce a usable paper worker?** Qwen3-8B was fine-tuned with rank-4 QLoRA on 35 replay-verified training conversations; checkpoint 50 was selected. | The recorded four-epoch run took 579.49 s on an A40. The training set contained 106 supervised actions and 4,812 target tokens. | Keep v5 as the tested checkpoint, but evaluate answers separately from tool reliability. Training loss alone cannot answer the product question. | [Training record](../release/qwen-v5-training/README.md), [dataset card](../data/sft/qasper-v5/README.md) |
| **Did v5 beat base Qwen on the same 40 code-agent questions?** Both ran against named QASPER papers. | Recorded-error episodes fell from 22/40 to 1/40. Official Answer F1 rose 19.92% → 30.69% overall, but fell 29.83% → 21.37% on the 20 answerable questions. Provisional supported-answer passes on answerable questions were 3/20 for both. | Better execution reliability and more refusals are observed; better grounded answers are **not established**. Inspect premature stopping and false refusals. | [Findings](QASPER_AGENT_STUDY.md), [saved trajectories](../release/envoybench-v0.1/run/results.json), [official score](../release/qasper-agent-study/qwen-official-score.json) |
| **What happens with a simpler fixed retrieval baseline?** BM25 chose passages before a single answer request to each checkpoint on the same 40 selected questions. | Official Answer F1 was 41.52 for base and 58.86 for v5. The v5 arm refused all 20 labelled unanswerable questions and falsely refused 9/20 labelled answerable ones. | Passage availability matters, but token F1 and refusal patterns do not prove semantic support. The different serving context also prevents a clean agent-versus-baseline causal claim. | [Baseline record](../release/qasper-agent-study/README.md#fixed-retrieval-baseline), [score](../release/qasper-agent-study/retrieval-baseline/official-score.json) |
| **Does v5 improve as a narrow passage reader?** Base and v5 were compared on a small matched development reader set. | Model-assisted review recorded 2 supported passes for base and 1 for v5; neither arm had an execution-failure episode. The predeclared improvement threshold was not met. | Do not promote v5 on reader quality from this test. It is small and not a held-out population estimate. | [Live summary](../examples/base-v5-reader-comparison/live-summary.json), [per-question results](../examples/base-v5-reader-comparison/per-question-results.json) |
| **Can better passage selection repair missing evidence?** An offline two-window comparison tried BM25, dense, hybrid, and hybrid plus reranking on exposed QASPER development sets. | The prespecified reranked candidate fully covered an annotated paragraph on 5/19 dev answerable questions versus BM25's 4/19; it missed the required two-question gain. Plain hybrid had the strongest secondary ablation, but was not the selected candidate. | The reranked method failed its decision gate; no production default changed. Test a selected passage with the reader before claiming answer improvement. | [Retrieval report](../examples/hybrid-passage-retrieval/README.md), [decision](../examples/hybrid-passage-retrieval/decision.json) |
| **Did plain hybrid passages help actual v5 responses?** A bounded live run changed passage ranges for three already exposed questions and kept the checkpoint, prompt and request budget fixed. | Complete requested answers improved from 0/2 historical BM25 cases to 1/2 under implementing-assistant review. The humor case still refused despite relevant terms in the supplied quote; the gap case still failed the structured-gap criterion. Zero Python execution errors were recorded. | The overall prespecified quality gate failed. This separates missing retrieval from failure to use visible evidence; it does not justify a general hybrid or model-quality claim. | [Live report](../examples/qasper-hybrid-v5/README.md), [claim review](../examples/qasper-hybrid-v5/claim-review.json) |
| **Can the two reader errors be expressed as valid actions?** Two reviewed final actions replaced saved endings after exact Docker replay of the three original episodes. | Five fixed-action episodes and 15 policy payloads replayed; the original three matched their saved packets before corrections ran. The authored corrections produced a specific humor answer and a structured bias gap. | This proves the interface can express the corrections. **No Qwen inference** chose those actions, so it does not measure learning or response to new feedback. | [Replay report](../examples/reader-correction-replay/README.md), [replay summary](../examples/reader-correction-replay/summary.json) |
| **Is the proposed RL reward safe enough to train against?** Saved controls and adversarial counterexamples exercised overlap and scoped reward candidates. | Each candidate admitted unacceptable full-reward controls; the audit found 12, 24, and 36 such records across the three candidates. These are existence proofs, not population error rates. | None of those candidate scorers was promoted as a semantic correctness signal. Independent answer/support review and a stronger gate remained necessary. | [Reward audit](../examples/reader-rl-investigation/README.md), [results](../examples/reader-rl-investigation/summary.json) |
| **Did the bounded reader RL pilot improve v5?** Native Prime-RL/Verifiers training began from pinned v5 on one A100. | Three of ten planned optimizer updates completed and passed fresh adapter reloads. Collection for update four ended with a 502 ReadTimeout during laptop sleep. No held-out paired evaluation ran. | Retain original v5. The run demonstrates real optimizer updates and an operational failure, **not an answer-quality gain or loss**. | [Pilot report](READER_RL_PILOT_20261025.md), [receipt summary](../examples/verifiers-port/reader-pilot-v2-20261025/summary.json) |
| **Can two reader requests share one Qwen engine?** The automatic profile sent ten fixed-input model jobs and nine telemetry jobs to one A5000/v5 worker. | Telemetry sampled two overlapping requests. Serial versus concurrent pair times differed by 0.068 s and 0.979 s in two rounds; the first is inside the predeclared 0.4 s uncertainty margin. | Concurrent admission worked, but a reliable latency advantage is **inconclusive**. No generated action was executed or graded for research usefulness. | [Profile report](../examples/two-reader-profile/automatic-live-1/README.md), [result](../examples/two-reader-profile/automatic-live-1/result.json) |
| **Did the research MCP beat ordinary web research plus Markdown notes?** A predeclared five-question paired pilot measured claims, decisions and fresh-session recovery. | Web/Markdown earned 10/10 decision points; MCP earned 9/10. Both recovered 5/5 decisions, and MCP was better in 0/5. Claim support was no worse for MCP in 3/5, below its 4/5 gate. | The pilot did **not** show a decision or recall advantage. Structured provenance is inspectable, but that alone is not a demonstrated productivity gain. | [Protocol and results](RESEARCH_MCP_PAIRED_PILOT_20261026.md), [per-case summary](../examples/research-mcp-paired-pilot-20261026/summary.json) |

## Paper → attempt → result example

The retrospective [PagedAttention question trail](../scripts/demo_research_question.py)
associates a verified passage from [PagedAttention, arXiv:2309.06180v1](https://arxiv.org/abs/2309.06180v1)
with the recorded two-reader serving profile above. The trail names the paper
as motivation; the saved profile shows two requests sharing one engine, while
the latency advantage remains unknown. This is an **agent-proposed research
connection**, not an experimentally isolated effect of PagedAttention.

The local question record is under `~/.envoy/research-second-brain/records/`.
It is a worked retrospective example, not an import of all the experiments in
this atlas. The source experiment artifacts remain in their linked repository
locations; nothing here rewrites or merges the raw runs.

## What this catalog covers

The table curates the decision-bearing runs above. The repo also retains setup
failures, dry runs, scorer audits, and earlier attempts under `examples/`,
`release/`, and local ignored `out/`. Some raw training and controller logs live
only in the original checkout's ignored `out/` directory; the committed reports
and receipts above are the portable sources. This atlas is a readable index, not
a new evaluation or a semantic judgment over every trace.

To use it without MCP, open this file in your editor or ask your coding assistant
to read it and follow its source links. For example: **“What did our hybrid
retrieval experiment change, what failed afterward, and which saved artifacts
support that conclusion?”** The assistant should inspect the linked run files
before proposing a new experiment.
