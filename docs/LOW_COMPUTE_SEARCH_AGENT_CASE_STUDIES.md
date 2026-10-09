# Small-compute search-agent training: independent project check

Research check, September 26, 2026. This corrects a possible reading of the
[paper-scale replication audit](RETRIEVAL_TRAINING_REPLICATION_AUDIT.md): **one
GPU is enough to test and sometimes improve a narrow search or retrieval
model**. The large papers' multi-GPU setups are not a minimum requirement for
every useful experiment. Results below are the project authors' reports, not
independent replications or performance predictions for Envoy.

| Project and public source | Data, model, compute | Reported base/control → trained result | What transfers, and what does not |
| --- | --- | --- | --- |
| [Siyanda Matika's document-search agent](https://matikasiyanda.github.io/blog/agent-rl/part-1-data/), [results](https://matikasiyanda.github.io/blog/agent-rl/part-3-results/) | Qwen3-1.7B; 400 format-teaching SFT trajectories, then 200 GRPO steps on 3,104 verified synthetic questions; 98 insurance documents / 3,785 chunks; **one RTX 4090 24 GB**. Author reports ~2 days for GRPO, plus ~14 h question generation, ~3 h teacher traces, and ~5.5 h difficulty sampling. | On 296 held-out questions, SFT agent NDCG **0.499 → RL 0.542**; best one-shot hybrid retriever **0.431**. On an unseen programming-book corpus, SFT **0.760 → RL 0.781**. | Closest evidence-worker design: multi-step `search`, `read`, `report`; output only document IDs, randomized each episode; corpus-independent transfer measured. Most transfer gain over one-shot retrieval existed *before RL*. Domain, corpus size, and model size differ from ours; the blog's linked project repositories were unavailable at this check, so the exact run is not independently runnable from the page. |
| [Yu's search-agent RLVR](https://github.com/yuyu0529nya/search-agent-rlvr/blob/main/reports/search_agent_rlvr_findings.md), [single-card launch script](https://github.com/yuyu0529nya/search-agent-rlvr/blob/main/scripts/grpo/run_search_agent.sh) | Qwen2.5-7B-Instruct, 4-bit QLoRA and GRPO; **one 32 GB GPU** for the initial experiment. Four iterations used 128 HotpotQA training questions × eight rollouts, with vLLM generation and updates alternating on the card; author reports ~20 min for that run. | Initial 300-question held-out exact match **0.390 → 0.460** at iteration 3, then **0.247** at iteration 4. Later F1/length-aware experiments reached **0.493** from a **0.387** base. | Closest size/VRAM precedent for Qwen action training. But BM25 searches each question's *own distractor set*, not a shared large corpus. The author's analysis attributes much of the accuracy gain to shorter answers; answer EM does not demonstrate improved evidence finding. Later ablations used two RTX 5090s. Adapters and raw eval rollouts are not published. |
| [OpenPipe ART·E](https://openpipe.ai/blog/art-e-mail-agent), [base comparison](https://openpipe.ai/blog/ruler), [code](https://github.com/OpenPipe/email-deep-research) | Qwen2.5-14B searches/reads Enron email; roughly 4,000 synthetic questions across 20 training and eight test inboxes; SQLite FTS5; GRPO. Final run: **one H100**, under a day, author-estimated **$80 GPU cost**. | On the team's task, base Qwen **41% → trained 96%** answer accuracy with the hand-tuned reward. | Strong small-team proof that training a multi-step search agent can matter. Its H100 has substantially more memory than our historical 32 GB card; synthetic inbox QA and model-judged answers are not scientific passage support. Final-run cost excludes data generation and preceding experiments. |
| [Paper-QA-RAG-LoRA](https://github.com/maheshpaulj/Paper-QA-RAG-LoRA) | MiniLM *embedding retriever*, trained on 232 hard-negative triplets from six papers for ~8 s on a Colab T4; evaluated on 125 questions from five different papers. | Hit@5 **0.776 → 0.856** without reranking. With a cross-encoder reranker, base **0.928** versus tuned **0.936**. | A genuinely tiny personal retrieval-model improvement and a useful control: cheap reranking nearly removed the fine-tune's incremental benefit. This is an encoder and one-shot retrieval, not a Qwen code-execution agent or a supported-answer result. |

The two closest personal examples make **compute feasibility** credible for a
bounded search-policy experiment. They do not establish Sonnet-level research
capability or a gain on Envoy's code-execution tasks. Matika's agent is 1.7B,
uses only 98 source documents, and reports a small RL gain beyond its SFT agent.
Yu's 7B/32 GB run optimizes short-answer exact match against a small
per-question corpus. OpenPipe shows a larger gain but used a larger GPU and a
narrower, partly synthetic task. The paper-QA example shows why a trained model
must beat a simple reranking control, not just an untuned weak baseline.

## Implication for Envoy

The historical one-32-GB Qwen3-8B QLoRA runs in this repository already prove
that **adapter training fits**; they do not prove a helpful search policy. The
limiting uncertainty is now whether our proposed training questions and search
environment expose a repeatable *search-decision* failure that a trained policy
can fix. We need the same held-out code-execution questions for base and trained
Qwen, an indexed shared corpus, a simple retrieval/reranking control, and
human-reviewed source support. That is one bounded experiment, not a demand to
reproduce a multi-GPU paper or keep adding eval sets indefinitely.

**Hypothesis:** after training on successful multi-step evidence-search
trajectories, Qwen will select more of the needed passages than base Qwen under
the same tools and budget, allowing an unchanged host model to write more fully
supported answers.

**Expected signal:** a paired gain in small-packet source coverage *and*
human-reviewed supported-answer rate, without unacceptable execution failures,
latency, or cost. Better tool syntax or answer exact match alone would not count.

**Decision rule:** first freeze and review the personal-vault baseline required
by [the project brief](GPT_ASTRA_IMPROVEMENT_BRIEF.md), identify named cases where
search decisions rather than missing documents or synthesis are the bottleneck,
and measure a small data-collection/QLoRA pilot budget. Only then spend on one
training run. If a straightforward retriever plus reranker already supplies the
needed evidence or the search-policy pilot has no held-out gain, stop that lane
and keep the negative result. The full [product and model gates](WEEKLY_RESEARCH_RADAR.md)
still apply.
