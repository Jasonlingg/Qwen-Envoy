# A new Qwen task: adaptive evidence discovery for the research radar

Research decision, September 26, 2026. This is a proposed experiment, **not** a
claim that a new checkpoint has already improved.

**Replication correction (September 26):** [the paper-by-paper compute/data
audit](RETRIEVAL_TRAINING_REPLICATION_AUDIT.md) shows that the 1,000–2,000-question
QLoRA proposal below is a low-resource pilot, **not a replication** of the
published methods or evidence that this sample size is sufficient. In particular,
the closer multi-turn search-worker precedent, s3, used 2,400 PPO examples on
five A100 80 GB GPUs plus a frozen 14B answer model. Do not budget or launch the
pilot from question count alone.

## Decision

Train Qwen to **find a small, complete set of source passages across a large AI-paper
library**. A weekly research question or proposed digest takeaway is the input. Qwen
writes Python using `search()`, `read()`, and `extract()`, changes its search when an
early result is weak, and returns at most five ranked passages with paper IDs and
exact locations. A fixed larger assistant writes the interpretation and digest from
that packet. Qwen is scored primarily on whether its evidence enables a *supported*
answer, not on whether it can imitate that answer's prose.

This differs from the QASPER run in four testable ways:

| | Earlier QASPER task | Proposed task |
| --- | --- | --- |
| Input | Question about a known paper or a few supplied candidates | Information need with no gold paper IDs in a large library |
| Model output | Free-form final answer plus citations | Bounded ranked source passages, coverage gaps, and no final synthesis |
| Hard part | Read and answer within a paper | Discover complementary papers and evidence among distractors, then stop |
| Training signal | Answer overlap or teacher answer trajectory | Gold-paper/evidence retrieval quality after each executable search action |

For example, the input could be “Which recent search-agent papers actually
measured whether training improved multi-hop evidence retrieval?” Qwen would
locate the relevant papers and return the passages containing each comparison,
metric, and evaluation condition. It would not write the cross-paper conclusion.
The weekly-digest writer would use that packet to explain what changed and what
the results do or do not establish.

This is a narrower model objective, but a real product task. Each weekly digest and
later MCP query needs relevant evidence. The user sees a source-backed update; the
host model is spared repeated exploration of the entire library. Merely returning
valid IDs or quote spans is insufficient: the fixed host must be able to make a
supported answer from the selected passages.

## Why this is the best training bet

The current answer-writing approach has not crossed the product gate. On the
eight-question, agent-authored learning-loop development pilot, the leading Qwen3
v5 adapter had **zero full passes** (five partial, three fail), versus base Qwen3's
zero/ four/ four and Sonnet 5's three/ three/ two. The result is too small and
self-reviewed to estimate a population effect, but it gives no basis for another
QASPER-style answer SFT. In the ten-question paper pilot, one-pass RAG scored
0.70/2 in source-visible review against iterative Envoy's 0.60/2 after retrieved
paper IDs were supplied. Candidate retrieval already covered 12 of 13 required
papers. Those observations argue against claiming that *more within-paper search*
alone will repair final synthesis. See [the model sweep](LEARNING_LOOP_MODEL_SWEEP.md)
and [RAG comparison](AI_PAPER_RAG_VS_ENVOY_EVAL.md).

The new task must therefore test **open-corpus discovery**. A six-paper pilot with
known candidate IDs is not a meaningful retrieval benchmark. Base Qwen3 already
had high citation precision/recall on that small pilot; there may be little
retrieval headroom there. A library with thousands of papers and several required
sources creates the failure mode that retrieval training can actually address.

Published work gives unusually direct support for this objective:

| Primary result | What it supports | What it does **not** prove |
| --- | --- | --- |
| [DeepRetrieval](https://arxiv.org/html/2503.00223) trained Qwen2.5-3B on retrieval feedback. On evidence-seeking retrieval, Natural Questions answer-span Hit@1 rose **25.0 → 35.5** (Claude 3.5 Sonnet: 35.7), and TriviaQA **44.4 → 58.4** (Sonnet: 57.1). Its PubMed publication recall rose **6.59 → 65.07**. | Qwen weights can gain substantially on a narrow, useful evidence-finding task and approach a stronger model on specific retrieval metrics. | The PubMed headline is **Recall@3,000**, which would be an unusably large evidence packet here. The model generated one query, not a multi-turn Python trajectory, and none of these scores measures digest accuracy. |
| [LeReT](https://arxiv.org/html/2410.23214) trained only a multi-hop query worker. Llama-3-8B two-hop HotpotQA document recall rose **54.7 → 69.8** after supervised context distillation and to **77.1** after preference optimization; with a fixed 70B answer model, exact match rose **41.0 → 52.5**. | Direct retrieval supervision can improve a multi-step small model and materially help an unchanged answerer. The SFT stage alone had a visible gain. | Its domain is Wikipedia, its model is not Qwen, and the main run used about **22,612 questions/105,506 preference pairs** and full FSDP training. |
| [TongSearch](https://aclanthology.org/2025.emnlp-main.1078.pdf) trained Qwen2.5-7B for query reasoning and reported BRIGHT BM25 nDCG@10 **27.9**, versus GPT-4o **26.5**. | Top-ten ranking quality, rather than huge-K recall, can be an attainable small-model specialization. | Its training pairs and parts of BRIGHT derive from StackExchange-like sources; the paper does not establish a clean overlap audit. It does not show Sonnet parity or scientific passage support. |
| [s3](https://aclanthology.org/2025.emnlp-main.1095.pdf) trained a multi-turn Qwen2.5-7B searcher with a frozen answerer. On the paper's QA setup, fixed E5 RAG yielded **48.5%** mean generation accuracy with a Qwen2.5-7B answerer, versus **56.1%** with s3-selected evidence. | Directly tests whether a trained search worker improves a fixed downstream model. Its 2,400 processed training examples are the closest published count to our proposed pilot. | It used **full PPO, five A100 80 GB GPUs, a 1 TB host, a frozen 14B model for reward, and a Wikipedia index**. Its examples were drawn from a 70,286-question pool filtered from 169,615. Its QA metric does not establish passage-level research support. |

These are precedents, not a combined performance prediction. In particular, the
near-Sonnet numbers apply to *retrieval*, never to unrestricted research synthesis.
The project's checkpoint must earn its own same-harness, paper-disjoint comparison.

## Data and competing baselines

[MDA-QA](https://aclanthology.org/2025.findings-emnlp.576.pdf) is the closest
AI-paper source: 6,804 cross-paper questions with typically two or three gold
arXiv papers. Its published retrieval baselines leave room but are nontrivial:
BM25 found all gold papers in the top ten for **21%** of questions; BGE plus
citation information reached **37%**. The
[released questions](https://huggingface.co/datasets/YeloDriver/MDAQA) are CC BY
4.0 and have only one `full` split. Their questions/answers were generated by a
model and a small sample was manually checked, so paper IDs are useful
development supervision, not a complete semantic-support gold standard.

The [SPIQA release](https://huggingface.co/datasets/google/spiqa/blob/main/README.md)
provides extracted paragraphs for roughly 25,000 scientific papers and a 321 MB
compressed paragraph archive. Before adopting it, audit a small import for paper-ID
coverage, table/paragraph quality, expanded size, and paper reuse terms. Do not
silently call all extracted text “complete full text.” The existing personal
Obsidian snapshot has only a welcome note, and the demo snapshot has two short
notes, so neither can establish open-corpus headroom or the final personal-vault
product gate. The repository's experiment rule therefore requires substantive
vault material, named questions, and reviewed baseline outputs before another
costly training run. Public MDA-QA results can establish method feasibility but
cannot substitute for that product baseline.

Every model arm gets the same frozen corpus, search backend, action budget, prompt
contract, and evidence-output format. Compare direct BM25, dense/hybrid retrieval
(including the published BGE-plus-citation idea), base Qwen3-8B, the existing v5
checkpoint as a diagnostic, Sonnet as a scout, and the new trained Qwen. A trained
Qwen win over base alone is insufficient if a non-model retriever gives better
evidence at lower cost. Keep Sonnet-only end-to-end answering as the economic
reference, including Qwen GPU startup and idle time in cost per *supported* answer.

## One bounded experiment

### Provisional first SFT run card (September 26, 2026)

This is a **budget and configuration proposal**, not a claim that this many
examples are required or that the data are ready. The corpus alignment,
personal-vault baseline, indexed search, and evidence-packet submission contract
in the steps below are prerequisites. Freeze the final settings after a small
collection/sequence-length measurement and before the full run.

| Item | Proposed first run |
| --- | --- |
| Model and objective | Fresh pinned `Qwen/Qwen3-8B` (`b968826d9c46dd6066d109eabc6255188de91218`); supervised fine-tuning (SFT) on **complete executed Python search trajectories** ending in at most five ranked exact source passages. The QASPER v5 adapter is a comparison arm, not the initialization. No GRPO, preference optimization, or hidden chain-of-thought targets in this run. |
| Candidate data | Explore **1,000–2,000 distinct accepted training questions/trajectories** from the 6,804-question [MDA-QA release](https://huggingface.co/datasets/YeloDriver/MDAQA), only where gold papers align to usable source text. Sample 2–3 real rollouts per question and keep replayed, source-checked successes. This implies roughly 2,000–6,000 *attempts* and perhaps 3,000–10,000 next-action targets if accepted episodes average 3–5 actions. These are planning assumptions, not measured yields. First measure 50 questions × 2–3 rollouts. |
| Split | Group by gold paper and near-duplicate question before training; aim for at least 100 held-out cross-paper questions if feasible and separately review at least 30 fresh paper/vault needs. Keep gold IDs, source reviews, and split metadata out of student prompts. |
| Training format | One JSONL `{"messages": [...]}` conversation per complete episode: system/tool contract, question, assistant Python action, **actual** user-role tool observation, further assistant actions/observations, final evidence packet. `scripts/train_sft.py` expands each assistant turn to one next-action example and masks all prompt/tool-output tokens from the loss. Verify every action and source offset against the frozen corpus/index. |
| Adapter | QLoRA: frozen 4-bit NF4 double-quantized base, BF16 computation, LoRA rank 4, alpha 8, dropout 0.05 on `q/k/v/o` and `gate/up/down` projections. This reuses the measured configuration rather than changing model size, adapter rank, and task together. |
| Optimizer settings | Propose **2 epochs**, learning rate **1e-4**, microbatch **1**, gradient accumulation **4** (effective batch 4), maximum context **8,192 tokens**, cosine schedule with **5% warmup**, gradient checkpointing, seed **42**, and one checkpoint per epoch. The LR is a conservative *choice* between this trainer's `2e-4` fresh-run default and the prior `5e-5` continuation; it has not been tuned on this task. Preserve epoch 1 so epoch 2 cannot silently replace it if behavior worsens. No LR sweep in the first experiment. |
| Cost basis | The previous 3,711-action, two-epoch continuation took **3 h 20 m** on one 32 GB RTX PRO 4500 (about **$2.40 optimization GPU time** at its historical $0.72/h). Depending on accepted action count and context length, the proposed SFT may take roughly **3–9 optimization GPU hours** on comparable hardware. Data rollouts, corpus setup, and paired inference are additional and may dominate. [Runpod's September 2026 pod list](https://www.runpod.io/pricing) gives A40 48 GB at $0.49/h, RTX A6000 48 GB at $0.53/h, RTX 5090 32 GB at $0.99/h, and A100 80 GB at $1.59/h; account/region availability and price must be checked at launch. Plan **$25–$100 total pod GPU allowance** for one pilot, including collection and evaluation, with a measured 50-question collection trial before setting a hard spending cap. Paid teacher API calls, human review, and persistent storage are separate. |

An illustrative row is `question → search() → actual results → revise the query
or read()/passage() → actual source text → ranked {paper ID, start, end} packet`.
Gold paper IDs may be used to *grade* a training rollout but are never shown to
the policy. A direct lookup of a supplied gold title would teach a shortcut,
not adaptive evidence discovery. The current answer-style `SUBMIT` parser and
MuSiQue reward do not implement this packet objective.

The first 50-question collection measurement should sample base-Qwen rollouts
without a paid teacher. If their verified success yield is too low, estimate a
separate capped teacher-data cost before adding Sonnet or another hosted model;
do not treat that API spend as part of the pod GPU allowance. Pin the optimizer
and installed trainer/library versions in the final run manifest rather than
silently relying on whatever defaults happen to be installed on a new pod.

The **expected signal** is a gain in all-required-paper coverage in at most
five passages and more fully supported answers from the same frozen host model.
The **promotion target** is at least **+10 absolute percentage points** in
small-packet all-gold-paper coverage versus base Qwen, paired wins exceeding
losses on human-reviewed supported answers, and no unacceptable regression in
execution failures, latency, or cost. This is a decision threshold, **not a
forecast**. Simple retrieval/reranking is an additional economic control: if it
provides the same evidence at lower cost, the Qwen search policy is not worth
deploying. Do not launch a second training algorithm automatically if the first
SFT run fails.

**Hypothesis.** With the identical executable search interface and budget, Qwen3-8B
trained on successful, source-checked search trajectories will retrieve more of
the papers and passages needed for cross-paper research than base Qwen3-8B. A
fixed host model will consequently produce more fully supported answers.

**Expected signal.** Higher all-required-paper recall at a small packet size,
better relevance of selected passages, fewer redundant or empty searches, and a
paired gain in host answers supported by those passages. A mere increase in
submission rate, valid offsets, or Recall@3,000 does not count.

**Before training:**

1. Audit a small SPIQA import and choose a meaningful AI-paper corpus. Freeze its
   revision and search index. Build disjoint train/development/test groups by
   paper and near-duplicate question cluster; inspect the support-paper graph so
   a paper cannot appear on both sides through a different question. Aim for at
   least 100 held-out cross-paper questions if the grouping permits; publish the
   actual IDs and any reduction before reading model outputs.
2. Freeze named, reviewed cross-paper questions and expected sources for the
   research-library/personal-vault task. Use a separate fresh-paper subset of at
   least 30 human-reviewed needs to check passage support and the resulting
   digest answers; this is the product check, not another model-selection set.
   Run base
   Qwen, simple retrieval, and Sonnet on this exact setup. Record prompt, model
   revision, decoding, corpus hash, IDs, seed, hardware, latency, and cost.
3. Continue to training only if there is a recurring *search-policy* gap that
   Qwen actions can plausibly fix. If a simple retriever already supplies enough
   evidence but the host still writes unsupported summaries, this task choice
   does not address the observed failure. If the corpus lacks the needed text,
   repair ingestion rather than train a model to guess it.

**One initial low-resource pilot, if the baseline supports it:** start from
`Qwen/Qwen3-8B`, not the QASPER-specialized v5 adapter. On roughly 1,000–2,000
training questions, sample a few real code-execution searches per question under
the same tool interface; score retrieval against training-only gold paper IDs;
retain runnable trajectories that find the sources without ever exposing gold IDs
to the policy. Verify chosen passage offsets and audit a sample for semantic
relevance. **The 1,000–2,000 range is an exploratory budget, not a supported
sample-size prediction; measure attempts, action tokens, and search latency on a
small collection pilot before fixing it.** Train one action-masked QLoRA SFT
adapter only if the resulting trajectories and baseline justify it. This is a
different optimizer, domain, and data scale from LeReT and s3. Preference or GRPO training is
**not** an automatic next step; it needs a documented SFT failure and a trustworthy
direct retrieval reward first. `src/env/reward.py` remains the MuSiQue reward and
must not be reused as a research-support reward.

**Decision rule.** On the frozen, never-trained-on questions, promote the adapter
only if it beats base Qwen on human-reviewed, *supported-answer* rate with paired
wins exceeding losses, improves small-packet all-gold-paper coverage by a
meaningful margin (target at least 10 absolute percentage points), and does not
introduce unacceptable execution failures or cost per supported answer relative
to simple retrieval and Sonnet. Report uncertainty and per-question outcomes; do
not claim a win from a few cases or a numeric target alone. If these conditions
fail, stop this training lane and publish the negative result rather than making
more arbitrary eval sets or automatically launching GRPO.

This experiment is complete only when a useful weekly Obsidian digest and MCP
query path also work. A retrieval score by itself is not the project's product
gate. See [the active product contract](WEEKLY_RESEARCH_RADAR.md).

## Why the other tasks rank lower now

**AI-paper experimental-result extraction** is the strongest *distinct product*
alternative: Qwen would turn a new paper's methods, datasets, scores, baselines,
and ablations into evidence-linked result cards. [AxCell](https://aclanthology.org/2020.emnlp-main.692/)
established production value; [MetaLead](https://aclanthology.org/2026.eacl-long.196.pdf)
has 3,568 human-curated result tuples from 43 ML papers and difficult open-domain
complete-tuple scores (o4-mini F1 **29.99**, Llama-3.3-70B **25.78**). Direct
adjacent Qwen evidence exists: [NSF-SciFy](https://aclanthology.org/2026.acl-long.2118.pdf)
raised Qwen2.5-7B claim-extraction F1 from approximately **0.400 to 0.654** on
grant abstracts. But that was single-pass extraction from teacher-labeled
abstracts, not full-paper numeric result extraction with code tools. A separate
[leaderboard study](https://arxiv.org/html/2408.10141) found that even extensive
fine-tuning failed to extract numeric scores when its compressed paper input did
not contain them. MetaLead has no span gold and restricts its dataset to
noncommercial research use. This is valuable work, but the immediate
base-to-trained Qwen bet is less direct than evidence discovery. Revisit only
after the chosen experiment reaches its stop condition.

**Citation/claim auditing** is also useful and clearly unlike QASPER: inspect a
draft claim against cited source text, find overstatements, and suggest repairs.
The [Citation Integrity](https://academic.oup.com/bioinformatics/article/40/7/btae420/7699794)
authors human-labeled 3,063 citations and found **39.18%** erroneous. Yet the
best trained retrieval/verifier system achieved only **0.52 macro-F1**, and the
strongest large apparent small-model gain in adjacent work
([MisSynth](https://arxiv.org/html/2510.26345)) was for classification with the
relevant excerpt already supplied. The full search-plus-semantic-repair task has
weaker evidence for a large Qwen improvement, and exact quote matching cannot
serve as a support judgment.
