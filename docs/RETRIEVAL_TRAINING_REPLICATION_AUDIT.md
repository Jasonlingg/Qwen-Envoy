# Retrieval-training papers versus Envoy's actual resources

Audit dated September 26, 2026. This checks whether the published training
results can be reproduced with the **data, compute, search system, and objective
currently in this repository**. It does not authorize a new GPU run.

**Scope correction:** the multi-GPU setups below describe faithful replication
of these papers, not the minimum hardware for any search-agent improvement.
[Independent small-compute projects](LOW_COMPUTE_SEARCH_AGENT_CASE_STUDIES.md)
report meaningful gains from Qwen search training on one 24–32 GB GPU. Those
results make a narrow, measured experiment credible, although none proves a
gain on our code-execution scientific-paper task.

## Finding

We can train a Qwen3-8B **QLoRA adapter** on one rented GPU. We cannot claim to
replicate DeepRetrieval, LeReT, TongSearch-QR, Search-R1, or s3 at their published
scale with the current data and infrastructure. The closest *data-count* precedent
is s3's 2,400-example run, but that was full PPO with five A100 80 GB GPUs, 1 TB
of host RAM, a frozen 14B answer model for the reward, and an indexed Wikipedia
corpus. Our proposed 1,000–2,000-question action-masked SFT is a new low-resource
experiment, not a reproduction or a result that these papers establish.

| Study | Actual training data and method | Published compute / search | Result relevant to us; important limit |
| --- | --- | --- | --- |
| [DeepRetrieval](https://arxiv.org/html/2503.00223), [code](https://github.com/pat-jj/DeepRetrieval) | Qwen2.5-3B-Instruct actor **and** critic, full PPO. PubMed: 12,801 labeled training questions, 10 epochs, about 128,000 configured query rollouts. NQ: 79,168 questions, five epochs. Reward checks retrieved gold publication IDs or answer strings; no ideal-query labels required. | **2 A100 80 GB + 1 TB RAM**. PubMed API or a 21-million-passage Wikipedia index. Total wall time and dollar cost not reported. | PubMed recall **6.59% → 65.07% among top 3,000**, and NQ answer-span Hit@1 **25.0% → 35.5%**. It emits **one query** and cannot adjust after seeing a search result. Neither metric is small-packet research support. |
| [LeReT](https://arxiv.org/html/2410.23214), [code](https://github.com/sher222/LeReT) | Main HotpotQA run: approximately **22,612 questions**, **105,506 preference pairs**; four sampled queries at each of two hops imply roughly **181,000 query/retrieval attempts** before prompt optimization. Context-distillation SFT for one epoch, then IPO for two; full-model FSDP. Direct reward uses gold supporting-document average precision. | ColBERTv2 Wikipedia-2017 abstracts index (released archive about 11.3 GB), DSPy/prompt sampling, TGI or Together inference. **GPU count and wall time not reported**, so an exact cost comparison is impossible. | Two-hop document recall **54.7% → 69.8% (SFT) → 77.1% (IPO)**; a fixed 70B answerer reached **41.0% → 52.5%** answer EM. Strong multi-hop precedent, but its labeled preference pipeline and full-model training are far beyond our current data. |
| [TongSearch-QR](https://aclanthology.org/2025.emnlp-main.1078.pdf), [code](https://github.com/bigai-nlco/TongSearch-QR) | Roughly **10,000** teacher-generated or **30,000** noisier StackExchange questions; full-parameter GRPO with **16 completions per prompt**. BGE similarity to known positive text is the training reward, without a live corpus search in that reward. | **4 A800 80 GB** for **16 h (1.5B)** or **48 h (7B)**, about 64 or 192 A800 GPU-hours. | Trained 7B reaches **27.9 nDCG@10** on BRIGHT with BM25. The paper does **not** give a same-base-Qwen rewrite control, so the 7B base-to-trained gain is unknown. One-shot query rewrite, not a multi-step code agent. Released default script uses 1.5B/V1/ZeRO-2, not the 7B headline configuration. |
| [Search-R1](https://arxiv.org/html/2503.09516), [code](https://github.com/PeterGriffinJin/Search-R1) | Qwen2.5-3B/7B interleaves searches and answers; retrieved text is masked from training loss. Released NQ/HotpotQA set has **169,615 training questions**. Published setup: **500 steps × 512 prompts**, up to four actions; GRPO uses five responses per prompt. Reward is final-answer exact match. | **8 H100 GPUs**; E5 retriever over Wikipedia-2018. Released FAISS index is about **64.6 GB**, with a further **5.1 GB compressed corpus**. Hours/cost not reported. | Closest precedent for the *interaction loop*. Reported Qwen2.5-7B-Base mean QA exact match **0.304 with RAG → 0.431 trained**. It does not train citation or research-answer support. Released training scripts differ from the paper's 500-step setting, so the published setup must be reconstructed explicitly. |
| [s3 / Search-Select-Serve](https://aclanthology.org/2025.emnlp-main.1095.pdf), [code](https://github.com/pat-jj/s3) | Qwen2.5-7B-Instruct generates up to three rounds of queries, selects documents, and stops. Frozen 14B model answers from selected documents. PPO reward is **answer accuracy with this packet minus answer accuracy from fixed top-k RAG**. The final run processes **2,400 examples** in 20 steps × batch 120, after first filtering a **169,615-question pool to 70,286** questions that naïve RAG misses. | **5 A100 80 GB + 1 TB RAM**, including dedicated 14B answer-model serving; E5/PySerini/FAISS over Wikipedia-2018. Reported training time **114 min**, or about **9.5 A100 GPU-hours** for the five-GPU job, excluding the rest of the data/evaluation pipeline. | Best match to a *search-worker plus frozen answerer* architecture. With a fixed Qwen2.5-7B answerer, mean QA generation accuracy was **48.5% for static E5 RAG versus 56.1% for s3**; the paper does not supply a directly comparable untrained-searcher arm under the same s3 controller. It does not measure scientific passage support. |

These are different tasks and denominators. DeepRetrieval's top-3,000 recall
cannot be compared with our desired five-passage packet; LeReT and Search-R1
mostly measure Wikipedia QA; TongSearch measures BRIGHT ranking; s3 measures
downstream short-answer correctness. None shows that 1,000–2,000 QLoRA code
trajectories can make Qwen3-8B produce supported AI-research answers.

## What we actually have

| Resource | Verified local state |
| --- | --- |
| Model training | `Qwen/Qwen3-8B` rank-4, 4-bit QLoRA works on one GPU. A fresh-from-base 35-conversation/106-action run took **579 s** of optimization on one A40 ([manifest](../out/research/qwen3-qasper-v5-sft-20260919/artifacts/full/run-manifest.json), [summary](../out/research/qwen3-qasper-v5-sft-20260919/summary.json)). A larger **995-conversation / 3,711-train-action** continuation took **3 h 20 m** on one 32 GB RTX PRO 4500 ([handoff](../CODEX_HANDOFF.md)). That time excludes generating, executing, scoring, and reviewing multiple search trajectories. |
| Model outcome | The large QASPER continuation **regressed** on its reserved 40 questions: starting adapter 15 passes, epoch 1 ten, epoch 2 four ([handoff](../CODEX_HANDOFF.md)). Low validation loss is insufficient evidence of autonomous improvement. |
| Search-policy data | The 995 examples were derived from **known-paper QASPER answer annotations**. We have **zero imported MDA-QA search-policy trajectories**, no verified large-library evidence packets, and no reviewed personal-vault baseline. The six-paper and 20-paper AI research sets, one-note personal vault, and two-note demo do not provide a paper-library retrieval test. The 10,340-document MuSiQue corpus is Wikipedia, not an AI-paper library. |
| Search path | Active Python [`search()`](../src/env/tools.py) loads all document JSON files, builds IDF across them, and scans every document or cached chunk on each call. The separate research path likewise scans text; the FAISS component is not the active code-agent search. There is no measured 25,000-paper indexed `search()` backend or per-query latency budget. |
| Objective | Active [`reward.py`](../src/env/reward.py) is the MuSiQue outcome reward: **0.8 answer token-F1 + 0.1 citation precision + 0.1 citation recall**. It is not a research-support judgment, document-AP signal, or s3-style gain-over-static-RAG signal. The current final-answer submission contract is not yet the proposed bounded evidence packet. |
| Runpod now | The old A40 and RTX PRO 4500 are **historical stopped-run records**, not verified active hardware. This session has no connected Runpod MCP, CLI, usable API key, or browser, so live pod state cannot be asserted. [Runpod's public pricing](https://www.runpod.io/pricing) lists Secure Cloud A100 80 GB at **$1.59/GPU-hour** (two start at $3.18/h before other charges), with standard listed host RAM of 117–125 GB per GPU; that does not establish availability of a 1 TB host. Current account-specific capacity and prices remain unverified. |

[MDA-QA](https://aclanthology.org/2025.findings-emnlp.576.pdf) is available
externally: **6,804 generated cross-paper questions** over more than 25,000
machine-learning papers, with source paper IDs. Its [Hugging Face release](https://huggingface.co/datasets/YeloDriver/MDAQA)
has a single `full` split, so it is **not already a train/test partition**. The
authors' manual check covered 30 questions, not every answer or supporting span.
The [SPIQA corpus release](https://huggingface.co/datasets/google/spiqa) could
provide extracted text, but we have not imported it, measured MDA-QA ID
coverage, or verified that its paragraphs contain the required evidence. Gold
paper IDs could support paper-recall training; they do not by themselves certify
which passage supports each claim. A paper-disjoint, near-duplicate-aware split
and human passage review are therefore needed before making support claims.

## Decision for this project

**Exact paper replication:** no, given the current one-GPU history, unimported
corpus, missing search index, and unmatched rewards. Renting the same number of
GPUs and importing the original Wikipedia/PubMed datasets could make selected
paper benchmarks possible, but it would be a separate reproduction project and
would not establish a useful weekly research radar. For LeReT, even the published
GPU-hours are unavailable.

**Low-resource adaptation:** plausible as an experiment, not a predicted gain.
s3 supplies the strongest reason to test a *multi-step evidence worker with a
frozen answerer and a fixed-retrieval control*. Its low example count does **not**
show that QLoRA SFT on our scientific corpus will work: it used PPO, stronger
parallel hardware, a hard-question selection pass, and answer-aware reward.
Search-R1 supports masking tool output in a multi-turn loss, while LeReT
supports direct retrieval feedback. We can borrow those design principles, but
must measure them under our own fixed task and budget.

The next material experiment is **a small, read-only data and retrieval audit**,
not GPU training. Hypothesis: enough MDA-QA support papers and actual paragraphs
can be aligned into a paper-disjoint scientific corpus, and fixed retrieval leaves
a recurring gap that a multi-step Qwen search policy could fix. Expected signal:
verified corpus coverage, measured fixed-query all-gold-paper recall at a small
packet size, and named base-Qwen failures caused by search decisions rather than
missing text or answer writing. Decision rule: proceed to a *budgeted* data
collection pilot only if the corpus includes the required text and base Qwen
misses evidence that an alternative search can find; otherwise repair ingestion,
retrieval, or the host-answer stage. Before any costly run, satisfy the repo's
personal-vault baseline requirement and estimate total attempts, supervised
action tokens, index memory, GPU-hours, and evaluation cost from a measured pilot.
The eventual trained-vs-base claim still requires supported-answer improvement
on the same held-out code-execution tasks with acceptable execution failures,
latency, and cost, plus a useful Obsidian digest and MCP query path.
