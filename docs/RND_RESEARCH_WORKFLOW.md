# How an R&D engineer turns new research into a decision

## Research question and scope

What should a research assistant help an ML/R&D engineer do after a new paper appears? This note synthesizes a qualitative study of data scientists, expert guidance on reading and evaluating technical papers, ML experimentation guidance, and learning research. It is a product-design argument for Envoy, not a claim that one universal workflow has been experimentally proven best.

## What the evidence says

| Finding | Evidence and limit | Design implication |
| --- | --- | --- |
| Engineers read with a purpose: understand a field, generate ideas, find applicable methods, and establish comparison baselines. | A [study of 20 academic and industry data scientists](https://arxiv.org/html/2301.03774v1) found that 17 sought methods they could apply. The sample is qualitative and should not be treated as a population estimate. | Start from the engineer's current problem and projects. Show how a paper might change a decision. |
| Discovery is both active and passive. People search by keyword, follow citations, track authors, and receive recommendations. | In that same study, 17 used keyword searches, 15 followed citations, and 13 searched authors' other work. | Combine topic alerts with backward/forward citation trails, known authors, and a user-controlled shortlist. |
| Most papers deserve triage before deep reading. Similar claims, missing details, and uncertain credibility slow researchers down. | All 20 study participants skimmed papers; 16 reported overload from similar papers; 13 described difficulty judging credibility. [Source](https://arxiv.org/html/2301.03774v1). | Give a brief first-pass card and reserve deeper analysis for a few papers. Do not treat venue, citations, or a model-generated score as proof. |
| The key intellectual task is finding the *delta*: how a paper differs from earlier work and whether its experiment establishes the advertised claim. | Nine of the study's participants specifically discussed this challenge. [Source](https://arxiv.org/html/2301.03774v1). A [CS paper-reading guide](https://www.cs.cmu.edu/~15712/papers/reading-levis.pdf) recommends multiple passes and critical examination of experiments. The guide is expert advice, not a controlled study. | For each shortlisted paper, compare method, dataset, baselines, metric, costs, and limitations with the nearest prior approach. |
| Papers alone may omit implementation details. Engineers use code, talks, blogs, forums, and peers to understand them. | The [data-scientist study](https://arxiv.org/html/2301.03774v1) observed these behaviors. | Link supplementary context, label its provenance, and keep informal commentary separate from the paper's own evidence. |
| For a working ML system, experimentation continues after the first successful result; evaluation sets, versioning, and monitoring matter. | [Interviews with 18 production ML engineers](https://arxiv.org/abs/2209.09125) found continual data collection, experimentation, evaluation, and monitoring. This study concerns production ML and does not prescribe how every R&D lab reads papers. | Keep paper notes connected to local experiment versions and later outcomes, so a failed idea does not re-enter the digest as untested novelty. |
| Adopting an idea needs a local baseline and a controlled test. | [Google's ML experimentation guidance](https://developers.google.com/machine-learning/managing-ml-projects/experiments) recommends a baseline, one small change at a time, and recording failures. The [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist) asks for claims that match evidence, experimental details, baselines, and uncertainty. | End with a proposed small experiment: hypothesis, control, metric, budget, and stop rule. Mark it as our proposal, not a result from the paper. |
| Durable learning benefits from retrieving and using knowledge, beyond rereading summaries. | A [review of retrieval-practice research](https://pubmed.ncbi.nlm.nih.gov/33006925/) finds that recalling information can improve later recall. Most evidence comes from educational settings, so its effect on working R&D engineers is an inference. | Let the engineer ask questions across saved notes, recall prior conclusions, and revisit decisions after experiments. Do not turn this into obligatory flashcards. |

## Proposed operating loop for Envoy

1. **State the live problem.** Keep a short, editable project profile: task, present baseline, constraints, open questions, and excluded topics. A weekly radar needs this context to rank papers for *this* engineer.
2. **Scan broadly; filter cheaply.** Fetch new and revised papers from selected sources, deduplicate by stable ID/version, and include citation or author trails from known strong papers. Show a small shortlist with a one-sentence relevance reason and an uncertainty flag. A feed of titles is not the output.
3. **Read shortlisted work in passes.** First capture the claim and method. Then inspect setup, baselines, ablations, failure cases, and code. Finally check whether the reported result supports the claimed improvement and whether the setting resembles ours. Preserve exact passages for material claims.
4. **Place it in the existing map.** Answer: what was known; what changed; what conflicts with another paper or our own run; what is still unknown. Link the new note to prior paper notes and experiment records. Do not merge conflicting claims into a bland consensus.
5. **Translate into a decision.** Choose **ignore**, **watch**, **read deeply**, or **test locally**. For a proposed test, write the smallest useful control experiment before coding: hypothesis, expected signal, matched baseline, dataset/split, budget, and stop condition.
6. **Run, record, and update.** Save the local result, including negative results and implementation differences, next to the literature note. Change the conclusion only when the evidence changes. Future digests should know that a paper was already tested here.
7. **Retrieve later.** Through MCP, a host assistant should answer questions such as “Why did we reject extra SFT?” using the original paper evidence *and* this project's experiment record. The answer should distinguish external findings from our observations and recommendations.

These steps are a proposed synthesis from the sources above. The precise ranking and review thresholds need evaluation with the user.

## The artifact to build: an evidence-to-decision card

For every paper worth saving, the durable Obsidian note should have these fields:

| Field | Example prompt |
| --- | --- |
| Identity | Which version, date, source URL, code, and dataset? |
| Claim | What does the paper actually claim, with an inspectable passage? |
| Delta | What changed from the closest prior approach? |
| Evidence | Which comparison, ablation, or failure case supports the claim? What is missing? |
| Transfer | Which assumptions match or fail in our code, data, hardware, and user task? |
| Decision | Ignore, watch, read deeply, or test locally; why? |
| Local test | Hypothesis, control, metric, budget, decision rule, and status. |
| Provenance | Clearly label **paper result**, **external commentary**, **our inference**, and **our measured result**. |

The weekly digest should summarize only the highest-value cards: **what changed, why it matters to current projects, evidence and caveats, and what action is worth taking**. For example, if a new paper proposed training an abstention behavior, Envoy should compare its setting with our observed Qwen over-answering regression, then suggest a matched test on our frozen questions. The paper's reported result, our measured regression, and the proposed test must appear as three distinct things.

## How to test this product claim

Replay two or three historical weeks for a narrow topic. Have the engineer mark a bounded candidate pool for relevance *before seeing Envoy's ranking*. Compare a plain chronological feed, a simple keyword/metadata shortlist, and Envoy's digest. Measure important-paper recall, shortlist precision, correctness and support of claims, whether the “delta” and caveats are accurate, time to a read/test decision, and whether the engineer actually uses a saved note later. Collect qualitative feedback on missed papers and unwanted recommendations. Do not optimize only for clicks or the number of notes produced.

Separately, compare base and trained Qwen on the same held-out code-execution questions over frozen paper/vault snapshots, with blind supported-answer review, citation checks, execution failures, latency, and cost. A working weekly digest does not prove model improvement; an improved Qwen score does not prove a useful digest. This two-gate distinction is the [active project plan](WEEKLY_RESEARCH_RADAR.md).

## Limits

The closest direct studies are small and observational. Reading guides and ML engineering documents are expert guidance rather than causal evidence about the best personal workflow. Learning-science findings on retrieval practice do not directly measure professional research decisions. The product design above is therefore a testable hypothesis, and the first useful validation is a real engineer judging real weekly decisions.
