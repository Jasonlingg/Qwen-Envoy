# Weekly AI learning loop

## Product direction

The user's September 25 direction is to **keep learning**, including recent AI/ML research that is interesting beyond the current Envoy experiment. The weekly research radar should become a personal learning loop: discover a few worthwhile ideas, understand how they work and what evidence supports them, connect them with prior knowledge, recall them later, and optionally apply one in a small test.

The weekly digest and Obsidian library remain the durable interface. The larger assistant queries that library through MCP. Qwen remains the bounded, code-executing research worker, and trained Qwen must still beat base Qwen on held-out supported-answer quality. A good reading experience alone does not satisfy the model-improvement gate.

## Why this shape

In [interviews with 20 data scientists](https://arxiv.org/html/2301.03774v1), participants used literature both to track a field and to solve current problems. They skimmed broadly, compared similar papers, and sought help from code, talks, and colleagues when a paper omitted context. This supports a two-lane product: **explore a topic** and **investigate a project question**. The study describes behavior; it does not prove this product will improve learning.

A [review of retrieval-practice research](https://pubmed.ncbi.nlm.nih.gov/33006925/) supports recalling material after learning it, although its evidence does not directly measure professional R&D work. Use lightweight recall prompts and later questions as a testable learning feature, not as a compulsory quiz system. For application, [ML experimentation guidance](https://developers.google.com/machine-learning/managing-ml-projects/experiments) supports baselines, small changes, and recording failures.

## The loop the user sees

1. **Choose a direction.** The user can maintain a few interests, such as tool-using agents, small-model training, and evaluation. Each can have a short “what I know / what I wonder” note. Project constraints are optional for curiosity-driven topics.
2. **Get a weekly shortlist.** The system scans newly published or revised work, deduplicates versions, and offers about three candidates: one closely relevant to a current interest, one that changes or challenges a familiar idea, and optionally one adjacent discovery. Each recommendation says why it appeared and what remains uncertain. These counts are MVP defaults, not research findings.
3. **Learn one idea deeply.** A selected paper gets a progressive explanation: a one-minute summary; the problem and mechanism; the difference from prior work; the experiment and limitations; and links to exact source passages. Claims from the paper, outside commentary, and the assistant's interpretation are labeled separately.
4. **Connect it.** The assistant compares the idea with saved paper notes and the user's own conclusions. It highlights agreements, disagreements, prerequisites, and unanswered questions. The user can correct the explanation or add a personal insight.
5. **Recall and use it.** The user can answer two or three short questions in their own words, ask follow-up questions through MCP, or choose a small project test. A later weekly note revisits an unresolved question or earlier decision. Applying an idea is optional; understanding it is a valid outcome.
6. **Update the library.** Save a weekly digest, durable paper/concept notes, the user's corrections, and any local experiment result in Obsidian. Keep source evidence and personal interpretations distinct. Later answers should retrieve both the original source and what the user learned or measured.

The interaction should feel like a knowledgeable research partner, not a homework app. “Skip,” “show me the method,” “compare with an older paper,” and “why did you recommend this?” are first-class actions.

## Smallest honest demo

Use one topic and one frozen week of papers. A first version can rely on a manually reviewed candidate pool while discovery and scheduling are built. Show three recommendations, one deeply explained paper with inspectable evidence, a comparison with one existing note, a short follow-up question, and the saved Obsidian digest. Then query that saved evidence from a host assistant through MCP. The current personal-vault snapshot contains only `Welcome.md`, so it cannot yet demonstrate cross-note learning; seed a substantive small collection before claiming that feature works.

The first example topic can be **tool-using AI agents and small-model evaluation**, because the repository already contains relevant papers and local Qwen experiments. This is a default for a pilot, not a permanent limit on what the user can study.

## Evaluation and decision rule

**Product hypothesis:** a short, evidence-backed weekly learning loop helps the user understand and reuse more relevant research than a chronological paper feed.

**Expected signal:** in two or three frozen historical weeks, the user marks the shortlist useful, can explain at least one selected idea accurately in their own words after a delay, can retrieve its evidence later, and can identify when a proposed application is only an inference. Compare with a plain recent-paper feed and record missed important papers, unsupported claims, time spent, and user corrections. Decide whether to continue based on those reviewed outcomes rather than note count or clicks. If the shortlist is weak, fix discovery and topic profiles; if explanations are weak, fix evidence gathering and synthesis; if later recall is weak, test the follow-up interaction before adding more content.

**Model hypothesis:** trained Qwen produces more supported, multi-step evidence packets than base Qwen on the same held-out code-execution tasks that feed this workflow. Compare checkpoint, corpus revision, question IDs, prompt, decoding, seed, hardware, execution errors, latency, and cost. Do not run another costly training job before a reviewed personal-vault baseline reveals a specific failure and a decision rule.

This plan extends the [weekly research radar](WEEKLY_RESEARCH_RADAR.md) and the [R&D workflow research review](RND_RESEARCH_WORKFLOW.md). It does not claim that the learning loop, scheduler, Obsidian writer, or MCP server has already shipped.

The first frozen, source-checked development task for Qwen is in the [learning-loop code pilot](LEARNING_LOOP_CODE_PILOT.md). Its eight questions test explanation, comparison, evidence critique, local application, and uncertainty. Base and targeted-SFT Qwen were run on September 26; Codex's blind development review tied them on answer quality, while SFT was much slower. Independent human review and a representative personal-vault test are still pending.
