# Second-brain landscape and Envoy positioning — September 26, 2026

## Recommendation

Build **Envoy as a learning debugger for an Obsidian vault**. The user asks in
chat: “What did I believe about this, what evidence or attempt changed it,
what happened when I applied it, and what should I try next?” Envoy returns
an inspectable, dated trail, not just a summary of stored notes. An approved
revision becomes a linked Markdown note; the older view remains visible.

Start with research and technical learning, where the current Qwen checkpoint's
paper-investigation training is most relevant. The vault may contain homework,
life lessons, and other subjects, but do not imply that v5 is proven on those
domains. “Apply it” can mean a small experiment, a practice problem, or a
decision followed up later; it need not mean running code or making a formal
study of everyday life. Chat remains the primary interaction, with notes and
Obsidian's graph as the durable, inspectable record.

This is a **specific product focus**, not a claim that chat, graphs, learning
quizzes, temporal memory, or a Qwen memory worker are new inventions. The
sources below document advertised capabilities; they are not a hands-on
comparative usability study.

## What already exists

| Project | Relevant existing capability | Consequence for Envoy |
| --- | --- | --- |
| [Copilot for Obsidian](https://github.com/logancyang/obsidian-copilot) and [Khoj](https://github.com/khoj-ai/khoj) | Multi-step agents or chat over vault material, search, writing, Obsidian access, and model choice; Khoj also supports automations and Qwen | A chat sidebar, search, or multi-agent wrapper is not a distinct product claim. |
| [Smart Second Brain](https://community.obsidian.md/plugins/smart-second-brain) and [Smart Connections](https://community.obsidian.md/plugins/smart-connections) | Chat/search with a vault, semantic or topic connections, and graph-oriented exploration | A prettier graph or semantic link suggestion is insufficient differentiation. |
| [Recall](https://www.recall.it/about), [RemNote AI Tutor](https://help.remnote.com/en/articles/10103884-ai-tutor-chat), and [Gemini/Notebook study notebooks](https://support.google.com/gemini/answer/16972047?hl=en) | Chat about saved sources, active-recall quizzes, spaced study, and learning progress; Recall also has an automatic knowledge graph | “Lifelong learning,” flashcards, and spaced review are established features. Measure real follow-through instead of claiming novelty here. |
| [Heptabase AI](https://wiki.heptabase.com/work-with-ai) and [Capacities AI](https://docs.capacities.io/reference/ai-assistant) | AI conversation across visual whiteboards, journals, PDFs, or a connected knowledge space, with traceability and graph-aware access | Cross-source visual synthesis and citations are also established. |
| [Hindsight](https://github.com/vectorize-io/hindsight) and its [Obsidian integration](https://github.com/vectorize-io/hindsight-obsidian) | Retain/recall/reflect, temporal and graph retrieval, evolving observations with supporting evidence, and Obsidian chat | Even evidence-backed belief updates and temporal memory have close prior art. |
| [Dria mem-agent](https://huggingface.co/blog/driaforall/mem-agent-blog) | A Qwen3-4B model trained with RL to use Python tools over Obsidian-like Markdown for memory retrieval, update, and clarification; public [code](https://github.com/firstbatchxyz/mem-agent) and [checkpoint](https://huggingface.co/driaforall/mem-agent) | “Fine-tuned Qwen + Markdown tools” is close prior art. Its reported gain is on its own 56-case scaffold, not evidence for Envoy's checkpoint. |

The **inference** from this landscape is that Envoy should be judged on a
particular recurring job: connecting what a person read or heard to what they
believed, tried, observed, revised, and later used. Existing products may be
configurable to do parts of this. The claim is that Envoy makes the trail and
its reliability central, visible, and measurable, rather than that no other
system could reproduce it.

## The smallest distinctive workflow

1. **Ask in chat.** “I thought iterative retrieval would beat BM25. What did
   we actually find, what is still untested, and what should I compare next?”
2. **Investigate the trail.** Search current and older concept notes, papers,
   hypotheses, experiment logs, corrections, and decisions. Follow dated
   `derived_from`, `tests`, `contradicts`, and `supersedes` relationships when
   present. A link is a candidate path, not proof of a claim.
3. **Show the evidence.** Return exact passages and Obsidian links for the
   earlier belief, attempted test, observed result, and later correction.
   Distinguish a missing outcome from a negative outcome. A source quote's
   existence does not establish that an interpretation is correct.
4. **Talk through the next move.** Nemotron explains the timeline and proposes
   one small application or follow-up question. The user can correct the
   interpretation. Any durable change is a proposed, approved, dated note.
5. **Close the loop later.** Ask what the user remembers before revealing the
   source, then ask whether they actually tried the proposed action and what
   happened. Save the reported result as a separate record, not as an
   automatically inferred success.

In Obsidian, a concept can link to a source, a dated hypothesis, an attempt,
its result, and a revision. Native [internal links and graph
view](https://obsidian.md/help/plugins/graph) display the relationships;
typed meanings live in note content/properties and a deterministic index.
The first build does not need a new graph database or a custom graph renderer.

## Why this fits the actual Qwen checkpoint—and where it does not

The selected model is Qwen3-8B plus the
`jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5` checkpoint-50 adapter, locally
present and hash-matched to the evaluated artifact. Its [44 reviewed QASPER
trajectories](../data/sft/qasper-v5/README.md) teach multi-step Python
`search`/`read`/`extract` actions and short submissions about research papers.
They do not teach personal-memory updating or reliable life coaching.

On the consumed eight-question/six-paper development sweep, v5 submitted
8/8 answers with no execution-error episodes, versus base Qwen3's 5/8
submissions and four error episodes. Yet v5 had **zero fully supported
answers** (five partial, three fail); base also had zero full passes. The
[model sweep](LEARNING_LOOP_MODEL_SWEEP.md) supports a narrow claim about tool
loop completion, not dependable final synthesis or held-out improvement.
The [personal-memory screen](PERSONAL_MEMORY_READINESS_EVAL.md) has not run
base or v5 on its nine-note fixture. A cheap BM25 top-five packet already
contains all 17 required-note occurrences across the seven answerable
synthetic cases. Qwen therefore has to earn its extra inference cost; we
cannot assume it beats search on this vault.

**Give v5 the read-only investigator job:** decide which dated documents to
inspect next, collect a compact set of exact passages, flag candidate
conflicts or missing outcomes, and expose its tool trace. Keep quote-offset
and source-scope verification deterministic. Nemotron would be the
conversational coordinator and propose interpretations; the user approves
persistent revisions. Qwen's packet and the verifier do not themselves
certify semantic support. A future model-improvement claim still requires
trained v5 to beat base Qwen on the same held-out code-execution task without unacceptable
execution, latency, or cost regressions.

The current `search`/`read`/`extract` environment does not have a graph
traversal tool. The existing sample's relationship IDs do not automatically
become Obsidian graph edges. Implement real internal links and a small
deterministic link index before describing graph-aware investigation as a
working feature. Keep the hash-locked nine-note evaluation fixture unchanged;
make a separate, versioned linked demo corpus.

## What would make a credible demo

Show one complete, realistic learning thread: a research claim, a source
paper, a user's initial expectation, an actual test and result, a revised
view, and a new decision. The user asks an implicit question about that
history in chat. The answer distinguishes **source fact**, **personal
interpretation**, **observed result**, and **next proposed test**; every
material historical claim opens its source in Obsidian. The user corrects or
approves a revision, and a later question finds both old and new views.
The weekly research radar can contribute a new paper that challenges an
existing hypothesis, making this an ongoing learning loop rather than a
static notebook demo.

Use the already registered personal-memory readiness screen to decide whether
v5 can be shown as a live investigator. Compare its evidence and downstream
answer with the BM25 and base-Qwen arms under the same frozen corpus and
budget. Then review a small set of **realistic longitudinal conversations**
for date accuracy, contradiction handling, unsupported claims, missing
outcomes, useful proposed follow-up, and user corrections. This is a product
check, not an excuse to create a sequence of new synthetic benchmarks. The
[LongMemEval](https://arxiv.org/abs/2410.10813) axes are useful components;
[LOCOMO-CONV](https://arxiv.org/abs/2609.03467) cautions that memory QA scores
alone can miss conversational failures. Neither benchmark measures whether
the person subsequently learned or applied an idea.

**Decision:** if v5 does not add essential evidence beyond cheap retrieval
or causes unsupported answers, use deterministic retrieval for the product
and keep the checkpoint as an experimental worker. Do not train another
model to justify the product concept before the basic chat, linked evidence,
and user learning thread are useful.
