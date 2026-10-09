# EnvoyBench in context

Scientific-paper QA, tool-using research agents, trace viewers, and even
base-versus-fine-tuned research-agent comparisons all predate EnvoyBench. Its
contribution is an **auditable combination**: QASPER open-ended questions in a
bounded environment where Qwen writes Python against paper tools, with matched
base/adapter runs and separate checks for source spans, answer support,
execution failures, latency, and cost. This is an engineering benchmark
protocol, not a claim to have invented paper-agent evaluation.

| Prior work | What it already established | What EnvoyBench changes |
| --- | --- | --- |
| [QASPER](https://arxiv.org/abs/2105.03011) | Expert-written questions, full papers, answerability and annotated evidence. | Reuses its questions in multi-turn executable episodes; the questions are not original. |
| [Qwen paper-QA LoRA](https://huggingface.co/whosouravsharma/paper-qa-lora) | A public Qwen2.5-1.5B adapter was trained on QASPER and compared with its base model on paper-disjoint QASPER questions. Its author reports citation-format gains alongside factuality and faithfulness regressions under one model judge. | Tests agentic retrieval and code execution as well as the answer; it must still prove a gain under independent review. |
| [PaperQA2 / LitQA2 / Aviary](https://arxiv.org/html/2412.21154) | Literature-search agents and a trained 8B agent compared with its untrained version. | Uses QASPER open-ended answers and model-authored Python over a frozen local paper corpus. |
| [PaperArena](https://arxiv.org/abs/2510.10909) | Multi-step scientific-literature QA, a code executor, and trace/error analysis. | Makes Python the main action interface and audits paired base/adapter changes on the same paper questions. |
| [ResearchQA](https://arxiv.org/abs/2607.11074) | Open-ended cited paper answers, deterministic quotation checks, refusal cases, and answer-quality review. | Measures a bounded tool trajectory in addition to the final answer and spans. |
| [AgentHop](https://arxiv.org/abs/2609.34428) | Scientific-paper questions in a multi-tool sandbox, with retrieval/synthesis and resource diagnostics. | Uses open-ended source-supported answers rather than multiple-choice final scoring. |
| [AstaBench](https://github.com/allenai/asta-bench) | A broad scientific-agent suite with standardized tools, sandboxed code, logs, and cost reporting. | Deliberately narrow: one paper-evidence task and one training intervention that can be audited per question. |
| [Inspect View](https://inspect.aisi.org.uk/log-viewer.html) and [SWE-bench experiments](https://github.com/SWE-bench/experiments) | Per-sample trajectories, scores, metadata, and reproducible run artifacts are established benchmark practice. | The local demo applies that pattern to Envoy's research-agent failure modes. |

The benchmark design borrows three especially useful practices. First, a score
must have a named task set, denominator, model revision, environment, and
scoring provenance. Second, a reviewer should be able to go from an aggregate
number to the exact code action, tool output, answer, and supporting passage on
one question. Third, executing an agent and judging its answer are separate
stages. The [PaperBench implementation](https://github.com/openai/frontier-evals/blob/main/project/paperbench/README.md)
also separates rollout, reproduction, and grading, although its task is
experiment replication rather than paper QA.

## How established agent benchmarks shape this protocol

An agent benchmark needs more than a question list. It needs a task with a
clear success condition, an environment and action budget, controlled tool
access, a scoring procedure, named baselines, and inspectable run records.
[AstaBench](https://allenai.org/blog/astabench) argues for reproducible tools,
cost-aware comparisons, standardized task interfaces, and diverse baselines.
[AgentHop](https://arxiv.org/html/2609.34428v1) shows why a final score should
be decomposed into retrieval, synthesis, tool-use, and resource failures.
[ResearchQA](https://arxiv.org/html/2607.11074v1) keeps deterministic quote
matching separate from answer-quality judgments. These are direct precedents
for EnvoyBench's frozen corpus, bounded executable actions, span diagnostics,
semantic review, and per-question traces. The existing historical review is
model-assisted and therefore only provisional.

The *control* is a matched run, not a different agent wrapper. Base Qwen and
the adapter must see the same questions, corpus snapshot, tools, prompt,
decoding, and step limit; the run records their exact identities, durations,
and code/observation traces. The model answer is then checked at two levels:
whether cited bytes really occur in the source, and whether the resulting
answer is actually supported and useful. The latter needs a reviewer who has
not seen model identities. A strict span match alone can reward irrelevant
quotes; a fluent answer can pass a model judge while inventing support. The
public [Qwen QASPER LoRA model card](https://huggingface.co/whosouravsharma/paper-qa-lora)
is a concrete warning: its author reports more source-formatted answers but
worse factual correctness and faithfulness than base under their judge.

The *unit of analysis* is a paired question. This lets a reviewer inspect a
specific improvement and a specific regression instead of inferring either
from an aggregate. It also supports a paired test of pass-rate changes. The
current 40-question candidate is deliberately balanced for answerability, so
its overall pass rate is a stress-test metric, not an estimate of natural
research traffic. One deterministic run per model cannot establish
run-to-run reliability; [tau-bench](https://arxiv.org/abs/2406.12045) shows
why repeated-trial reliability can matter for tool agents. Repeated runs and
simple retrieval-only or no-tool baselines should follow once the source
references and answer rubric are independently reviewed.

The demo therefore has a frozen-candidate panel with **no score**, an explicitly
historical development result with model-assisted review, and real paired traces
to inspect. A completed EnvoyBench run can replace the historical panel using
`--run-dir`; semantic pass counts appear only when a completed review bundle is
loaded. The visual presentation is useful for debugging and demonstration, but
the viewer itself is not the scientific novelty.

Before publishing a held-out result, the candidate references need independent
review, followed by blinded review of model answers. A simple retrieval baseline
and repeated runs would strengthen the comparison. QASPER is public, so paper
disjointness from local fine-tuning does **not** prove that a pretrained model
never encountered the test papers. The candidate deliberately has equal numbers
of answerable and unanswerable questions; its aggregate rate should be reported
as an abstention stress test, not natural QASPER prevalence.
