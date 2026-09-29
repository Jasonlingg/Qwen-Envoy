# Research-library harness architecture decision

Date: 2026-09-28. Status: proposed design, based on a repository audit and primary-source research. This document does not claim that the complete flow has run.

The build-ready flow, contracts, failure behavior, and diagrams are in [the harness design](RESEARCH_LIBRARY_HARNESS_DESIGN.md).

## Decision

Build a **local Python workflow with one bounded agent loop**. Qwen investigates immutable papers through three read-only tools. Python verifies provenance and controls every transition. Nemotron drafts from a small, verified evidence bundle. Python checks the draft and stages it under the Obsidian vault's `_inbox/`. Only the user moves reviewed notes into `library/`.

This is a two-model system, but it does not need a free-form conversation among agents. The investigation is open-ended; verification, writing permissions, and promotion are not. Anthropic recommends predictable workflows for known stages and agent loops where the model needs to choose the next search step. Its parallel research agents helped independent, breadth-first searches, while adding substantial token use. A 2026 controlled study likewise found gains on parallelizable tasks and losses on sequential ones. Another 2026 study found that single-agent systems matched or beat multi-agent variants on multi-hop questions under matched reasoning-token budgets; it does not directly test this heterogeneous Qwen/Nemotron pipeline. These findings support starting with one investigator and measuring any proposed parallelism on our own tasks. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents); [Anthropic's research system](https://www.anthropic.com/engineering/multi-agent-research-system); [Google Research scaling study](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/); [matched-budget multi-hop study](https://arxiv.org/abs/2604.02460).

| Pattern | When it helps | Fit here |
| --- | --- | --- |
| One unrestricted model with all tools | Short tasks with one trusted context and few side effects | It would obscure Qwen's measured contribution and give the drafter unnecessary vault access. |
| Fixed workflow with a bounded worker | Known stages with one adaptive search subtask | Best initial fit: Qwen can change search strategy, while code enforces verification and writing boundaries. |
| Manager plus parallel research workers | Several independent research branches that can run concurrently | Consider only for broad questions after a matched-budget trial shows better evidence coverage. |
| Peer-to-peer swarm or group chat | Open-ended coordination where handoffs themselves add value | No measured need in a small paper library; more handoffs complicate cost, tracing, and source accountability. |

```text
question or scheduled paper selection
             |
             v
local Python harness ----> append-only run record outside the vault
             |
             v
Qwen investigator -- bounded Python --> sandboxed search/read/extract
             |                              |
             |                     read-only source broker
             |                              |
             v                              v
candidate evidence                 immutable paper store/index
             |                              |
             v                              +--> read-only MCP for external assistants
provenance verifier
             |
             v
verified evidence bundle + trusted question/template
             |
             v
Nemotron draft (no source tools or vault write access)
             |
             v
claim/citation checks --> single staging writer --> vault/_inbox/<run-id>/
                                                   |
                                             user reviews and moves
                                                   v
                                             vault/library/
```

The external MCP and the sandbox tool facade should call the *same* versioned source API. Qwen need not run a general MCP client or hold MCP credentials. A host broker can expose a pinned allowlist of read-only document tools, log their actual inputs and returned passage IDs, and reject any other request. `search`, `read`, and `extract` are the conceptual API; the existing `passage`, `search_within`, and `list_docs` helpers should remain versioned until a paired experiment justifies changing the checkpoint's tool interface. For code-driven tool use, Anthropic reports context and composition benefits but explicitly warns that generated code needs sandboxing, resource limits, and monitoring. Its 2026 agent design separates the harness, sandbox, and durable session log. [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp); [Managed Agents architecture](https://www.anthropic.com/engineering/managed-agents); [MCP tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools).

## Trust and data contracts

1. **Sources:** The paper store is immutable per revision. Each extracted paper has a source ID, publication version, extraction hash, and canonical text. A passage ID is derived from that source version, character offsets, and quote hash. Local `E1` aliases can be used within one answer, but they are not durable library identifiers. `_inbox/` is excluded from all retrieval; only original papers and user-promoted `library/` notes can feed future runs.
2. **Investigator:** Qwen may compose Python and inspect tool results for at most a configured number of actions and tool calls. The sandbox gets no vault mount, source files, API keys, or general network. The broker returns bounded results. Its log, rather than an AST heuristic or Qwen's own claim, establishes which IDs were actually retrieved. The current implementation already has Docker isolation, no network, and a step cap; it still mounts source JSON directly and has no broker ledger.
3. **Evidence bundle:** The verifier rebuilds the bundle from the pinned source revision. Required fields are `run_id`, question, corpus/index revisions, model/checkpoint and tool versions, `passages[{passage_id, source_id, source_version, start, end, quote, quote_hash}]`, candidate findings with passage IDs, counterevidence, and unresolved gaps. Qwen's findings remain proposals. Quote and offset checks establish **provenance**, not that a claim follows from the quote.
4. **Drafter:** Nemotron receives the trusted question and note template plus the bounded, verified bundle. It does not receive Qwen's raw code/transcript, broad vault history, or tools that write. Paper quotations are still untrusted data, so instructions inside a paper cannot alter the workflow. If the task asks how a paper relates to the user's work, the bundle must include reviewed project context with its own provenance, or Nemotron must label the connection as an inference. Evidence-first generation has research support for making citations more local and easier to review. [Attribute First, then Generate](https://aclanthology.org/2024.acl-long.182/); [CaMeL, capability boundaries for prompt injection](https://arxiv.org/abs/2503.18813).
5. **Claim gate:** Ask Nemotron for atomic factual claims, each tied to passage IDs, and render the note from that structured draft. Code rejects unknown IDs, omitted citations, changed quote bytes, stale revisions, and claims with no cited passage. A separate semantic review decides whether each passage actually supports its claim; automatic support judgments are fallible. The first version should expose uncertain claims for human review and never describe structural citation validity as factual correctness. [ALCE](https://arxiv.org/abs/2305.14627); [FActScore](https://arxiv.org/abs/2305.14251); [AttributionBench](https://aclanthology.org/2024.findings-acl.886/).
6. **Single writer:** The harness's sole vault mutation is an exclusive, atomic stage into `_inbox/<run-id>/`. It rejects traversal, symlinks, existing targets, and any `library/` destination. There is no model-facing or MCP promotion tool. User promotion is a manual Obsidian/file action after checking relevance and semantic support. The run record, kept outside the vault, includes inputs, model identities and settings, code and tool results, source hashes, bundles, checks, failures, timing, token use, and staged paths. A failed run cannot leave a note that looks reviewed.

For insufficient evidence, the verifier returns a typed `needs_more_evidence` result. The harness may allow one bounded Qwen follow-up, then stages a clearly incomplete research card or abstains. It does not let the two models negotiate indefinitely. A compact evidence bundle protects Nemotron's context budget; it also means Qwen can omit decisive evidence, so coverage and contradiction-seeking must be evaluated separately. [Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).

## What the repository has and what it lacks

| Boundary | Current implementation | Required change |
| --- | --- | --- |
| Bounded Qwen code investigation | `src/product/qwen_investigator.py` uses a Docker-only REPL and checks exact spans | Put source access behind a logged, narrow broker; prevent direct source-file reads if tool-mediated behavior is the claim |
| Verified handoff | `src/research/code_exec_packet.py` and `scripts/research_library_mcp.py` check corpus identity and quote offsets | Add durable passage IDs and a typed, independently rebuilt bundle; keep semantic support a separate status |
| Nemotron synthesis | `src/product/chat.py` passes checked evidence to `src/product/nemotron_chat.py` for the personal-vault chat path | Add the same constrained drafting contract for public papers and weekly notes |
| MCP | `scripts/research_library_mcp.py` is read-only and exposes high-level paper operations | Reuse one source service under MCP and sandbox tools; do not expose vault write or promotion |
| Vault writing | `src/product/weekly_digest.py` can write directly to `AI Research/`; `src/product/memory.py` and `src/research/vault.py` have other live-vault write paths | Route all harness-generated notes through one `_inbox/` writer and remove direct `library/` publication paths |
| Durable run | Development eval saves trajectories; production MCP does not persist a whole-run record | Save one append-only, reproducible run record outside the vault |

The current W39 note preview and four-question paper development set are drafts, not human-reviewed proof of the end-to-end system. The twelve held-out question drafts must remain blind until the evaluation protocol is fixed.

## Build and evaluation order

1. **Enforce the vault boundary first.** Implement `_inbox/` staging and make every existing vault-writing route use it or fail. Test direct `library/` attempts, path traversal, symlink escapes, duplicate run IDs, and mid-write failures. Success means zero harness writes outside `_inbox/`.
2. **Version the source service and evidence bundle.** Use one store for the sandbox facade and MCP; add durable passage IDs, actual tool-return logging, and an independent verifier. Test forged spans, changed source revisions, undiscovered IDs, and empty evidence. Success means these inputs cannot pass the provenance gate.
3. **Connect public-paper Nemotron drafting.** Give it only the trusted task/template and verified bundle, return structured claims, check references, and stage a draft. Test unsupported-claim and prompt-injection examples. Human review remains the semantic gate.
4. **Run the real product flow.** Review a weekly draft for usefulness and claim support, promote it manually, then query the promoted library through MCP. Measure time to review, useful-paper rate, unsupported claims, and whether the answer reopens the underlying paper.
5. **Compare models under one harness.** Human-review the development references, then run base and trained Qwen on the same frozen paper tasks, prompt, tool service, decoding, budget, and hardware. Review paired answers blind and report supported-answer quality, source coverage, execution failures, latency, tokens, and cost. Only use the held-out set under its predeclared protocol. This follows the repository's two project gates and agent-evaluation guidance to assess both traces and final outcomes. [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

Use plain Python plus an append-only event table in SQLite, or JSONL with immutable artifacts, for this first linear workflow. LangGraph offers durable checkpoints and human interrupts, and AutoGen supports more dynamic team patterns, but neither is necessary to implement the present five-stage path. Adopt more orchestration machinery only when crash recovery or branching cannot be handled cleanly in the local harness. [LangGraph persistence](https://langchain-ai.github.io/langgraph/reference/); [AutoGen teams](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/teams.html).
