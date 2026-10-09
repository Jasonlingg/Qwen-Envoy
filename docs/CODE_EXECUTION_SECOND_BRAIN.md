# Code-execution second brain

> September 28 scope clarification: the user has no pre-existing substantive
> Obsidian vault. Seed a growing AI-research library from version-pinned public
> papers. Qwen investigates frozen sources with multi-step Python tools and
> returns evidence; the host drafts and the reviewed writer saves notes. The
> first new base-versus-trained Qwen claim is scoped to that public research
> library, per the [prospective corpus amendment](QWEN_RESEARCH_LIBRARY_CORPUS_AMENDMENT_20260928.md).
> Personal notes can be added later without pretending they already exist.

## Product goal

Build a small knowledge-query agent that searches a growing Obsidian research library and returns evidence
to a larger assistant. The small model does the bounded, repeatable work: write Python, call
`search()`, `read()`, and `extract()`, inspect results across several turns, and submit an answer
with source IDs. The larger assistant can decide when research is needed and explain the returned
evidence to the user.

This boundary is the reason to use Qwen. It sits between the paper corpus and a stronger reasoning
model as a cheaper research worker. Qwen converts a bounded question into a compact, cited evidence
packet; Claude, GPT, or another host model reasons over that packet in the wider conversation. A
successful checkpoint does not need to replace the host model. It needs to reduce expensive
context and retrieval work without losing evidence, inventing support, or hiding uncertainty.

The user-facing product is now a [chat-first lifelong-learning second brain](PERSONAL_LEARNING_MEMORY_BUILD_PLAN.md)
over an Obsidian vault. People ask how their understanding changed, inspect the
dated linked sources, and revisit or apply what they learned. The
[weekly research radar](WEEKLY_RESEARCH_RADAR.md) supplies one ongoing stream of
AI-research evidence and a digest; Claude, GPT, or another host can also query
the library through MCP. The Qwen worker remains a bounded evidence
investigator inside that wider conversation.

The differentiated capability is reliable exploration through executable code. The first new
product-task comparison uses a frozen public AI-paper library; Obsidian holds
the reviewed notes that accumulate from it. MuSiQue remains a historical labeled
benchmark, not the product improvement gate.

## Active architecture

1. `scripts/research_vault.py import` freezes a selected Markdown/PDF folder as JSON documents.
2. `src/env/tools.py` exposes that frozen corpus through Python functions.
3. `src/env/repl.py` executes the model's code across a persistent multi-step session.
4. A Qwen policy observes each result and chooses the next code step or `SUBMIT:`.
5. The caller receives the answer, cited document IDs, and the full auditable trajectory.

Obsidian is the durable linked memory and a navigation surface through its
native graph and backlinks. A chat side panel is an eventual interface option;
the backend does not require a running Obsidian process. Imported snapshots
stay outside the vault and are immutable for reproducible runs. Frontmatter
IDs alone do not create graph edges, so approved notes need actual internal
links and a small deterministic index of their typed relationships.

## Evidence as of September 15, 2026

- On the same 50 real MuSiQue questions, base Qwen2.5-7B scores 0.158 and the SFT adapter scores
  0.176. This supports the narrow claim that supervised code-execution trajectories improved the
  measured exploration task.
- Both published GRPO model IDs contain the same adapter. The shared adapter scores 0.172, so the
  available artifacts do not show that GRPO improves on SFT or how performance changed over time.
- A real local Obsidian vault imported successfully into the existing corpus schema. `read()` and
  `extract()` worked immediately. That smoke exposed a one-document TF-IDF bug in `search()`, now
  covered by a regression test.
- The local/GPU execution backend now assigns one long-lived Python worker to each episode. An
  action executes once, variables remain in that worker, and parallel episodes use separate
  processes. The Docker backend still uses cumulative replay and needs a parity change before it
  is treated as an equivalent evaluation backend.
- These facts do not yet establish that the trained model is a useful personal research agent. The
  vault needs substantive notes, named questions, reviewed answers, and a base-versus-SFT run.
- Base Qwen3-8B (untrained) scored 0.445 average reward on the frozen 10-question AI-paper pilot
  after fixing a Qwen3-specific bug: `qwen_common.py` never disabled the model's native thinking
  mode, so it burned its whole token budget on `<think>` reasoning and never reached executable
  code. Citation precision/recall are strong (0.85/0.95) — retrieval already works. The gap is in
  answer discipline: it fails an abstention trap outright, drops explicit "keep separate"
  framing instructions, and once misdescribed a correctly-cited paper's actual mechanism. See
  [the full pilot writeup](QWEN3_BASELINE_PILOT.md) for per-question detail and the reproduce
  command.

## Paused work

The JSON-action agent in `src/research/agent.py`, `search_paper`, QASPER conversion, routing, and
reranker experiments are paused. QASPER can return later as a source of paper questions and
evidence, but generated demonstrations must teach Python tool use in the `src/env/` protocol.

## Next experiment

Create a small private evaluation set over the frozen vault. Start with 10 to 20 questions that
require finding, combining, or checking information across notes. Record expected source notes and
review answers manually; exact string matching is insufficient for personal notes.

The current, bounded before/after version of this experiment is specified in
[the targeted Qwen vault transfer test](TARGETED_QWEN_VAULT_TRANSFER_TEST.md).
It locks twelve new transfer questions and a separate four-question development
baseline before any targeted SFT run; the two-note demo snapshot is not enough
to instantiate its gold questions.

**Hypothesis:** the SFT checkpoint completes more vault questions with supported answers than the
base checkpoint because it learned the executable research loop on MuSiQue.

**Expected signal:** higher supported-answer rate and fewer empty searches, syntax failures, and
repeated actions on the same questions under identical decoding settings.

**Decision rule:** continue domain-specific code-trajectory collection only if the review shows a
clear recurring failure that better demonstrations can teach. If base and SFT both retrieve well,
build the orchestrator boundary next. If both fail because the corpus is too small or poorly
structured, improve the vault and retrieval before training.

### Minimal frozen pilot

The runnable pilot is intentionally small: six reserved papers and ten questions in
`data/research/code_exec_pilot_v1.json`. It adds exact character spans to the existing submission
format without changing the MuSiQue reward. On a GPU machine with the SFT adapter available:

```bash
BASE_MODEL_PATH=Qwen/Qwen3-8B \
CHECKPOINT_PATH=jasonlingg/qwen-envoy-qwen3-8b-qasper-targeted-sft-v1 \
  ./scripts/run_ai_paper_code_eval.sh
```

The command runs base and SFT with the same questions, corpus, tools, step limit, and seed. It then
creates a blind review bundle. Read `blind-review/review.md`, enter `pass`, `partial`, or `fail` for
each answer in `blind-review/review.json`, and reveal the system identities only afterward:

```bash
python scripts/review_code_exec_pilot.py score \
  --review out/research/<run>/blind-review/review.json \
  --key out/research/<run>/blind-review/blind-key.json \
  --output out/research/<run>/human-score.json
```

The primary metric is the human supported-answer pass rate. Exact span validity, required-paper
recall, execution errors, repeated actions, steps, and latency are diagnostics. Token-overlap
reward remains visible for compatibility but is not evidence of good research.
