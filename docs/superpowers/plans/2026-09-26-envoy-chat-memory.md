# Envoy Chat Memory Implementation Plan

> **For agentic workers:** Use test-first development and verify the integrated chat path before calling this increment complete.

**Goal:** A local chat API delegates vault investigation to a bounded Qwen worker when configured, sends checked evidence to Nemotron when configured, and preserves approval-gated Obsidian memory.

**Architecture:** Keep the existing vault snapshot and approval service. Add small, injectable product modules for link indexing, Qwen investigation, Nemotron planning/answering, and one chat orchestration boundary. Default to explicitly labeled lexical/evidence-only behavior when model services are unavailable.

**Tech Stack:** Python, FastAPI, the existing vault importer and code-execution tools, Docker sandbox, pytest.

**Spec:** [2026-09-26-envoy-chat-memory-design.md](../specs/2026-09-26-envoy-chat-memory-design.md)

## Global constraints

- Do not touch training/reward code or the user's protected uncommitted files.
- Leave the hash-locked nine-note fixture unchanged.
- Qwen-generated Python executes only with a Docker image available; no local fallback.
- Model identities and fallback modes must appear in each response.
- Chat cannot persist a memory; reuse explicit approval for writes.

## Review focus

- A Qwen span with a real doc ID but altered quote must fail provenance validation.
- A no-evidence result must not become a fabricated source or a confident answer.
- A missing Docker image or model endpoint must not trigger local code execution.
- An unrelated or ambiguous Obsidian link must not silently resolve to a note.
- A chat turn must not save a vault note without the approval endpoint.

### Task 1: Snapshot graph

**Files:** `src/product/graph.py`, `tests/test_product_graph.py`

**Interfaces:** `build_graph(snapshot: Path) -> dict`; `one_hop_neighbors(graph: dict, doc_id_or_record_id: str) -> list[dict]`.

- [x] Write and run tests for exact wikilinks, relative Markdown links, typed revisions, backlinks, and ambiguous or unsafe targets; confirm the new tests fail before code.
- [x] Implement the read-only graph builder and run `pytest tests/test_product_graph.py -q`.
- [x] Check the existing synthetic corpus hash remains unchanged.

### Task 2: Bounded Qwen investigation

**Files:** `src/product/qwen_investigator.py`, `tests/test_qwen_investigator.py`

**Interfaces:** `QwenInvestigator.investigate(question: str, snapshot: Path) -> dict` returns status, exact evidence, corpus hash, model identity, and trace.

- [x] Write and run tests with fake policy/REPL for a valid two-step investigation, no evidence, invalid span, and absent Docker image; confirm failures precede code.
- [x] Implement Docker-only execution and span materialization without calling reward or requiring gold answers.
- [x] Run `pytest tests/test_qwen_investigator.py -q` and the code-execution boundary tests.

### Task 3: Nemotron planning and explanation

**Files:** `src/product/nemotron_chat.py`, `tests/test_nemotron_chat.py`

**Interfaces:** `NemotronChatClient.plan(question, history) -> {investigate, query}` and `.answer(question, evidence_packet, history) -> checked explanation`.

- [x] Write and run tests for planning, exact citation IDs, no-evidence restraint, missing key, and per-instance authorization; confirm the tests fail first.
- [x] Implement bounded, product-specific prompts and request transport without process-wide environment mutation.
- [x] Run `pytest tests/test_nemotron_chat.py -q`.

### Task 4: Chat service and local API

**Files:** `src/product/chat.py`, `scripts/personal_memory_web.py`, `tests/test_personal_memory_chat.py`, `tests/test_personal_memory_web.py`, `docs/PERSONAL_MEMORY_LOCAL_DEMO.md`

**Interfaces:** `ChatService.reply(question: str, snapshot: Path, history: list[dict]) -> dict`; `POST /api/chat` accepts question and optional session ID.

- [x] Write and run tests for delegation order, verified handoff, lexical fallback, evidence-only status, no write, and stale review; confirm failure before implementation.
- [x] Add the orchestration boundary and API endpoint, keeping approval separate.
- [x] Run the product tests, Ruff, then a real localhost HTTP smoke on a copy of the sample vault. Record actual model-call status.

### Task 5: Final verification

**Files:** this plan and the local demo documentation.

- [x] Inspect the diff for protected files, fixture changes, unintended network access, and misleading model labels.
- [x] Run the narrow product suite, related evidence and vault tests, `ruff check` on touched Python files, and `git diff --check`.
- [x] Report which calls were real versus injected or unavailable; do not claim a Qwen/Nemotron live run without one.
