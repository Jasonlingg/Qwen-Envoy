# Envoy chat memory: first vertical slice

Date: 2026-09-26

## Intended outcome

Turn the existing localhost evidence-and-approval prototype into a chat-first
learning debugger over a user-selected Obsidian vault. A person asks what they
previously believed or tried; the service finds dated, linked evidence;
Nemotron explains the result with source references; and a durable revision
is written only through the existing explicit approval action. Qwen v5 is the
optional, bounded, read-only investigator called by the chat coordinator.
The existing form UI is not the product design and will receive no styling
work in this slice.

Success for this increment is a local JSON chat flow with inspectable exact
source passages and honest model identities. It is not a claim that v5 beats
base Qwen, that answers are semantically verified, or that a live Nebius call
was made without credentials. The weekly digest and held-out model-improvement
gates remain separate.

## Components and flow

1. `personal_memory_web.py` retains a user-selected vault, immutable snapshot,
   review IDs, source opening, and approval-gated append-only writes. It adds
   `/api/chat` and short in-process conversation history keyed by session ID.
2. A Nemotron client receives the latest question and bounded history. Its
   planning turn chooses whether to call the vault investigator and supplies
   only a query, never a filesystem path. Its answer turn receives a compact,
   verified packet; it cannot write to the vault.
3. When configured, a Qwen v5 policy writes Python against frozen-corpus
   `search`/`read`/`extract` tools in a Docker-only, network-disabled sandbox,
   with a fixed action budget. No local Python fallback is allowed. Its
   candidate answer is untrusted; exact spans are checked independently.
   Without a Qwen endpoint or sandbox image, the app uses labeled lexical
   retrieval. Without a Nemotron key, it returns an evidence-only response,
   not an invented model answer.
4. A deterministic link index reads actual Obsidian links and dated relation
   IDs from the snapshot. It exposes neighboring notes and unresolved links;
   it never guesses ambiguous note titles. The frozen nine-note evaluation
   fixture stays unchanged.
5. The source verifier checks doc IDs, offsets, quotes, corpus hash, and the
   five-passage cap. Nemotron receives the passage text and dated context but
   not arbitrary tool stdout. The API returns model/retrieval identities,
   source paths and opening links, checked references, and uncertainty.

## Boundaries

- Vault sources and model output are untrusted text. A valid quotation proves
  provenance, not semantic support.
- The selected vault is the read scope. New notes use the existing approval
  route. A chat answer cannot silently update memory.
- Remote Nemotron calls need an operator-provided key and an explicit remote
  mode on the selected vault. The offline flow must remain usable without
  uploading private excerpts.
- Sessions are in-process for this slice; Obsidian Markdown is the durable
  cross-conversation memory. Restarting the server may end a chat session but
  must not erase approved memory.
- No training, MuSiQue reward changes, graph database, scheduled research
  discovery, or public Qwen execution endpoint belongs to this increment.

## Acceptance checks

- With injected fake model clients, one chat request delegates to the
  investigator, returns a checked source-linked answer, and creates no note.
- Missing Qwen or sandbox is reported and uses labeled lexical retrieval;
  missing Nemotron yields evidence-only output. No component impersonates a
  model that did not run.
- Tampered or out-of-snapshot spans cannot reach the answerer. No-evidence
  is represented without inventing a quotation. A stale approval review is
  rejected after the vault changes.
- Link indexing preserves valid neighbors, backlinks, and revision edges and
  reports missing or ambiguous links.
- Narrow product tests, relevant existing tests, static lint, and a local
  HTTP smoke pass succeed before reporting the slice as working.
