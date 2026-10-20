# Assigned-paper handoff and research gaps

The arXiv MCP has a new opt-in reader path. The controller gives each reader the
title, version and document ID of its one assigned paper, derived from the
hash-checked snapshot. Receiving this metadata does not count as reading. The
reader must still inspect source text and submit exact evidence spans.

After inspecting text, a reader can report an unresolved evidence need:

```python
report_gap(
    missing="I have not found measurements for a larger batch size.",
    follow_up_query="speculative decoding larger batch throughput measurements"
)
```

This is a standalone control action parsed with AST/literal validation, never
executed as Python on the host. It uses one action and one model response. A
report does not count as a document-tool action, end the investigation, or
launch a search. The reader reports gaps before submitting its findings or
using the existing bounded abstention protocol.

The MCP response includes both verified source spans and `research_gaps`.
Each gap carries the controller-owned paper/document IDs,
`review_status: "unreviewed_model_report"` and `absence_proven: false`.
The host can assess the suggested query and call the existing search tools.
No autonomous expansion, new index or retrieval/reranking system was added.

## Protocol and safeguards

- Assigned mode requires harness V2 and a single version-pinned public paper.
  The expected mode comes from trusted reader configuration, so removing a
  packet's mode cannot bypass assignment verification.
- Policy, investigator and packet use the same effective system-prompt bytes.
  Both old discovery instructions are replaced only in assigned mode. The V1
  preamble and default V1/V2 prompts remain unchanged.
- Assigned mode deliberately accepts a narrower Python syntax: one direct
  document-tool call with scalar literal arguments, optionally wrapped in
  `print()`, or `hits = tool(...); print(hits)` / the same with `result`.
  It rejects imports, conditionals, functions, loops, arbitrary expressions,
  tool redefinitions and other assignments before execution. Control actions
  `report_gap(...)` and existing literal submissions are handled separately.
- A permitted inspection must display actual nonempty assigned source text:
  matching offset-backed windows, an extracted source match, or a full/bounded
  read. Metadata-only output and tool names inside dead code cannot unlock
  submission or gap reporting. This checks source access/provenance, not
  semantic reading or claim support.
- Up to three distinct gaps are allowed; `missing` is bounded to 600 characters
  and `follow_up_query` to 300. Reports and format failures stay inside the
  existing request/action limits. Gap records cannot replace the two distinct
  successful document actions required for abstention.
- Malformed ancillary gaps are excluded with `gap_report_error` while valid
  source evidence remains. Other paper readers' evidence survives a reader
  failure. Optional synthesis receives the gaps explicitly as unreviewed
  suggestions, separate from evidence.
- Normal model-authored tool Python still runs only in Docker, with no host
  fallback. No training, reward or model-weight changes were made.

## Enable it

The existing MCP tool surface is unchanged. Use the separate opt-in entry point
with an already configured endpoint:

```bash
python scripts/arxiv_assigned_reader_mcp.py \
  --reader qwen \
  --qwen-endpoint http://127.0.0.1:8000/v1 \
  --qwen-model envoy \
  --qwen-checkpoint '<actual pinned checkpoint identifier>' \
  --sandbox-image '<available Docker image or digest>'
```

`--reader nemotron` instead uses the existing Nebius configuration; synthesis
is off unless explicitly enabled. The Runpod queue entry point supports
`--assigned-paper-reader` and retains its shared durable request/token ledger.
These entry points configure a client; they do not deploy or start compute.
Invoking the reader against a paid endpoint can still incur provider charges.
No hosted endpoint was enabled for this implementation. A final Runpod read
confirmed endpoint `q05pjrvguakieg` remains min 0/max 0, with zero active workers
and zero queued/in-progress jobs.

The host can continue to call `search_papers`, choose paper IDs, and call
`investigate(question, arxiv_ids=[...])`. It then receives available findings
and gap suggestions in the same tool response and decides whether to search
again. Exact quotes remain provenance checks; the host must assess support.

## Verification and limits

The first handoff commit is `9043a32`. Final independent review covered the
handoff together with gap reporting, prompt identity, default compatibility and
the Docker boundary. It found a static-AST inspection weakness exposed by this mode:
a `read()` call in dead code could count as inspection. Assigned mode now uses
the constrained/displayed-source contract above, with regressions for dead
code, function definitions, tool/global shadowing and metadata-only output.
Review also found and corrected the residual V2 prompt discovery instruction.
`14da384` contains gap reporting and the inspection fixes. `523dddb` fixes an
older queue-CLI dependency on an uncommitted image parameter, with regressions
for both reader modes. The new assigned builder also handles the image itself.

Focused verification covers controller-derived IDs, mode/hash tampering,
premature reports, literal-only parsing, report limits, recovery, malformed
ancillary fields, per-paper failures and a real in-process MCP client. Optional
Docker tests use scripted responses through the real MCP and Docker sandbox:
one positive source-plus-gap case and one negative dead-code case. They make
zero actual model-inference requests. Final verification passed **1,588 tests
in 68.49 seconds**, with no skips and three existing SWIG deprecation warnings.
The full run used the pinned Docker image and cached tokenizers with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Ruff passed for every changed/new
Python file. A clean export of `14da384`, excluding all user uncommitted files,
also passed 90 relevant tests. Commands, hashes, separate focused results and
the final infrastructure read are in the
[validation artifact](../../examples/assigned-reader-gaps/validation.json).

This feature has **not been live-tested with v5 or Nemotron**. The narrower
action grammar and gap command are a new protocol, not the exact v5 training
protocol. The earlier [successful known-paper diagnostic](2026-10-20-known-paper-v5-diagnostic.md)
does not establish compliance with this new path. There is no measured answer
quality, latency or cost improvement, no state-of-the-art claim, and no claim
that fine-tuned Qwen improved. The next model test should be separately bounded
and compare the new path against retrieval feeding the host directly.
Rejected actions can consume the budget; the narrower grammar is an explicit
tradeoff whose effect on live model behavior is still unknown.

All ten protected files retain their original hashes, including the user's
uncommitted server/config/documentation changes and the three draft plan/spec
files. The new entry point avoids editing those pending changes.
