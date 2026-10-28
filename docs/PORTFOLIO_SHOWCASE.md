# Qwen Envoy — portfolio walkthrough

The strongest public evidence is the [Qwen agent case study](TECHNICAL_CASE_STUDY.md):
a Docker-sandboxed code-execution worker, QLoRA training, saved paired evaluation,
and inspectable traces. The personal research MCP is a complementary workflow
for recalling prior experiments and recording paper-to-result decisions. It
checks source bytes, artifact bindings, revisions and result arithmetic while
the host assistant reasons over the evidence. The optional reader is not a
proven automatic evidence-card writer; the core MCP workflow does not call Qwen.

```text
ML question → recall related attempts → paper search/open/passages
            → source-linked research note → frozen plan → measured result
            → next-session recap
```

## Show it in two minutes

Use a new output directory for each command. Both run locally with the project dependencies; neither starts a GPU or calls a model.

1. Run a **live source-tool check**. It uses bounded source requests and records both a compact summary and full local responses:

   ```sh
   python scripts/smoke_arxiv_research_mcp.py \
     --library-root out/my-mcp-source-demo/library \
     --detail-output out/my-mcp-source-demo/detail.json \
     --output out/my-mcp-source-demo/summary.json
   ```

   On October 26, this path made **19 actual MCP calls**, opened **three versioned arXiv papers**, and matched **9/9 selected passage spans** to stored source text in 20.25 seconds. It made **zero model requests**. The exact-span result checks provenance; it does not judge whether a passage supports a claim. Fresh source availability and timing can differ.

2. Run the **research-to-result record demo**:

   ```sh
   python scripts/demo_experiment_mode.py --output out/my-mcp-record-demo
   ```

   Open `out/my-mcp-record-demo/RESUME.md` to see what a fresh assistant session receives, and `report.json` for each tool response and its hash. This demo exercises **16 MCP calls across two stdio processes**, including a related-work lookup after restart. It preserves two research notes and an artifact-bound result, accepts identical retries, and rejects conflicting edits. It replays previously recorded paper and serving evidence; it does not perform a new live paper investigation or infer a latency win. A [committed example](../examples/experiment-mode-complete/README.md) contains the earlier workflow and its limitations.

The live source check and the record demo exercise different stages. The
[hackathon research demo](HACKATHON_SUBMISSION_2026.md) connects a public paper
and saved note to an **opt-in live Nemotron 3.5 Lightning call on Nebius Token
Factory**. Its October 26 successful run used 755 input and 273 output tokens;
the selected claim still required manual support review. The earlier
[combined walkthrough](../examples/experiment-mode-walkthrough/README.md)
connects live paper discovery to a saved draft. None of these demos measures a
productivity advantage over ordinary Codex browsing.

## Resume-ready wording

> **Qwen Envoy** — Built a Docker-sandboxed Qwen3-8B paper agent, fine-tuned it
> with QLoRA, and compared base and trained checkpoints on 40 selected QASPER
> questions. Recorded-error episodes fell from 22/40 to 1/40; provisional
> supported-answer passes on answerable questions stayed 3/20. Published saved
> traces and an optional source-linked MCP for experiment recall.

This wording reports a reliability observation, not a demonstrated answer-quality
gain. The support judgments are model-assisted and await human review. Keep the
MCP's product claims separate from the QASPER model-evaluation numbers.

## What is proved, and what remains

The verified result is an **auditable workflow**, not superior research quality. The MCP can make source text and measured results inspectable across sessions. Its source checks prove that cited text came from stored papers; exact quotes alone do not prove semantic support. The October 26 live check was an agent-authored development smoke, not an independent human review.

This personal tool requires the host to write research notes and results explicitly. Its related-work tool searches **local word overlap**, not meaning, and returns IDs for the host to inspect. It does not automatically capture every coding-agent session, provision GPUs, or provide a team dashboard. The optional Qwen reader has not shown a reliable answer-quality advantage, and existing native Codex comparison sessions did not call the MCP. There is **no measured MCP-versus-Codex web win** or fine-tuned-Qwen improvement claim.

The strongest next comparison is a paired set of real ML research decisions with the same host model using ordinary web tools versus this MCP, retaining actual MCP calls and a claim-level source review. That requires new answers and human assessment; the smoke counts above cannot stand in for it.

## Inspect the implementation

- [Combined MCP and tool flow](EXPERIMENT_MODE.md) · [server entry point](../scripts/experiment_mode_extended_mcp.py)
- [Versioned arXiv source tools and limits](ARXIV_RESEARCH_MCP.md) · [live source report](../reports/arxiv-mcp-live-20261016.json)
- [Completed experiment example](../examples/experiment-mode-complete/README.md) · [source-to-draft walkthrough](../examples/experiment-mode-walkthrough/README.md)
- [QASPER Agent Studio, a separate model case study](TECHNICAL_CASE_STUDY.md)
