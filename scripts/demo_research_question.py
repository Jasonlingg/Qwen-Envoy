"""One real paper-to-result question trail; no model, GPU, or paid request."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import Client

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_mode_extended_mcp import create_server  # noqa: E402
from scripts.export_serving_experiment import build_example, encoded  # noqa: E402
from src.product.arxiv_library import ArxivLibrary  # noqa: E402
from src.product.experiment_discovery import ExperimentDiscovery  # noqa: E402
from src.product.ml_experiment_evidence import EvidenceResolver  # noqa: E402
from src.product.ml_experiment_store import ExperimentStore  # noqa: E402
from src.product.paper_discovery import PaperDiscovery  # noqa: E402

PAPER = "2309.06180v1"
QUESTION = "Did the PagedAttention idea lead to a useful two-reader Qwen serving result?"


async def run(output: Path) -> dict:
    output = output.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    exports = output / "exports"
    exports.mkdir()
    bundle = build_example(ROOT)
    for name, value in bundle.items():
        (exports / name).write_bytes(encoded(value))
    library = ArxivLibrary(output / "library")
    store_root = output / "records"
    store = ExperimentStore(store_root, evidence_resolver=EvidenceResolver(
        library_root=library.root, artifact_root=exports))
    discovery = ExperimentDiscovery(PaperDiscovery(library.root / "discovery-cache"))
    calls = []

    async def call(client, tool, args):
        response = await client.call_tool(tool, args)
        calls.append({"tool": tool, "is_error": bool(response.is_error)})
        if response.is_error:
            raise ValueError(f"{tool} failed: {str(response.content)[:400]}")
        return response.structured_content

    async with asyncio.timeout(240):
        async with Client(create_server(store, library, discovery)) as client:
            opened = await call(client, "open_paper", {"arxiv_id": PAPER})
            if opened["status"] != "ok":
                detail = opened.get("detail", opened["status"])
                raise ValueError(f"Public paper unavailable: {detail}")
            found = await call(client, "find_evidence", {"query":
                "PagedAttention fixed size blocks key value memory serving",
                "arxiv_ids": [PAPER], "top_k": 3})
            if not found["evidence"]:
                raise ValueError("No passage found in the opened paper")
            span = found["evidence"][0]
            passage = await call(client, "get_passage", {"arxiv_id": PAPER,
                "start": span["start"], "length": min(span["end"] - span["start"], 900)})
            question = await call(client, "start_research_question", {"question": QUESTION})
            question_id = question["question_id"]
            saved = await call(client, "save_experiment_brief", {"brief": {"question": QUESTION}})
            experiment_id = saved["experiment_id"]
            note = {"note_id": "paged_attention_paper", "question": QUESTION,
                "findings": [{"claim": "The paper describes a KV-memory mechanism; "
                              "it does not measure this Qwen serving setup.",
                              "reference_ids": ["paged_attention_span"]}],
                "references": [{"kind": "paper_span", "reference_id": "paged_attention_span",
                    "arxiv_id": PAPER, "start": passage["start"], "end": passage["end"],
                    "quote": passage["text"]}],
                "uncertainties": ["Whether the mechanism improves local latency or quality "
                                  "is not established by this paper passage."],
                "proposed_test": "Measure matched warm and cold Qwen requests.",
                "origin": "host_research", "data_kind": "recorded"}
            await call(client, "record_experiment_research", {
                "experiment_id": experiment_id, "expected_revision": 1, "note": note})
            basis = {"experiment_id": experiment_id,
                     "research_note_id": note["note_id"],
                     "reference_id": "paged_attention_span"}
            question = await call(client, "consider_research_paper", {
                "question_id": question_id, "expected_revision": 1,
                "paper": {"paper_id": "paged_attention", "source": basis,
                    "relevance": "Its KV-memory mechanism motivated inspecting "
                                 "our concurrent reader setup.",
                    "limitation": "The paper does not establish our latency or "
                                  "two-reader memory fit.",
                    "author_kind": "agent_proposed"}})
            await call(client, "save_experiment_brief", {"experiment_id": experiment_id,
                "expected_revision": 1, "brief": bundle["brief.json"]})
            frozen = await call(client, "freeze_experiment_plan", {
                "experiment_id": experiment_id, "expected_revision": 2})
            result = bundle["result-input.json"]
            recorded = await call(client, "record_experiment_result", {
                "experiment_id": experiment_id, "plan_sha256": frozen["plan_sha256"],
                "result": result})
            question = await call(client, "link_paper_to_result", {
                "question_id": question_id, "expected_revision": question["revision"],
                "link": {"link_id": "paged_attention_to_serving", "paper": basis,
                    "result": {"experiment_id": experiment_id, "run_id": result["run_id"]},
                    "method_tried": "Ran the saved two-request Qwen/vLLM serving diagnostic.",
                    "observed_outcome": "Both requests shared one engine; a latency "
                                        "win was not established.",
                    "interpretation": "The paper suggested a memory question, but "
                                      "this run does not validate the paper's method "
                                      "or a Qwen speedup.",
                    "next_step": "Run matched real reader tasks with complete timings "
                                 "and answer-quality checks.",
                    "author_kind": "agent_proposed"}})
            question = await call(client, "record_question_view", {
                "question_id": question_id, "expected_revision": question["revision"],
                "view": {"view_id": "after_serving_diagnostic",
                    "statement": "Concurrent admission worked; a latency benefit remains unknown.",
                    "reason": "The saved result was inconclusive under its own criteria.",
                    "next_question": "Do representative matched reader tasks improve "
                                     "end-to-end latency without answer-quality loss?",
                    "basis_link_ids": ["paged_attention_to_serving"],
                    "author_kind": "agent_proposed"}})

        restarted = ExperimentStore(store_root)
        async with Client(create_server(restarted, library, discovery)) as client:
            found_questions = await call(client, "find_research_questions", {
                "query": "PagedAttention Qwen readers"})
            recovered = await call(client, "get_research_question", {
                "question_id": question_id})
            if (not found_questions["matches"]
                    or found_questions["matches"][0]["question_id"] != question_id
                    or recovered != question
                    or recorded["selected_result"]["assessment"]["overall"]
                    != "inconclusive"):
                raise ValueError("Restarted question trail or result did not match")

    report = {"schema_version": "research-question-demo-v1",
              "question_id": question_id, "experiment_id": experiment_id,
              "paper": PAPER, "assessment": "inconclusive",
              "mcp_calls": calls, "model_requests": 0, "gpu_seconds": 0,
              "question": recovered}
    (output / "report.json").write_bytes(encoded(report))
    (output / "question.json").write_bytes(encoded(recovered))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="New directory for the question, source and result records")
    args = parser.parse_args()
    report = asyncio.run(run(args.output))
    print(json.dumps({"output": str(args.output),
                      "question_id": report["question_id"],
                      "calls": len(report["mcp_calls"]),
                      "assessment": report["assessment"]}))


if __name__ == "__main__":
    main()
