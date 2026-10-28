"""Local experiment context plus model-free paper research, without GPU startup."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server import MCPServer
from mcp.types import CallToolResult
from pydantic import StrictInt

from scripts.arxiv_research_mcp import LOCAL, REMOTE
from scripts.arxiv_research_mcp import create_server as source_server
from scripts.ml_experiment_mcp import (
    INSTRUCTIONS,
    SAVE,
    _errors,
    _present,
    register_experiment_tools,
)
from src.product.arxiv_library import ArxivLibrary
from src.product.experiment_context import resume_context
from src.product.experiment_discovery import ExperimentDiscovery
from src.product.ml_experiment_evidence import EvidenceResolver
from src.product.ml_experiment_research_notes import ResearchNote
from src.product.ml_experiment_store import ExperimentStore
from src.product.paper_discovery import PaperDiscovery

MODE = """
Experiment mode connects research to the user's ongoing ML experiment.
Create a question-only draft with save_experiment_brief, or list_experiments and
resume_experiment with the chosen ID. IDs are explicit: no shared active experiment.
Resume before researching or revising so you inspect prior attempts, setup and
results. Use ordinary host web tools for repositories, model cards and docs; use
paper tools when paper evidence is relevant. Do not research on every turn.
Record useful findings with record_experiment_research against the brief revision
you inspected. The host supplies this note explicitly; source calls do not save
notes automatically. Qwen reports from a separately configured worker can be
summarized the same way. This server invokes no model or GPU. Do not send private
experiment context to an external reader without authorization.
Research notes are append-only context. They do not revise a hypothesis, freeze
a plan, execute a test, or replace measurements. A later plan change requires an
explicit brief revision or a successor if the plan is frozen. Record actual supplied
measurements with record_experiment_result against the frozen plan, then resume.
Preserve conflicting evidence and unknowns. Findings and proposed tests are
unreviewed. External URLs are references only; this server does not fetch them.
Use get_experiment for full or older records if resume context is truncated.
"""


def create_server(store, library, discovery, *, passage_reader=None):
    server = MCPServer("Envoy Experiment Mode", instructions=INSTRUCTIONS + MODE)
    register_experiment_tools(server, store)
    if passage_reader is not None:
        @server.tool(annotations=SAVE.model_copy(update={"open_world_hint": True}))
        @_errors
        def read_experiment_passages(experiment_id: str, expected_revision: StrictInt,
                                     note_id: str, question: str, arxiv_id: str,
                                     windows: list[dict[str, StrictInt]]) -> dict[str, Any]:
            """Read selected passages with Qwen; save candidate evidence or an unresolved gap.

            Makes bounded external model calls only when invoked. Sends the question,
            paper metadata and selected passage text. Does not activate a GPU. Exact
            quotes prove provenance, not meaning. A note ID is one durable attempt;
            identical retries return its saved report without another model call.
            Gaps cover inspected passages only. Suggested next searches are not executed.
            """
            return passage_reader.read(experiment_id, expected_revision, note_id,
                                       question, arxiv_id, windows)
    sources = source_server(library, discovery, tool_profile="sources")

    # Delegate to the existing source implementation through the public SDK.
    @server.tool(annotations=REMOTE, structured_output=False)
    async def search_papers(query: str, since: str | None = None, until: str | None = None,
                            limit: int = 10) -> CallToolResult:
        """Discover papers by relevance, or by date when since/until is set."""
        return await sources.call_tool("search_papers", {
            "query": query, "since": since, "until": until, "limit": limit})

    @server.tool(annotations=REMOTE, structured_output=False)
    async def get_citations(arxiv_id: str, direction: str = "references",
                            limit: int = 20) -> CallToolResult:
        """List references or citing papers; source availability may be limited."""
        return await sources.call_tool("get_citations", {
            "arxiv_id": arxiv_id, "direction": direction, "limit": limit})

    @server.tool(annotations=REMOTE, structured_output=False)
    async def open_paper(arxiv_id: str) -> CallToolResult:
        """Open and cache version-pinned paper text, outline and coverage."""
        return await sources.call_tool("open_paper", {"arxiv_id": arxiv_id})

    @server.tool(annotations=LOCAL, structured_output=False)
    async def find_evidence(query: str, arxiv_ids: list[str], top_k: int = 8) -> CallToolResult:
        """Rank exact passages across already opened papers; not a support judgment."""
        return await sources.call_tool("find_evidence", {
            "query": query, "arxiv_ids": arxiv_ids, "top_k": top_k})

    @server.tool(annotations=LOCAL, structured_output=False)
    async def get_passage(arxiv_id: str, start: int = 0, length: int = 4_000) -> CallToolResult:
        """Inspect exact text from an opened paper, at most 8000 characters."""
        return await sources.call_tool("get_passage", {
            "arxiv_id": arxiv_id, "start": start, "length": length})

    @server.tool(annotations=SAVE)
    @_errors
    def record_experiment_research(experiment_id: str, expected_revision: StrictInt,
                                   note: ResearchNote) -> dict[str, Any]:
        """Append source-checked, unreviewed research to the exact inspected brief revision.

        An identical note ID/content/revision retry is safe. This does not edit the
        brief or frozen plan. References use already cached papers or selected exports.
        """
        return _present(store.record_research(experiment_id, expected_revision,
                                             note.model_dump(mode="json")))

    @server.tool(annotations=LOCAL)
    @_errors
    def resume_experiment(experiment_id: str) -> dict[str, Any]:
        """Resume current setup, research and latest supplied result without model calls.

        Context is bounded; get_experiment retrieves full and older records.
        """
        return resume_context(store.get(experiment_id))

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("~/.envoy/ml-experiments"))
    parser.add_argument("--library-root", type=Path, default=Path("~/.envoy/arxiv-library"))
    parser.add_argument("--artifact-root", type=Path,
                        help="Explicit directory of selected sanitized JSON run exports")
    parser.add_argument("--qwen-endpoint", help="Optional OpenAI-compatible reader endpoint")
    parser.add_argument("--qwen-model")
    parser.add_argument("--qwen-checkpoint")
    parser.add_argument("--sandbox-image", default="rlm-sandbox")
    args = parser.parse_args()
    library = ArxivLibrary(args.library_root.expanduser())
    store = ExperimentStore(args.store_root, evidence_resolver=EvidenceResolver(
        library_root=library.root, artifact_root=args.artifact_root))
    discovery = ExperimentDiscovery(PaperDiscovery(library.root / "discovery-cache"))
    passage_reader = None
    if args.qwen_endpoint or args.qwen_model or args.qwen_checkpoint:
        try:
            from src.env.repl import PersistentREPL
            from src.policies.bounded_http import bounded_transport, validate_endpoint
            from src.product.assigned_reader import build_assigned_reader
            from src.product.experiment_budget import ExperimentBudget
            from src.product.experiment_tokenizer import load_tokenizer
            from src.product.passage_reader_service import PassageReaderService
            from src.product.passage_scope import PASSAGES_V2
            if not (args.qwen_endpoint and args.qwen_model and args.qwen_checkpoint):
                raise ValueError("passage reader requires endpoint, model and checkpoint")
            validate_endpoint(args.qwen_endpoint)
            if not PersistentREPL._docker_available(args.sandbox_image):
                raise ValueError("Docker sandbox image unavailable; no inference dispatched")
            tokenizer = load_tokenizer("Qwen/Qwen3-8B",
                                      "b968826d9c46dd6066d109eabc6255188de91218")
            def factory():
                budget = ExperimentBudget(tokenizer, bounded_transport,
                                          requests_max=4, tokens_max=32768)
                reader = build_assigned_reader("qwen", qwen_endpoint=args.qwen_endpoint,
                    qwen_model=args.qwen_model, qwen_checkpoint=args.qwen_checkpoint,
                    sandbox_image=args.sandbox_image, max_steps=4,
                    reader_guidance=PASSAGES_V2, policy_transport=budget)
                return reader, budget
            passage_reader = PassageReaderService(store, library, factory,
                                                  library.root / "passage-reader-reports")
        except (ValueError, OSError, ImportError) as exc:
            parser.error(str(exc))
    create_server(store, library, discovery, passage_reader=passage_reader).run(transport="stdio")


if __name__ == "__main__":
    main()
