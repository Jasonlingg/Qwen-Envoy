"""Personal experiment research MCP; explicit Qwen endpoint, no compute provisioning."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from scripts.arxiv_research_mcp import create_server as source_server
from src.env.repl import PersistentREPL
from src.policies.bounded_http import bounded_transport, validate_endpoint
from src.product.arxiv_library import ArxivLibrary
from src.product.assigned_reader import build_assigned_reader
from src.product.experiment_discovery import ExperimentDiscovery
from src.product.experiment_research import ExperimentResearcher
from src.product.experiment_tokenizer import load_tokenizer
from src.product.paper_discovery import PaperDiscovery


def create_server(library, discovery, researcher, *, runs_root=None):
    server = source_server(library, discovery, tool_profile="sources")

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def research_experiment(question: str, constraints: str = "") -> dict[str, Any]:
        """Research one ML experiment question using bounded Qwen search and Docker reading.

        Makes model calls only when explicitly invoked. Returns excerpts, unreviewed
        findings/test proposals, gaps, usage and traces, including partial results.
        Papers only: cannot check current repository or hardware compatibility.
        """
        try:
            result = researcher.run(question, constraints)
        except ValueError as exc:
            raise ToolError(str(exc)[:500]) from exc
        if runs_root is not None:
            try:
                root = Path(runs_root)
                root.mkdir(parents=True, exist_ok=True)
                path = root / f"{uuid4().hex}.json"
                result["saved_report"] = str(path.resolve())
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(result, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
            except OSError as exc:
                result.pop("saved_report", None)
                result["save_error"] = f"{type(exc).__name__}: {exc}"[:500]
        return result

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, default=Path("~/.envoy/arxiv-library"))
    parser.add_argument("--qwen-endpoint", required=True, help="OpenAI-compatible vLLM URL")
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--qwen-checkpoint", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    parser.add_argument("--tokenizer-revision",
                        default="b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--sandbox-image", default="rlm-sandbox")
    parser.add_argument("--requests-max", type=int, default=18)
    parser.add_argument("--tokens-max", type=int, default=147456)
    args = parser.parse_args()
    try:
        validate_endpoint(args.qwen_endpoint)
        if not PersistentREPL._docker_available(args.sandbox_image):
            raise ValueError("Docker sandbox image is unavailable; no model calls were made")
        # Never download tokenizer code or weights at server startup.
        tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_revision)

        def reader_factory(transport):
            reader = build_assigned_reader(
                "qwen", qwen_endpoint=args.qwen_endpoint, qwen_model=args.qwen_model,
                qwen_checkpoint=args.qwen_checkpoint, sandbox_image=args.sandbox_image,
                max_steps=6, reader_guidance="focused-v1", policy_transport=transport)
            reader.model_identity.update(tokenizer=args.tokenizer,
                                         tokenizer_revision=args.tokenizer_revision)
            return reader

        # Validate limits/config without dispatching any model request.
        from src.product.experiment_budget import ExperimentBudget
        budget = ExperimentBudget(tokenizer, bounded_transport,
                                  requests_max=args.requests_max, tokens_max=args.tokens_max)
        reader_factory(budget)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    root = args.library_root.expanduser()
    library = ArxivLibrary(root)
    discovery = ExperimentDiscovery(PaperDiscovery(root / "discovery-cache"))
    researcher = ExperimentResearcher(library, discovery, reader_factory, tokenizer,
        bounded_transport, requests_max=args.requests_max, tokens_max=args.tokens_max)
    create_server(library, discovery, researcher,
                  runs_root=root / "experiment-runs").run(transport="stdio")


if __name__ == "__main__":
    main()
