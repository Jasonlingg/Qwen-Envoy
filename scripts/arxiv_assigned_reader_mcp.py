"""Opt-in arXiv MCP: assign each reader one pinned paper; never provision compute."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.arxiv_research_mcp import create_server
from src.product.arxiv_library import ArxivLibrary
from src.product.arxiv_mcp_config import DEFAULT_NEMOTRON, build_reasoner
from src.product.assigned_reader import build_assigned_reader
from src.product.paper_discovery import PaperDiscovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, default=Path("~/.envoy/arxiv-library"))
    parser.add_argument("--reader", choices=["qwen", "nemotron"], required=True)
    parser.add_argument("--qwen-endpoint")
    parser.add_argument("--qwen-model")
    parser.add_argument("--qwen-checkpoint")
    parser.add_argument("--nemotron-model", default=DEFAULT_NEMOTRON)
    parser.add_argument("--reasoner", choices=["none", "nemotron"], default="none")
    parser.add_argument("--sandbox-image", default="rlm-sandbox")
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()
    try:
        reader = build_assigned_reader(
            args.reader, qwen_endpoint=args.qwen_endpoint, qwen_model=args.qwen_model,
            qwen_checkpoint=args.qwen_checkpoint, nemotron_model=args.nemotron_model,
            sandbox_image=args.sandbox_image, max_steps=args.max_steps)
        reasoner = build_reasoner(args.reasoner, args.nemotron_model)
    except ValueError as exc:
        parser.error(str(exc))
    root = args.library_root.expanduser()
    create_server(ArxivLibrary(root), PaperDiscovery(root / "discovery-cache"),
                  investigator=reader, reasoner=reasoner).run(transport="stdio")


if __name__ == "__main__":
    main()
