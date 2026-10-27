"""Experiment-mode MCP plus source-bound cross-paper decision justifications."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.arxiv_research_mcp import LOCAL  # noqa: E402
from scripts.experiment_mode_mcp import create_server as experiment_server  # noqa: E402
from scripts.ml_experiment_mcp import _errors  # noqa: E402
from src.product.arxiv_library import ArxivLibrary  # noqa: E402
from src.product.experiment_discovery import ExperimentDiscovery  # noqa: E402
from src.product.ml_experiment_evidence import EvidenceResolver  # noqa: E402
from src.product.ml_experiment_store import ExperimentStore  # noqa: E402
from src.product.paper_discovery import PaperDiscovery  # noqa: E402
from src.product.research_justification import justification_packet  # noqa: E402


def create_server(store, library, discovery, *, nemotron_key=None,
                  nemotron_transport=None):
    """Add one read-only view while retaining the pinned experiment-mode file."""
    server = experiment_server(store, library, discovery, nemotron_key=nemotron_key,
                               nemotron_transport=nemotron_transport)

    @server.tool(annotations=LOCAL)
    @_errors
    def get_research_justification(experiment_id: str, note_id: str) -> dict[str, Any]:
        """Read a saved cross-paper comparison with exact source receipts.

        Comparisons are caller-authored; quote validity does not prove support.
        """
        view = store.get(experiment_id, research_note_id=note_id)
        return justification_packet(view["selected_research_note"])

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("~/.envoy/ml-experiments"))
    parser.add_argument("--library-root", type=Path, default=Path("~/.envoy/arxiv-library"))
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--enable-nemotron", action="store_true",
                        help="Opt in to bounded Nebius tools; calls still require invocation")
    parser.add_argument("--nebius-env-file", type=Path,
                        help="Read only NEBIUS_API_KEY from this existing private .env file")
    args = parser.parse_args()
    nemotron_key = None
    if args.enable_nemotron:
        from dotenv import dotenv_values
        path = args.nebius_env_file or ROOT / ".env"
        nemotron_key = os.environ.get("NEBIUS_API_KEY") or dotenv_values(path).get(
            "NEBIUS_API_KEY")
        if not nemotron_key:
            parser.error("--enable-nemotron needs NEBIUS_API_KEY")
    library = ArxivLibrary(args.library_root.expanduser())
    store = ExperimentStore(args.store_root, evidence_resolver=EvidenceResolver(
        library_root=library.root, artifact_root=args.artifact_root))
    discovery = ExperimentDiscovery(PaperDiscovery(library.root / "discovery-cache"))
    create_server(store, library, discovery,
                  nemotron_key=nemotron_key).run(transport="stdio")


if __name__ == "__main__":
    main()
