"""Explicit full research MCP with the pinned Qwen v5 reader on Runpod.

No reasoner, no deployment, no inference retries. Credentials are read only by
this server, never placed in the host model's configuration or sandbox.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.arxiv_research_mcp import create_server
from scripts.runpod_qwen_bootstrap import BASE, BASE_REVISION, ADAPTER, ADAPTER_REVISION, SUBFOLDER
from src.policies.runpod_queue import RunpodQueueTransport
from src.product.arxiv_library import ArxivLibrary
from src.product.arxiv_mcp_config import build_reader
from src.product.paper_discovery import PaperDiscovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--audit-path", type=Path, required=True,
                        help="New exclusive file; restart requires a new explicitly budgeted session")
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--sandbox-image", default="rlm-sandbox")
    args = parser.parse_args()
    key = os.environ.get("RUNPOD_API_KEY")
    if not key and args.credentials_file:
        key = tomllib.loads(args.credentials_file.expanduser().read_text()).get("apikey")
    if not key:
        parser.error("RUNPOD_API_KEY or --credentials-file required")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION,
                                               trust_remote_code=False)
    transport = RunpodQueueTransport(args.endpoint_id, key, tokenizer, args.audit_path)
    reader = build_reader("qwen", qwen_endpoint=f"https://api.runpod.ai/v2/{args.endpoint_id}",
                          qwen_model="envoy",
                          qwen_checkpoint=f"{ADAPTER}@{ADAPTER_REVISION}/{SUBFOLDER}",
                          harness_version="v2", max_steps=8,
                          policy_transport=transport, sandbox_image=args.sandbox_image)
    root = args.library_root.expanduser()
    create_server(ArxivLibrary(root), PaperDiscovery(root / "discovery-cache"),
                  investigator=reader, reasoner=None, tool_profile="full").run(transport="stdio")


if __name__ == "__main__":
    main()
