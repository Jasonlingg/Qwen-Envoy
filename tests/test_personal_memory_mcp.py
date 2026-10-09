"""Protocol checks for the read-only personal-memory MCP bridge."""

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp import Client, StdioServerParameters  # noqa: E402

from scripts.personal_memory_mcp import create_server  # noqa: E402
from src.product.memory import freeze_vault  # noqa: E402
from src.research.agent import load_snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_VAULT = ROOT / "data/product_memory/sample_vault"
SCRIPT = ROOT / "scripts/personal_memory_mcp.py"


def _snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    freeze_vault(SAMPLE_VAULT, snapshot)
    return snapshot


def test_mcp_search_and_source_are_read_only_and_review_bound(tmp_path):
    snapshot = _snapshot(tmp_path)
    manifest, docs = load_snapshot(snapshot)

    async def exercise() -> None:
        async with Client(create_server(snapshot)) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "search_memory", "get_memory_source",
            }
            assert all(tool.annotations.read_only_hint for tool in listed.tools)
            assert all(tool.annotations.open_world_hint is False for tool in listed.tools)

            result = await client.call_tool("search_memory", {
                "query": "phone-drawer focus plan changes", "top_k": 5,
            })
            assert not result.is_error
            review = result.structured_content
            assert review["status"] == "evidence_found"
            assert review["corpus_hash"] == manifest["corpus_hash"]
            assert review["retriever"] == "lexical_paragraph_baseline"
            assert review["warning"]
            assert review["review_id"]
            assert len(review["evidence"]) == 3
            for item in review["evidence"]:
                assert docs[item["doc_id"]]["text"][item["start"]:item["end"]] == item["quote"]

            selected = review["evidence"][1]
            source = await client.call_tool("get_memory_source", {
                "review_id": review["review_id"], "doc_id": selected["doc_id"],
                "start": selected["start"],
                "length": selected["end"] - selected["start"],
            })
            assert not source.is_error
            assert source.structured_content["text"] == selected["quote"]
            assert source.structured_content["source_path"] == selected["source_path"]
            assert source.structured_content["corpus_hash"] == manifest["corpus_hash"]

            unreviewed_id = next(doc_id for doc_id in docs
                                 if doc_id not in {item["doc_id"] for item in review["evidence"]})
            denied = await client.call_tool("get_memory_source", {
                "review_id": review["review_id"], "doc_id": unreviewed_id,
            })
            assert denied.is_error
            unknown = await client.call_tool("get_memory_source", {
                "review_id": "not-a-review", "doc_id": selected["doc_id"],
            })
            assert unknown.is_error

            absent = await client.call_tool("search_memory", {
                "query": "xylophone quasars", "top_k": 5,
            })
            assert not absent.is_error
            assert absent.structured_content["status"] == "no_evidence"
            assert absent.structured_content["evidence"] == []
            assert absent.structured_content["uncertainty"]

    asyncio.run(exercise())


def test_mcp_rejects_snapshot_changes_after_startup(tmp_path):
    snapshot = _snapshot(tmp_path)

    async def exercise() -> None:
        async with Client(create_server(snapshot)) as client:
            result = await client.call_tool("search_memory", {
                "query": "phone-drawer focus plan changes",
            })
            review = result.structured_content
            doc_id = review["evidence"][0]["doc_id"]
            document = snapshot / "corpus" / f"{doc_id}.json"
            document.write_text(document.read_text() + "\n")
            stale = await client.call_tool("get_memory_source", {
                "review_id": review["review_id"], "doc_id": doc_id,
            })
            assert stale.is_error

    asyncio.run(exercise())


def test_stdio_client_can_query_server_from_another_working_directory(tmp_path):
    snapshot = _snapshot(tmp_path)

    async def exercise() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(SCRIPT), "--snapshot", str(snapshot)],
            cwd=tmp_path,
        )
        async with Client(params) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "search_memory", "get_memory_source",
            }
            result = await client.call_tool("search_memory", {
                "query": "phone-drawer focus plan changes", "top_k": 3,
            })
            assert not result.is_error
            review = result.structured_content
            assert review["evidence"]
            item = review["evidence"][0]
            opened = await client.call_tool("get_memory_source", {
                "review_id": review["review_id"], "doc_id": item["doc_id"],
                "start": item["start"], "length": item["end"] - item["start"],
            })
            assert not opened.is_error
            assert opened.structured_content["text"] == item["quote"]

    asyncio.run(exercise())
