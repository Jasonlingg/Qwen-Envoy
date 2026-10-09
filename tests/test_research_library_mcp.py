"""Read-only public-paper MCP tools and the optional code-execution handoff."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp import Client, StdioServerParameters  # noqa: E402

from scripts.research_library_mcp import (  # noqa: E402
    PAPER_PROMPT_HASH,
    TOOL_PREAMBLE_HASH,
    create_server,
)
from src.env.tools import SEARCH_PROTOCOL_VERSION  # noqa: E402
from src.eval.artifacts import content_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research_library_mcp.py"


def _snapshot(tmp_path: Path) -> tuple[Path, dict[str, dict]]:
    snapshot = tmp_path / "paper-snapshot"
    corpus = snapshot / "corpus"
    corpus.mkdir(parents=True)
    texts = {
        "arxiv_2609_11111v1": (
            "A retrieval worker searches the library. It then reads exact source passages."
        ),
        "arxiv_2609_22222v1": (
            "A verifier checks citations. It cannot prove that a claim is supported."
        ),
    }
    docs = {}
    papers = []
    for index, (doc_id, source) in enumerate(texts.items(), 1):
        arxiv_id = f"2609.{index * 11111:05d}v1"
        doc = {
            "doc_id": doc_id,
            "title": "Retrieval worker" if index == 1 else "Citation verifier",
            "text": source,
            "sections": [{"section": "Abstract", "start": 0, "end": len(source)}],
            "metadata": {
                "arxiv_id": arxiv_id,
                "source_url": f"https://arxiv.org/abs/{arxiv_id}",
                "submitted": "2026-09-22",
                "coverage": "abstract_only",
            },
        }
        (corpus / f"{doc_id}.json").write_text(json.dumps(doc), encoding="utf-8")
        docs[doc_id] = doc
        papers.append({"doc_id": doc_id, "arxiv_id": arxiv_id})
    manifest = {
        "schema_version": "research-snapshot-v1",
        "status": "complete",
        "corpus_hash": content_hash(corpus),
        "papers": papers,
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot, docs


def test_paper_search_source_access_and_unconfigured_qwen(tmp_path):
    snapshot, docs = _snapshot(tmp_path)

    async def exercise() -> None:
        async with Client(create_server(snapshot)) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "search_research_papers", "get_research_source",
                "investigate_research_question",
            }
            assert all(tool.annotations.read_only_hint for tool in listed.tools)
            investigate = next(tool for tool in listed.tools
                               if tool.name == "investigate_research_question")
            assert investigate.annotations.open_world_hint is True

            found = await client.call_tool("search_research_papers", {
                "query": "retrieval worker searches", "top_k": 2,
            })
            assert not found.is_error
            review = found.structured_content
            assert review["status"] == "evidence_found"
            assert review["retriever"] == "lexical_paper_search"
            hit = review["evidence"][0]
            assert docs[hit["doc_id"]]["text"][hit["start"]:hit["end"]] == hit["quote"]

            source = await client.call_tool("get_research_source", {
                "review_id": review["review_id"], "doc_id": hit["doc_id"],
                "start": hit["start"], "length": hit["end"] - hit["start"],
            })
            assert not source.is_error
            assert source.structured_content["text"] == hit["quote"]
            other = next(doc_id for doc_id in docs if doc_id != hit["doc_id"])
            denied = await client.call_tool("get_research_source", {
                "review_id": review["review_id"], "doc_id": other,
            })
            assert denied.is_error

            not_configured = await client.call_tool("investigate_research_question", {
                "question": "How does the retrieval worker search?",
            })
            assert not not_configured.is_error
            assert not_configured.structured_content["status"] == "model_not_configured"
            assert not_configured.structured_content["model_requests_attempted"] == 0

            changed = snapshot / "corpus" / f"{hit['doc_id']}.json"
            changed.write_text(changed.read_text() + "\n", encoding="utf-8")
            stale = await client.call_tool("search_research_papers", {
                "query": "retrieval worker",
            })
            assert stale.is_error

    asyncio.run(exercise())


def test_qwen_packet_is_rechecked_against_pinned_papers(tmp_path):
    snapshot, docs = _snapshot(tmp_path)
    doc_id, doc = next(iter(docs.items()))
    quote = "reads exact source passages"
    start = doc["text"].index(quote)

    class FakeInvestigator:
        def __init__(self) -> None:
            self.calls = 0
            self.quote = quote
            self.tool_hash = TOOL_PREAMBLE_HASH

        def investigate(self, question: str, selected_snapshot: Path) -> dict:
            self.calls += 1
            assert selected_snapshot == snapshot
            assert question == "What does the worker read?"
            return {
                "status": "evidence_found", "source_domain": "public_papers",
                "corpus_hash": content_hash(snapshot / "corpus"),
                "candidate_claims": [{"text": "It reads source passages."}],
                "evidence": [{"doc_id": doc_id, "start": start,
                              "end": start + len(quote), "quote": self.quote}],
                "model_identity": {"checkpoint": "fake-checkpoint"},
                "model_requests_attempted": 2,
                "trajectory": [{"step": 1}, {"step": 2}],
                "system_prompt_sha256": PAPER_PROMPT_HASH,
                "tool_search_version": SEARCH_PROTOCOL_VERSION,
                "tool_preamble_sha256": self.tool_hash,
            }

    worker = FakeInvestigator()

    async def exercise() -> None:
        async with Client(create_server(snapshot, investigator=worker)) as client:
            result = await client.call_tool("investigate_research_question", {
                "question": "What does the worker read?",
            })
            assert not result.is_error
            packet = result.structured_content
            assert packet["candidate_answer"] == "It reads source passages."
            assert packet["semantic_support"] == "not_reviewed"
            assert packet["evidence"][0]["quote"] == quote
            assert packet["evidence"][0]["source_url"] == doc["metadata"]["source_url"]
            assert packet["prompt_hash"] == PAPER_PROMPT_HASH
            assert packet["tool_search_version"] == SEARCH_PROTOCOL_VERSION
            assert worker.calls == 1

            opened = await client.call_tool("get_research_source", {
                "review_id": packet["review_id"], "doc_id": doc_id,
                "start": start, "length": len(quote),
            })
            assert not opened.is_error
            assert opened.structured_content["text"] == quote

            worker.tool_hash = "0" * 64
            wrong_tools = await client.call_tool("investigate_research_question", {
                "question": "What does the worker read?",
            })
            assert wrong_tools.is_error
            worker.tool_hash = TOOL_PREAMBLE_HASH
            worker.quote = "fabricated quotation"
            rejected = await client.call_tool("investigate_research_question", {
                "question": "What does the worker read?",
            })
            assert rejected.is_error
            assert worker.calls == 3

    asyncio.run(exercise())


def test_stdio_paper_server_has_no_model_calls(tmp_path):
    snapshot, _ = _snapshot(tmp_path)

    async def exercise() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(SCRIPT), "--snapshot", str(snapshot)],
            cwd=tmp_path,
        )
        async with Client(params) as client:
            result = await client.call_tool("search_research_papers", {
                "query": "retrieval worker", "top_k": 1,
            })
            assert not result.is_error
            assert result.structured_content["evidence"]

    asyncio.run(exercise())


def test_paper_server_rejects_manifest_identity_mismatch(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    path = snapshot / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["papers"][0]["arxiv_id"] = "2609.99999v1"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest disagrees"):
        create_server(snapshot)
