"""arXiv research MCP: six read-only tools over offline fakes."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from scripts.arxiv_research_mcp import create_server  # noqa: E402
from src.product.arxiv_mcp_config import build_reader  # noqa: E402
from src.product.net import SourceUnavailable  # noqa: E402
from tests.arxiv_fakes import FakeInvestigator, article_page, make_library  # noqa: E402

QUOTE = "Eviction saves 40 percent of memory"


class FakeDiscovery:
    def __init__(self, fail=False):
        self.fail = fail

    def search(self, query, since=None, until=None, limit=10):
        if self.fail:
            raise SourceUnavailable("semantic_scholar", "rate limited")
        return {"source": "semantic_scholar", "mode": "relevance", "papers": [
            {"arxiv_id": "2609.12345", "title": "KV", "openable": True}], "coverage_note": ""}

    def citations(self, arxiv_id, direction="references", limit=20):
        return {"arxiv_id": arxiv_id, "direction": direction, "papers": []}


class FakeReasoner:
    def __init__(self, fail=False):
        self.fail = fail

    def synthesize(self, merged, documents):
        if self.fail:
            raise ValueError("reasoner quoted text outside the verified evidence")
        return {"resolution": "answer", "answer": "About 40 percent.", "evidence": []}


def _run(server, body):
    async def go():
        async with Client(server) as client:
            return await body(client)
    return asyncio.run(go())


def test_tools_are_listed_with_read_only_annotations(tmp_path):
    library, _ = make_library(tmp_path, {})
    async def body(client):
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert set(tools) == {"search_papers", "get_citations", "open_paper",
                              "find_evidence", "get_passage", "investigate"}
        assert all(tool.annotations.read_only_hint for tool in tools.values())
        assert tools["investigate"].annotations.open_world_hint is True
        assert tools["get_passage"].annotations.open_world_hint is False
    _run(create_server(library, FakeDiscovery()), body)


def test_open_find_and_read_exact_text(tmp_path):
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    async def body(client):
        opened = (await client.call_tool(
            "open_paper", {"arxiv_id": "2609.12345"})).structured_content
        assert opened["coverage"] == "html_paragraphs" and opened["status"] == "ok"
        found = (await client.call_tool("find_evidence", {
            "query": "eviction memory", "arxiv_ids": ["2609.12345v2"]})).structured_content
        hit = found["evidence"][0]
        passage = (await client.call_tool("get_passage", {
            "arxiv_id": "2609.12345v2", "start": hit["start"],
            "length": hit["end"] - hit["start"]})).structured_content
        assert passage["text"] == hit["quote"]
        bad = await client.call_tool("get_passage", {"arxiv_id": "2609.12345v2", "length": 9000})
        assert bad.is_error
    _run(create_server(library, FakeDiscovery()), body)


def test_source_unavailable_is_a_status_not_an_error(tmp_path):
    library, _ = make_library(tmp_path, {})
    async def body(client):
        result = await client.call_tool("search_papers", {"query": "kv cache"})
        assert not result.is_error
        assert result.structured_content["status"] == "source_unavailable"
        assert result.structured_content["source"] == "semantic_scholar"
    _run(create_server(library, FakeDiscovery(fail=True)), body)


def test_investigate_without_reader_reports_not_configured(tmp_path):
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    async def body(client):
        result = (await client.call_tool("investigate", {
            "question": "How much memory?", "arxiv_ids": ["2609.12345v2"]})).structured_content
        assert result["status"] == "model_not_configured"
    _run(create_server(library, FakeDiscovery()), body)


def test_investigate_with_synthesis_and_failed_synthesis(tmp_path):
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    reader = FakeInvestigator({"2609.12345v2": QUOTE})
    async def body(client):
        return (await client.call_tool("investigate", {
            "question": "How much memory?", "arxiv_ids": ["2609.12345"],
            "synthesize": True})).structured_content
    ok = _run(create_server(library, FakeDiscovery(), investigator=reader,
                            reasoner=FakeReasoner()), body)
    assert ok["status"] == "evidence_found" and ok["synthesis"]["answer"] == "About 40 percent."
    failed = _run(create_server(library, FakeDiscovery(), investigator=reader,
                                reasoner=FakeReasoner(fail=True)), body)
    assert failed["synthesis"] is None and "outside the verified" in failed["synthesis_error"]
    assert failed["evidence"]  # the verified packet still comes back


def test_too_many_papers_is_rejected_before_any_fetch(tmp_path):
    library, fake = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    async def body(client):
        return await client.call_tool("investigate", {
            "question": "q", "arxiv_ids": ["2609.12345v2"] * 6})
    result = _run(create_server(library, FakeDiscovery(),
                                investigator=FakeInvestigator({})), body)
    assert result.is_error and fake.calls == []


def test_reader_configuration_errors(monkeypatch):
    assert build_reader("none") is None
    with pytest.raises(ValueError, match="qwen"):
        build_reader("qwen", qwen_endpoint="http://127.0.0.1:8000/v1")
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NEBIUS_API_KEY"):
        build_reader("nemotron")
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    reader = build_reader("nemotron")
    assert reader.harness_version == "v2" and reader.protocol_variant == "nemotron"


@pytest.mark.parametrize('kind', ['qwen', 'nemotron'])
@pytest.mark.parametrize('harness', ['v1', 'v2'])
def test_effective_prompt_and_cleaner_are_identical_across_layers(monkeypatch, kind, harness):
    import hashlib

    from src.product.harness_v2 import clean_v2_action
    from src.product.qwen_investigator import effective_system_prompt
    monkeypatch.setenv('NEBIUS_API_KEY', 'test')
    reader = build_reader(kind, qwen_endpoint='http://localhost:8000/v1', qwen_model='fake',
                          qwen_checkpoint='pinned-fake', harness_version=harness)
    policy = reader.policy_factory()
    assert reader.system_prompt == policy.system_prompt == effective_system_prompt(
        'public_papers', reader.protocol_variant, harness)
    assert hashlib.sha256(policy.system_prompt.encode()).hexdigest() == hashlib.sha256(
        reader.system_prompt.encode()).hexdigest()
    if harness == 'v2':
        assert policy._action_cleaner is clean_v2_action
        assert 'Never call submit()' not in policy.system_prompt
