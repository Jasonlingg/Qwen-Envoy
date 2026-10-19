"""Explicit model-free MCP profile, with unchanged default tool availability."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from scripts import arxiv_research_mcp as mcp_script  # noqa: E402
from tests.arxiv_fakes import article_page, make_library  # noqa: E402
from tests.test_arxiv_research_mcp import FakeDiscovery  # noqa: E402

SOURCES = {"search_papers", "get_citations", "open_paper", "find_evidence", "get_passage"}


def test_sources_profile_exposes_only_source_tools_and_instructions(tmp_path):
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    server = mcp_script.create_server(library, FakeDiscovery(), tool_profile="sources")

    async def body():
        async with Client(server) as client:
            tools = (await client.list_tools()).tools
            assert {tool.name for tool in tools} == SOURCES
            assert all(tool.annotations.read_only_hint for tool in tools)
            advertised = client.instructions + " ".join(tool.description or "" for tool in tools)
            assert all(word not in advertised.lower() for word in
                       ["investigate", "reader", "qwen", "reasoner"])
            assert "provenance, not support" in client.instructions
            opened = (await client.call_tool("open_paper", {
                "arxiv_id": "2609.12345v2"})).structured_content
            assert opened["status"] == "ok"
            passage = (await client.call_tool("get_passage", {
                "arxiv_id": "2609.12345v2", "start": 0, "length": 10})).structured_content
            assert passage["text"] == library.document("2609.12345v2")["text"][:10]
            hidden = await client.call_tool("investigate", {
                "question": "memory?", "arxiv_ids": ["2609.12345v2"]})
            assert hidden.is_error

    asyncio.run(body())


@pytest.mark.parametrize("kwargs", [
    {"tool_profile": "unknown"},
    {"tool_profile": "sources", "investigator": object()},
    {"tool_profile": "sources", "reasoner": object()},
])
def test_invalid_or_model_enabled_source_profile_is_rejected(kwargs):
    with pytest.raises(ValueError):
        mcp_script.create_server(None, None, **kwargs)


@pytest.mark.parametrize("model_flag", ["--reader", "--reasoner"])
def test_source_cli_rejects_model_opt_in_before_construction(monkeypatch, capsys, model_flag):
    def unexpected(*args, **kwargs):
        pytest.fail("model construction must not occur for conflicting profile options")

    monkeypatch.setattr(mcp_script, "build_reader", unexpected)
    monkeypatch.setattr(mcp_script, "build_reasoner", unexpected)
    monkeypatch.setattr("sys.argv", ["arxiv_research_mcp.py", "--tool-profile", "sources",
                                     model_flag, "nemotron"])
    with pytest.raises(SystemExit) as exc:
        mcp_script.main()
    assert exc.value.code == 2
    assert "requires --reader none and --reasoner none" in capsys.readouterr().err


def test_source_cli_wires_explicit_profile_without_model_configuration(monkeypatch, tmp_path):
    calls = []

    class Server:
        def run(self, *, transport):
            calls.append(("run", transport))

    def source_server(library, discovery, **kwargs):
        calls.append(("server", kwargs))
        return Server()

    monkeypatch.setattr(mcp_script, "ArxivLibrary", lambda root: object())
    monkeypatch.setattr(mcp_script, "PaperDiscovery", lambda root: object())
    monkeypatch.setattr(mcp_script, "build_reader", lambda kind, **kwargs: (
        calls.append(("reader", kind)) or None))
    monkeypatch.setattr(mcp_script, "build_reasoner", lambda kind, model: (
        calls.append(("reasoner", kind)) or None))
    monkeypatch.setattr(mcp_script, "create_server", source_server)
    monkeypatch.setattr("sys.argv", ["arxiv_research_mcp.py", "--tool-profile", "sources",
                                     "--library-root", str(tmp_path)])
    mcp_script.main()
    assert calls == [("reader", "none"), ("reasoner", "none"),
                     ("server", {"investigator": None, "reasoner": None,
                                 "tool_profile": "sources"}), ("run", "stdio")]
