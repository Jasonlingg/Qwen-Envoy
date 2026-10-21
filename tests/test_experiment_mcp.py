"""The new tool is opt-in; source tools and invalid inputs dispatch no inference."""
import asyncio
import json

from mcp import Client

from scripts.experiment_research_mcp import create_server
from tests.arxiv_fakes import article_page, make_library
from tests.experiment_fakes import FINISH, researcher
from tests.test_arxiv_research_mcp import FakeDiscovery


def test_real_mcp_and_saved_report(tmp_path):
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    discovery = FakeDiscovery()
    worker, transport = researcher(library, discovery, ['search_papers("memory")',
        'read_paper("2609.12345", "Memory?")', "inspect", "submit", FINISH])
    server = create_server(library, discovery, worker, runs_root=tmp_path / "runs")
    async def run():
        async with Client(server) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert "research_experiment" in tools and "investigate" not in tools
            assert tools["research_experiment"].annotations.open_world_hint
            await client.call_tool("search_papers", {"query": "memory"})
            assert transport.calls == []
            invalid = await client.call_tool("research_experiment", {"question": ""})
            assert invalid.is_error and transport.calls == []
            reply = await client.call_tool("research_experiment", {
                "question": "Reduce memory?", "constraints": "Preserve quality"})
            assert not reply.is_error
            return reply.structured_content
    result = asyncio.run(run())
    assert result["status"] == "brief_ready" and len(transport.calls) == 5
    reports = list((tmp_path / "runs").glob("*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text()) == result


def test_save_failure_does_not_discard_evidence(tmp_path):
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    worker, _ = researcher(library, FakeDiscovery(), ['search_papers("memory")',
        'read_paper("2609.12345", "Memory?")', "inspect", "submit", FINISH])
    invalid_root = tmp_path / "file"
    invalid_root.write_text("not a directory")
    server = create_server(library, FakeDiscovery(), worker, runs_root=invalid_root)
    async def run():
        async with Client(server) as client:
            reply = await client.call_tool("research_experiment", {"question": "Memory?"})
            return reply.structured_content
    result = asyncio.run(run())
    assert result["evidence"] and result["save_error"] and "saved_report" not in result
