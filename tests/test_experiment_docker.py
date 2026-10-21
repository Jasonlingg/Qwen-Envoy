"""Real MCP and network-disabled Docker; scripted Qwen responses, no paid inference."""
import asyncio
import os

import pytest
from mcp import Client

from scripts.experiment_research_mcp import create_server
from src.env.repl import PersistentREPL
from src.product.assigned_reader import build_assigned_reader
from src.product.experiment_research import ExperimentResearcher
from tests.arxiv_fakes import article_page, make_library
from tests.experiment_fakes import FINISH, ScriptedTransport
from tests.test_arxiv_research_mcp import FakeDiscovery
from tests.test_experiment_budget import Tokenizer

IMAGE = os.environ.get("ENVOY_ASSIGNED_TEST_DOCKER_IMAGE")
pytestmark = pytest.mark.skipif(not IMAGE, reason="set ENVOY_ASSIGNED_TEST_DOCKER_IMAGE")


@pytest.mark.parametrize("malicious", [False, True])
def test_search_read_finish_via_mcp_and_real_docker(tmp_path, malicious):
    assert PersistentREPL._docker_available(IMAGE)
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    library.open("2609.12345")
    source = library.document("2609.12345")["text"]
    quote = "Eviction saves 40 percent of memory"
    start = source.index(quote)
    marker = tmp_path / "host-marker"
    actions = ['search_papers("memory")', 'read_paper("2609.12345", "Memory savings?")']
    if malicious:
        actions.append(f'__import__("pathlib").Path({str(marker)!r}).write_text("bad")')
    actions.extend(['print(search_within("arxiv_2609_12345v2", "memory eviction"))',
        f'print(passage("arxiv_2609_12345v2", {start}, {len(quote)}))',
        'submit(answer="Memory savings are reported.", citations=["arxiv_2609_12345v2"], '
        f'evidence=[{{"doc_id":"arxiv_2609_12345v2","start":{start},'
        f'"end":{start + len(quote)}}}])', FINISH])
    transport = ScriptedTransport(actions)
    def factory(budget):
        return build_assigned_reader("qwen", qwen_endpoint="http://localhost:8000/v1",
            qwen_model="fixture", qwen_checkpoint="fixture-not-a-model", max_steps=6,
            reader_guidance="focused-v1", sandbox_image=IMAGE, policy_transport=budget)
    discovery = FakeDiscovery()
    worker = ExperimentResearcher(library, discovery, factory, Tokenizer(), transport)
    async def run():
        async with Client(create_server(library, discovery, worker)) as client:
            reply = await client.call_tool("research_experiment", {"question": "Memory?"})
            assert not reply.is_error
            return reply.structured_content
    result = asyncio.run(run())
    assert result["status"] == "brief_ready" and result["evidence"][0]["quote"] == quote
    assert result["reader_runs"][0]["execution"] == "docker_only"
    assert len(transport.calls) == (7 if malicious else 6)
    assert result["budget"]["requests_dispatched"] == len(transport.calls)
    assert not marker.exists()
    if malicious:
        assert "Action rejected" in result["reader_runs"][0]["trajectory"][0]["observation"]
