"""Optional real Docker/MCP regression with scripted responses, never paid inference."""
import asyncio
import os

import pytest

pytest.importorskip("mcp")
from mcp import Client

from scripts.arxiv_research_mcp import create_server
from src.env.repl import PersistentREPL
from src.product.assigned_reader import build_assigned_reader
from tests.arxiv_fakes import article_page, make_library
from tests.test_arxiv_research_mcp import FakeDiscovery

IMAGE = os.environ.get("ENVOY_ASSIGNED_TEST_DOCKER_IMAGE")
pytestmark = pytest.mark.skipif(not IMAGE, reason="set ENVOY_ASSIGNED_TEST_DOCKER_IMAGE")


@pytest.mark.parametrize("dead_code", [False, True])
def test_scripted_mcp_with_real_docker_source_access(tmp_path, dead_code):
    assert PersistentREPL._docker_available(IMAGE), "configured Docker image must be available"
    library, fake = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    library.open("2609.12345v2")
    source = library.document("2609.12345v2")["text"]
    quote = "Eviction saves 40 percent of memory"
    start = source.index(quote)
    first = ('if False:\n    read("arxiv_2609_12345v2")\nprint("Metadata only.")'
             if dead_code else 'print(search_within("arxiv_2609_12345v2", "eviction memory"))')
    actions = iter([first, 'report_gap(missing="Latency has not been established.", '
        'follow_up_query="cache eviction latency measurements")',
        'submit(answer="Memory savings are reported.", citations=["arxiv_2609_12345v2"], '
        f'evidence=[{{"doc_id":"arxiv_2609_12345v2","start":{start},'
        f'"end":{start + len(quote)}}}])'])
    scripted_requests = []
    def transport(url, payload, headers, timeout):
        scripted_requests.append(payload)
        return {"choices": [{"message": {"content": next(actions)}}]}
    reader = build_assigned_reader("qwen", qwen_endpoint="http://scripted-no-model",
        qwen_model="fake", qwen_checkpoint="fake", policy_transport=transport,
        max_steps=3, sandbox_image=IMAGE)
    server = create_server(library, FakeDiscovery(), investigator=reader)
    fetches = len(fake.calls)
    async def call():
        async with Client(server) as client:
            reply = await client.call_tool("investigate", {"question": "Memory and latency?",
                "arxiv_ids": ["2609.12345v2"]})
            assert not reply.is_error
            return reply.structured_content
    result = asyncio.run(call())
    assert len(scripted_requests) == 3 and len(fake.calls) == fetches
    if dead_code:
        assert result["status"] == "incomplete"
        assert result["evidence"] == result["research_gaps"] == []
    else:
        assert result["status"] == "evidence_found"
        assert result["evidence"][0]["quote"] == quote
        assert len(result["research_gaps"]) == 1
        assert result["research_gaps"][0]["absence_proven"] is False
