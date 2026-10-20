"""Local MCP integration with scripted model responses; no GPU or external HTTP."""
import asyncio
import json

import pytest

pytest.importorskip("mcp")
from mcp import Client

from scripts.arxiv_research_mcp import create_server
from src.product.assigned_reader import build_assigned_reader
from tests.arxiv_fakes import article_page, make_library
from tests.test_arxiv_research_mcp import FakeDiscovery
from tests.test_assigned_reader import SourceREPL


def test_mcp_returns_verified_findings_and_paper_scoped_gap_with_other_reader_failure(tmp_path):
    library, fake = make_library(tmp_path, {"2609.12345v2": {"html": article_page()},
                                           "2609.54321v1": {"html": article_page()}})
    for version in ("2609.12345v2", "2609.54321v1"):
        library.open(version)
    fetch_count = len(fake.calls)
    def transport(url, payload, headers, timeout):
        messages = payload["messages"]
        first = next(m["content"] for m in messages if m["role"] == "user")
        assert '"counts_as_source_inspection": false' in first
        if "arxiv_2609_54321v1" in first:
            raise TimeoutError("scripted second-paper failure")
        turn = sum(m["role"] == "assistant" for m in messages)
        actions = ['print(search_within("arxiv_2609_12345v2", "memory"))',
            'report_gap(missing="I have not found latency measurements.", '
            'follow_up_query="cache eviction inference latency batch size")',
            'submit(answer="Memory savings are reported; latency remains unclear.", '
            'citations=["arxiv_2609_12345v2"], '
            'evidence=[{"doc_id":"arxiv_2609_12345v2","start":0,"end":10}])']
        return {"choices": [{"message": {"content": actions[turn]}}]}
    reader = build_assigned_reader("qwen", qwen_endpoint="http://scripted", qwen_model="fake",
        qwen_checkpoint="fake-pinned", policy_transport=transport, max_steps=3)
    reader.repl_factory = lambda _: SourceREPL(library.document("2609.12345v2")["text"])
    server = create_server(library, FakeDiscovery(), investigator=reader)
    async def call():
        async with Client(server) as client:
            return await client.call_tool("investigate", {"question": "Memory and latency?",
                "arxiv_ids": ["2609.12345v2", "2609.54321v1"]})
    reply = asyncio.run(call())
    assert not reply.is_error
    result = reply.structured_content
    assert result["status"] == "evidence_found" and len(result["evidence"]) == 1
    assert result["papers"][1]["status"] == "error"
    gap, = result["research_gaps"]
    assert gap["arxiv_id"] == "2609.12345v2" and gap["doc_id"] == "arxiv_2609_12345v2"
    assert gap["review_status"] == "unreviewed_model_report" and not gap["absence_proven"]
    assert result["follow_up_owner"] == "mcp_host" and result["semantic_support"] == "not_reviewed"
    assert result["model_requests_attempted"] == 4
    assert len(fake.calls) == fetch_count  # gap recording launches no network search


def test_reasoner_receives_unreviewed_gaps_without_turning_them_into_evidence():
    from src.product.reasoner import Reasoner
    sent = []
    def transport(url, payload, headers, timeout):
        sent.append(payload)
        return {"choices": [{"message": {"content": json.dumps({
            "resolution": "unanswerable", "answer": "Unanswerable",
            "citations": [], "evidence_quotes": []})}}]}
    reasoner = Reasoner("http://scripted", "fake", "", transport=transport)
    gap = {"missing": "No latency result found.", "follow_up_query": "latency",
           "review_status": "unreviewed_model_report", "absence_proven": False}
    merged = {"question": "Latency?", "research_gaps": [gap], "evidence": [{
        "doc_id": "d", "arxiv_id": "2609.12345v2", "quote": "memory savings",
        "start": 0, "end": 14, "evidence_id": "E1"}]}
    reasoner.synthesize(merged, {"d": "memory savings"})
    content = sent[0]["messages"][1]["content"]
    packet = json.loads(content.removeprefix("EVIDENCE PACKET\n"))
    assert packet["research_gaps"] == [gap]
    assert "not evidence of absence" in packet["gap_warning"]
    assert len(packet["evidence"]) == 1 and "missing" not in packet["evidence"][0]
