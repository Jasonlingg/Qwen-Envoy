"""The second-brain draft has an explicit disclosure and citation boundary."""

import asyncio
import json

import pytest
from mcp import Client

from scripts.experiment_mode_mcp import create_server
from src.product.experiment_nemotron import ENDPOINT, MAX_OUTPUT_TOKENS, MODEL
from src.product.research_question_nemotron import draft_question
from src.product.research_questions import ResearchQuestions
from tests.test_arxiv_research_mcp import FakeDiscovery
from tests.test_research_questions import _records


def test_question_draft_requires_consent_and_never_saves_model_output(tmp_path):
    library, root, store, link = _records(tmp_path)
    questions = ResearchQuestions(root, store)
    opened = questions.start("Does this paper's memory method help Qwen?")
    key = opened["question_id"]
    questions.consider_paper(key, 1, {"paper_id": "method", "source": link["paper"],
        "relevance": "Possible memory reduction.", "limitation": "Different task.",
        "author_kind": "agent_proposed"})
    before = questions.link(key, 2, link)
    calls = []

    def transport(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        packet = json.loads(payload["messages"][1]["content"])
        assert [item["source_kind"] for item in packet["evidence"]] == [
            "verified_paper_span", "saved_result_summary"]
        assert packet["evidence"][1]["data_kind"] == "synthetic"
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "answer": "The saved example is synthetic [E2]; a real test remains open.",
            "claims": [{"text": "The saved example is synthetic", "evidence_ids": ["E2"]}],
            "limitations": ["No measured Qwen improvement is established."]})}}],
            "usage": {"prompt_tokens": 800, "completion_tokens": 120}}

    with pytest.raises(ValueError, match="share_local_context"):
        draft_question(questions, key, "test-key", transport=transport)
    assert not calls
    result = draft_question(questions, key, "test-key", share_local_context=True,
                            transport=transport)
    assert len(calls) == result["request_count"] == 1
    assert calls[0][0] == ENDPOINT and calls[0][1]["model"] == MODEL
    assert calls[0][1]["max_tokens"] == MAX_OUTPUT_TOKENS
    assert result["checks"]["semantic_support"] == "not_reviewed"
    assert result["status"] == "unreviewed_not_saved"
    assert questions.get(key) == before

    async def via_mcp():
        async with Client(create_server(store, library, FakeDiscovery(),
                                        nemotron_key="test-key",
                                        nemotron_transport=transport)) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            assert "draft_research_question" in tools
            denied = await client.call_tool("draft_research_question", {"question_id": key})
            assert denied.is_error
            called = await client.call_tool("draft_research_question", {
                "question_id": key, "share_local_context": True})
            assert not called.is_error
            assert called.structured_content["status"] == "unreviewed_not_saved"

    asyncio.run(via_mcp())
    assert len(calls) == 2


def test_question_draft_rejects_unknown_evidence_and_truncation(tmp_path):
    _, root, store, link = _records(tmp_path)
    questions = ResearchQuestions(root, store)
    key = questions.start("Can the paper guide a test?")["question_id"]
    questions.link(key, 1, link)

    def bad_reference(*_):
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "answer": "Unsupported [E9]", "claims": [{"text": "Unsupported",
                                                   "evidence_ids": ["E9"]}],
            "limitations": []})}}]}

    with pytest.raises(ValueError, match="unknown evidence"):
        draft_question(questions, key, "test-key", share_local_context=True,
                       transport=bad_reference)

    def truncated(*_):
        return {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}

    with pytest.raises(ValueError, match="output-token cap"):
        draft_question(questions, key, "test-key", share_local_context=True,
                       transport=truncated)
