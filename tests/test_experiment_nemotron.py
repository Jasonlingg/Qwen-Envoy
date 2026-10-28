"""The optional Nebius boundary is tested without paid requests."""

import asyncio
import json

import pytest
from mcp import Client

from scripts.experiment_mode_extended_mcp import create_server
from src.product.experiment_nemotron import (
    ENDPOINT,
    MAX_OUTPUT_TOKENS,
    MODEL,
    explain_public_research,
)
from src.product.ml_experiment_store import ExperimentStore
from tests.test_arxiv_research_mcp import FakeDiscovery
from tests.test_ml_experiment_evidence import sources
from tests.test_ml_experiment_research_notes import note


def _record(tmp_path, *, data_kind="recorded", paper=True):
    library, _, resolver, refs = sources(tmp_path)
    store = ExperimentStore(tmp_path / "records", evidence_resolver=resolver)
    saved = store.save_brief({"question": "Which memory method fits?"})
    research = note(refs[:1] if paper else refs[1:2])
    research["data_kind"] = data_kind
    if not paper:
        research["findings"] = []
    store.record_research(saved["experiment_id"], 1, research)
    return store, library, saved["experiment_id"]


def test_one_capped_public_paper_call_and_valid_reference_checks(tmp_path):
    store, _, key = _record(tmp_path)
    calls = []

    def transport(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        packet = json.loads(payload["messages"][1]["content"])
        assert len(packet["evidence"]) == 1
        assert packet["evidence"][0]["source_url"].startswith("https://arxiv.org/")
        assert "local_artifact" not in payload["messages"][1]["content"]
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "answer": "The excerpt mentions memory [E1].",
            "claims": [{"text": "The excerpt mentions memory", "evidence_ids": ["E1"]}],
            "limitations": ["No workload measurement is supplied."]})}}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 70}}

    result = explain_public_research(store, key, "memory_search", "test-key",
                                     transport=transport)
    assert len(calls) == result["request_count"] == 1
    assert calls[0][0] == ENDPOINT
    assert calls[0][1]["model"] == MODEL
    assert calls[0][1]["max_tokens"] == MAX_OUTPUT_TOKENS
    assert calls[0][2]["Authorization"] == "Bearer test-key"
    assert result["checks"]["claims_with_valid_evidence"] == 1
    assert result["checks"]["semantic_support"] == "not_reviewed"


@pytest.mark.parametrize("kind", ["synthetic", "no_paper"])
def test_ineligible_note_never_calls_model(tmp_path, kind):
    store, _, key = _record(tmp_path, data_kind=("synthetic" if kind == "synthetic"
                                                 else "recorded"), paper=kind != "no_paper")

    def forbidden(*_):
        pytest.fail("model called")

    with pytest.raises(ValueError):
        explain_public_research(store, key, "memory_search", "test-key",
                                transport=forbidden)


def test_opt_in_mcp_tool_uses_selected_note_only(tmp_path):
    store, library, key = _record(tmp_path)
    requests = []

    def transport(url, payload, headers, timeout):
        requests.append(payload)
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "answer": "Insufficient evidence [E1].",
            "claims": [{"text": "The passage discusses memory", "evidence_ids": ["E1"]}],
            "limitations": ["Review support manually."]})}}]}

    async def run():
        async with Client(create_server(store, library, FakeDiscovery(),
                                        nemotron_key="test-key",
                                        nemotron_transport=transport)) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            assert "explain_experiment_research" in tools
            result = await client.call_tool("explain_experiment_research", {
                "experiment_id": key, "note_id": "memory_search"})
            assert not result.is_error
            assert result.structured_content["semantic_support"] == "not_reviewed"
    asyncio.run(run())
    assert len(requests) == 1
