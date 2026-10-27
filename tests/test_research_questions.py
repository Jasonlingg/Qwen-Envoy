"""A research question links an inspected paper to a saved attempt, across sessions."""

import asyncio

import pytest
from mcp import Client

from scripts.experiment_mode_mcp import create_server
from src.product.ml_experiment_store import ExperimentStore
from src.product.research_questions import ResearchQuestions
from tests.ml_experiment_result_fixtures import paired_result, planned_brief
from tests.test_arxiv_research_mcp import FakeDiscovery
from tests.test_ml_experiment_evidence import sources
from tests.test_ml_experiment_research_notes import note


def _records(tmp_path):
    library, _, resolver, refs = sources(tmp_path)
    root = tmp_path / "records"
    store = ExperimentStore(root, evidence_resolver=resolver)
    source = store.save_brief({"question": "Could this paper help memory?"})
    store.record_research(source["experiment_id"], 1, note(refs[:1]))
    attempt = store.save_brief(planned_brief())
    frozen = store.freeze_plan(attempt["experiment_id"], 1)
    result = paired_result()
    store.record_result(attempt["experiment_id"], frozen["plan_sha256"], result)
    link = {"link_id": "paper_to_latency_run", "paper": {
        "experiment_id": source["experiment_id"], "research_note_id": "memory_search",
        "reference_id": "paper"}, "result": {
        "experiment_id": attempt["experiment_id"], "run_id": result["run_id"]},
        "method_tried": "Applied the paper's memory idea to the synthetic serving test.",
        "observed_outcome": "The synthetic fixture reports lower latency.",
        "interpretation": "This fixture cannot establish a real speedup.",
        "next_step": "Measure real warm and cold requests.",
        "author_kind": "agent_proposed"}
    return library, root, store, link


def test_question_trail_survives_restart_and_rejects_false_links(tmp_path):
    _, root, store, link = _records(tmp_path)
    questions = ResearchQuestions(root, store)
    opened = questions.start("Does a memory method help our Qwen reader?")
    question_id = opened["question_id"]
    assert opened["revision"] == 1 and opened["current_view"] is None
    linked = questions.link(question_id, 1, link)
    receipt = linked["paper_to_result_links"][0]["receipt"]
    assert receipt["paper_quote"] and receipt["paper_source_url"].startswith("https://")
    assert receipt["result_data_kind"] == "synthetic"
    assert questions.link(question_id, 1, link) == linked  # Stable retry.
    with pytest.raises(ValueError, match="different immutable"):
        questions.link(question_id, 2, {**link, "interpretation": "Changed"})
    with pytest.raises(ValueError, match="Stale"):
        questions.link(question_id, 1, {**link, "link_id": "another"})
    with pytest.raises(ValueError, match="verified paper span"):
        questions.link(question_id, 2, {**link, "link_id": "bad_source",
                                      "paper": {**link["paper"], "reference_id": "missing"}})
    assert questions.get(question_id) == linked

    view = {"view_id": "after_pilot", "statement": "A real speedup remains unknown.",
            "reason": "The available result is synthetic, not a measured reader run.",
            "next_question": "What happens on matched real requests?",
            "basis_link_ids": [link["link_id"]], "author_kind": "agent_proposed"}
    with pytest.raises(ValueError, match="basis"):
        questions.record_view(question_id, 2, {**view, "basis_link_ids": ["other"]})
    updated = questions.record_view(question_id, 2, view)
    assert updated["current_view"]["view"]["next_question"].startswith("What happens")
    assert questions.record_view(question_id, 2, view) == updated
    restarted = ResearchQuestions(root, ExperimentStore(root))
    assert restarted.get(question_id) == updated
    assert restarted.find("Qwen memory")["matches"][0]["question_id"] == question_id
    assert restarted.find()["matches"][0]["current_view"] == view["statement"]
    assert "not causation" in updated["warning"]


def test_question_tools_work_over_mcp_after_restart(tmp_path):
    library, root, store, link = _records(tmp_path)

    async def run():
        async with Client(create_server(store, library, FakeDiscovery())) as client:
            created = await client.call_tool("start_research_question", {
                "question": "Does the paper's memory idea help Qwen?"})
            assert not created.is_error
            key = created.structured_content["question_id"]
            linked = await client.call_tool("link_paper_to_result", {
                "question_id": key, "expected_revision": 1, "link": link})
            assert not linked.is_error
            assert linked.structured_content["paper_to_result_links"][0]["receipt"][
                "result_data_kind"] == "synthetic"
        async with Client(create_server(ExperimentStore(root), library,
                                        FakeDiscovery())) as client:
            found = await client.call_tool("find_research_questions", {"query": "Qwen memory"})
            assert found.structured_content["matches"][0]["question_id"] == key
            current = await client.call_tool("get_research_question", {"question_id": key})
            assert current.structured_content["revision"] == 2
            rejected = await client.call_tool("record_question_view", {
                "question_id": key, "expected_revision": 1,
                "view": {"view_id": "stale", "statement": "Unknown", "reason": "Stale",
                         "next_question": "Recheck", "author_kind": "agent_proposed"}})
            assert rejected.is_error

    asyncio.run(run())
