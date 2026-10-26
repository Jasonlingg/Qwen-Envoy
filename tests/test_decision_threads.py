"""Durable decisions stay linked to the exact attempts they cite."""

import asyncio
import json

import pytest
from mcp import Client

from scripts.experiment_mode_mcp import create_server
from src.product.decision_threads import DecisionThreads
from src.product.ml_experiment_store import ExperimentStore
from tests.ml_experiment_result_fixtures import paired_result, planned_brief
from tests.test_arxiv_research_mcp import FakeDiscovery
from tests.test_ml_experiment_evidence import sources


def _decision(decision_id, outcome, basis, *, author="agent_proposed", revisit=None):
    return {"decision_id": decision_id, "outcome": outcome,
            "rationale": "This is a stated choice, not a scientific verdict.",
            "revisit_when": revisit, "author_kind": author, "basis": basis}


def test_thread_links_attempts_and_decisions_across_restart(tmp_path):
    root = tmp_path / "records"
    store = ExperimentStore(root)
    first = store.save_brief({"question": "Should two Qwen readers run concurrently?"})
    first_id = first["experiment_id"]
    threads = DecisionThreads(root, store)
    thread = threads.create("Should two Qwen readers run concurrently?", first_id)
    thread_id = thread["thread_id"]
    assert thread["revision"] == 1

    deferred = _decision("defer_initial", "defer", [{"experiment_id": first_id}],
                         revisit="Measure warm and cold latency on matched requests.")
    thread = threads.record(thread_id, 1, deferred)
    receipt = thread["latest_decision"]["basis_receipts"][0]
    assert receipt["brief_sha256"] == first["revisions"][-1]["sha256"]
    assert receipt["result_sha256"] is None
    assert thread["latest_decision"]["choice"]["author_kind"] == "agent_proposed"
    assert threads.record(thread_id, 1, deferred) == thread  # Exact retry.
    with pytest.raises(ValueError, match="different immutable content"):
        threads.record(thread_id, 2, {**deferred, "rationale": "Changed history"})

    second = store.save_brief(planned_brief())
    second_id = second["experiment_id"]
    frozen = store.freeze_plan(second_id, 1)
    result = paired_result(run_id="synthetic_concurrency")
    recorded = store.record_result(second_id, frozen["plan_sha256"], result)
    thread = threads.link(thread_id, 2, second_id)
    with pytest.raises(ValueError, match="Stale"):
        threads.link(thread_id, 2, first_id)
    with pytest.raises(ValueError, match="linked experiment"):
        threads.record(thread_id, 3, _decision("bad_basis", "adopt",
                       [{"experiment_id": store.save_brief({"question": "Other?"})
                         ["experiment_id"]}]))
    with pytest.raises(ValueError, match="No recorded result"):
        threads.record(thread_id, 3, _decision("missing_run", "adopt",
                       [{"experiment_id": second_id, "run_id": "missing"}]))

    accepted = _decision("adopt_after_test", "adopt", [
        {"experiment_id": second_id, "run_id": result["run_id"]}], author="user_reported")
    thread = threads.record(thread_id, 3, accepted)
    assert thread["revision"] == 4
    assert thread["latest_decision"]["basis_receipts"][0]["result_sha256"] == (
        recorded["selected_result"]["sha256"])
    assert [item["choice"]["outcome"] for item in thread["decisions"]] == ["defer", "adopt"]

    restarted = DecisionThreads(root, ExperimentStore(root))
    assert restarted.get(thread_id) == thread
    search = restarted.find("parallel Qwen readers")
    assert search["matches"][0]["thread_id"] == thread_id
    assert search["matches"][0]["latest_decision"] == "adopt"

    index_file = root / f"{thread_id}.json"
    raw = json.loads(index_file.read_text())
    raw["index"]["question"] = "Tampered question"
    index_file.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="missing, unsafe, or corrupt"):
        restarted.get(thread_id)


def test_decision_thread_tools_work_after_mcp_restart(tmp_path):
    library, _, resolver, _ = sources(tmp_path)
    root = tmp_path / "records"
    store = ExperimentStore(root, evidence_resolver=resolver)

    async def run():
        async with Client(create_server(store, library, FakeDiscovery())) as client:
            saved = await client.call_tool("save_experiment_brief", {
                "brief": {"question": "Should we use two Qwen readers?"}})
            experiment_id = saved.structured_content["experiment_id"]
            created = await client.call_tool("create_decision_thread", {
                "question": "Should we use two Qwen readers?", "experiment_id": experiment_id})
            assert not created.is_error
            thread_id = created.structured_content["thread_id"]
            decision = await client.call_tool("record_decision", {
                "thread_id": thread_id, "expected_revision": 1,
                "decision": _decision("defer_speed", "defer", [
                    {"experiment_id": experiment_id}], revisit="Run a matched load test.")})
            assert not decision.is_error
            assert decision.structured_content["latest_decision"]["choice"]["outcome"] == "defer"

        async with Client(create_server(ExperimentStore(root), library,
                                        FakeDiscovery())) as client:
            found = await client.call_tool("find_decision_threads", {
                "question": "Qwen readers"})
            assert found.structured_content["matches"][0]["thread_id"] == thread_id
            loaded = await client.call_tool("get_decision_thread", {"thread_id": thread_id})
            assert loaded.structured_content["revision"] == 2
            assert loaded.structured_content["experiment_ids"] == [experiment_id]
            assert "caller-declared" in loaded.structured_content["warning"]
            invalid = await client.call_tool("record_decision", {
                "thread_id": thread_id, "expected_revision": 2,
                "decision": _decision("bad", "defer", [{"experiment_id": experiment_id}])})
            assert invalid.is_error

    asyncio.run(run())
