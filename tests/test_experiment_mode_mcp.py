"""Real MCP flows with fixture sources; no model or paid infrastructure calls."""

import asyncio
import shutil
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

from scripts.experiment_mode_mcp import create_server
from src.product.experiment_context import MAX_CONTEXT_CHARS
from src.product.experiment_discovery import ExperimentDiscovery
from src.product.ml_experiment_store import ExperimentStore
from tests.ml_experiment_result_fixtures import paired_result, planned_brief
from tests.test_arxiv_research_mcp import FakeDiscovery
from tests.test_ml_experiment_evidence import sources
from tests.test_ml_experiment_research_notes import note

TOOLS = {"save_experiment_brief", "get_experiment", "list_experiments",
         "find_related_experiments",
         "freeze_experiment_plan", "record_experiment_result", "record_experiment_research",
         "resume_experiment", "search_papers", "get_citations", "open_paper",
         "find_evidence", "get_passage"}


def test_real_mcp_question_research_revision_result_restart_and_resume(tmp_path):
    library, exports, resolver, refs = sources(tmp_path)
    store = ExperimentStore(tmp_path / "records", evidence_resolver=resolver)

    async def run():
        server = create_server(store, library, ExperimentDiscovery(FakeDiscovery()))
        async with Client(server) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert set(tools) == TOOLS
            assert "resume_experiment" in client.instructions
            assert "unreviewed" in client.instructions
            for name, tool in tools.items():
                assert tool.annotations.open_world_hint == (name in {
                    "search_papers", "get_citations", "open_paper"})
                assert tool.annotations.read_only_hint == (name in {
                    "resume_experiment", "get_experiment", "list_experiments",
                    "find_related_experiments", "search_papers",
                    "get_citations", "open_paper", "find_evidence", "get_passage"})
            saved = await client.call_tool("save_experiment_brief", {
                "brief": {"question": "Reduce inference memory?"}})
            key = saved.structured_content["experiment_id"]
            discovered = await client.call_tool("search_papers", {"query": "memory"})
            assert discovered.structured_content["status"] == "ok"
            opened = await client.call_tool("open_paper", {"arxiv_id": "2609.12345v2"})
            assert opened.structured_content["status"] == "ok"
            found = await client.call_tool("find_evidence", {
                "query": "memory", "arxiv_ids": ["2609.12345v2"]})
            assert not found.is_error
            inspected = await client.call_tool("get_passage", {
                "arxiv_id": "2609.12345v2", "start": 0, "length": 40})
            assert inspected.structured_content["text"] == (
                library.document("2609.12345v2")["text"][:40])
            paper_ref = {**refs[0], "quote": inspected.structured_content["text"]}
            first = await client.call_tool("record_experiment_research", {
                "experiment_id": key, "expected_revision": 1, "note": note([paper_ref])})
            assert not first.is_error
            research = first.structured_content["selected_research_note"]
            assert research["evidence_receipts"][0]["value"] == inspected.structured_content["text"]
            assert "Linked research" in first.structured_content["markdown"]
            assert "human" not in research
            revised = await client.call_tool("save_experiment_brief", {
                "experiment_id": key, "expected_revision": 1, "brief": planned_brief()})
            assert not revised.is_error
            frozen = await client.call_tool("freeze_experiment_plan", {
                "experiment_id": key, "expected_revision": 2})
            plan_hash = frozen.structured_content["plan_sha256"]
            result = paired_result()
            recorded = await client.call_tool("record_experiment_result", {
                "experiment_id": key, "plan_sha256": plan_hash, "result": result})
            assert not recorded.is_error
            record = recorded.structured_content["selected_result"]

        # Resume after both client and store restart, even with the sources gone.
        (exports / "run.json").unlink()
        shutil.rmtree(library.papers_dir)
        restarted = ExperimentStore(tmp_path / "records")
        async with Client(create_server(restarted, library, FakeDiscovery())) as client:
            resumed = await client.call_tool("resume_experiment", {"experiment_id": key})
            assert not resumed.is_error
            context = resumed.structured_content
            assert context["revision"] == 2 and context["plan_sha256"] == plan_hash
            assert context["latest_result_run_id"] == result["run_id"]
            assert context["latest_research_note_id"] == "memory_search"
            assert context["research_note_count"] == context["result_count"] == 1
            assert '"matches_current_brief_revision": false' in context["context"]
            assert '"data_kind": "synthetic"' in context["context"]
            assert "Needs a local measurement" in context["context"]
            assert "semantic support" in context["warning"]
            full = await client.call_tool("get_experiment", {"experiment_id": key,
                "research_note_id": "memory_search", "run_id": result["run_id"]})
            assert full.structured_content["selected_result"] == record
            assert full.structured_content["selected_research_note"] == research
        return key

    asyncio.run(run())


def test_bad_notes_and_source_errors_return_actionable_tool_errors(tmp_path):
    library, _, resolver, refs = sources(tmp_path)
    store = ExperimentStore(tmp_path / "records", evidence_resolver=resolver)
    saved = store.save_brief({"question": "Memory?"})
    key = saved["experiment_id"]

    async def run():
        async with Client(create_server(store, library, FakeDiscovery())) as client:
            for args in [
                {"experiment_id": "../escape"}, {"experiment_id": key, "expected_revision": True},
                {"experiment_id": key, "expected_revision": 2},
            ]:
                rejected = await client.call_tool("record_experiment_research", {
                    "note": note(refs), **args})
                assert rejected.is_error
            invalid = await client.call_tool("get_passage", {
                "arxiv_id": "2609.12345v2", "length": 8001})
            assert invalid.is_error
            missing = await client.call_tool("resume_experiment", {"experiment_id": "../escape"})
            assert missing.is_error
            # Provider failure remains structured data, as in the existing source server.
        async with Client(create_server(store, library, FakeDiscovery(fail=True))) as client:
            unavailable = await client.call_tool("search_papers", {"query": "memory"})
            assert not unavailable.is_error
            assert unavailable.structured_content["status"] == "source_unavailable"
    asyncio.run(run())
    assert store.get(key) == saved


def test_resume_context_is_bounded_and_full_data_remains_accessible(tmp_path):
    library, _, resolver, _ = sources(tmp_path)
    store = ExperimentStore(tmp_path / "records", evidence_resolver=resolver)
    saved = store.save_brief({"question": "Memory?", "constraints": ["x" * 4000] * 20,
                              "unknowns": ["y" * 4000] * 30})
    key = saved["experiment_id"]
    store.record_research(key, 1, note())
    async def run():
        async with Client(create_server(store, library, FakeDiscovery())) as client:
            resumed = (await client.call_tool("resume_experiment", {
                "experiment_id": key})).structured_content
            assert resumed["context_truncated"]
            assert len(resumed["context"]) <= MAX_CONTEXT_CHARS + 100
            assert "truncated" in resumed["context"]
            assert "Unknowns" in resumed["truncated_sections"]
            assert "Needs a local measurement" in resumed["context"]
            full = (await client.call_tool("get_experiment", {
                "experiment_id": key})).structured_content
            assert full["brief"]["constraints"] == ["x" * 4000] * 20
    asyncio.run(run())


def test_real_stdio_mode_starts_without_docker_tokenizer_or_endpoint(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/experiment_mode_mcp.py"
    library, _, _, refs = sources(tmp_path)
    root = tmp_path / "records"
    parameters = StdioServerParameters(command=sys.executable,
        args=[str(script), "--store-root", str(root), "--library-root", str(library.root)],
        cwd=str(tmp_path))

    async def run():
        async with Client(parameters) as client:
            assert {t.name for t in (await client.list_tools()).tools} == TOOLS
            saved = await client.call_tool("save_experiment_brief", {
                "brief": {"question": "Can we reduce memory?"}})
            key = saved.structured_content["experiment_id"]
            result = await client.call_tool("record_experiment_research", {
                "experiment_id": key, "expected_revision": 1, "note": note(refs[:1])})
            assert not result.is_error
        async with Client(parameters) as client:
            resumed = await client.call_tool("resume_experiment", {"experiment_id": key})
            assert not resumed.is_error
            assert resumed.structured_content["research_note_count"] == 1
            assert resumed.structured_content["latest_result_run_id"] is None
    asyncio.run(run())
