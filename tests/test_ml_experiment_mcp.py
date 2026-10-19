import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp import Client, StdioServerParameters  # noqa: E402

from scripts.ml_experiment_mcp import create_server  # noqa: E402
from src.product.ml_experiment_store import ExperimentStore  # noqa: E402
from tests.ml_experiment_fixtures import complete_brief  # noqa: E402
from tests.ml_experiment_result_fixtures import paired_result, planned_brief  # noqa: E402
from tests.test_ml_experiment_evidence import sources  # noqa: E402
from tests.test_ml_experiment_outcomes import attach_measurements  # noqa: E402

TOOLS = {"save_experiment_brief", "get_experiment", "list_experiments", "freeze_experiment_plan",
         "record_experiment_result"}


def test_question_first_tools_annotations_errors_and_freeze(tmp_path):
    store = ExperimentStore(tmp_path)

    async def body():
        async with Client(create_server(store)) as client:
            tools = (await client.list_tools()).tools
            assert {tool.name for tool in tools} == TOOLS
            for tool in tools:
                assert tool.annotations.open_world_hint is False
                assert tool.annotations.read_only_hint == (tool.name in {
                    "get_experiment", "list_experiments"})
            assert "Start with the user's" in client.instructions
            assert "not semantic support" in client.instructions
            saved = await client.call_tool("save_experiment_brief", {
                "brief": {"question": "Would a smaller model help?"}})
            assert not saved.is_error
            row = saved.structured_content
            assert row["status"] == "draft" and "hypothesis" in row["missing_fields"]
            assert row["brief"]["current_setup"]["hardware"]["value"] is None
            key = row["experiment_id"]
            early = await client.call_tool("freeze_experiment_plan", {
                "experiment_id": key, "expected_revision": 1})
            assert early.is_error
            revised = await client.call_tool("save_experiment_brief", {
                "brief": complete_brief(), "experiment_id": key, "expected_revision": 1})
            assert not revised.is_error and revised.structured_content["revision"] == 2
            frozen = await client.call_tool("freeze_experiment_plan", {
                "experiment_id": key, "expected_revision": 2})
            assert not frozen.is_error and frozen.structured_content["plan_sha256"]
            assert "Frozen plan SHA-256" in frozen.structured_content["markdown"]
            assert "brief" not in frozen.structured_content["frozen_plan"]
            edited = await client.call_tool("save_experiment_brief", {
                "brief": complete_brief(), "experiment_id": key, "expected_revision": 2})
            assert edited.is_error
            unknown = await client.call_tool("record_experiment_result", {})
            assert unknown.is_error
            listing = (await client.call_tool("list_experiments", {})).structured_content
            assert listing["experiments"][0]["experiment_id"] == key

    asyncio.run(body())


@pytest.mark.parametrize("name,args", [
    ("list_experiments", {"limit": True}),
    ("get_experiment", {"experiment_id": "../../.env"}),
    ("save_experiment_brief", {"brief": {"question": "Question", "evidence_receipts": []}}),
    ("save_experiment_brief", {"brief": {"question": "Question", "human_review": True}}),
    ("save_experiment_brief", {"brief": {"question": "Question"}, "expected_revision": True}),
    ("save_experiment_brief", {"brief": {"question": "Question", "constraints": ["x" * 4_000] * 20,
                                        "unknowns": ["y" * 4_000] * 30,
                                        "differences": ["z" * 4_000] * 20}}),
    ("save_experiment_brief", {"brief": {"question": "Question", "references": [{
        "kind": "external_reference", "reference_id": "docs", "url": "https://example.org",
        "description": "Server has no configured source resolver"}]}}),
])
def test_bad_mcp_arguments_do_not_write_records(tmp_path, name, args):
    store = ExperimentStore(tmp_path)

    async def body():
        async with Client(create_server(store)) as client:
            assert (await client.call_tool(name, args)).is_error
            assert store.list()["experiments"] == []

    asyncio.run(body())


def test_long_markdown_is_explicitly_truncated_without_losing_saved_fields(tmp_path):
    store = ExperimentStore(tmp_path)

    async def body():
        async with Client(create_server(store)) as client:
            result = await client.call_tool("save_experiment_brief", {"brief": {
                "question": "Compare methods", "constraints": ["x" * 4_000] * 20}})
            assert not result.is_error
            row = result.structured_content
            assert row["markdown_truncated"] and len(row["markdown"]) < 20_100
            assert len(row["brief"]["constraints"]) == 20
            assert len(store.get(row["experiment_id"])["brief"]["constraints"][0]) == 4_000

    asyncio.run(body())


def test_source_receipts_are_visible_and_rendered_as_untrusted_text(tmp_path):
    _, exports, resolver, refs = sources(tmp_path)
    store = ExperimentStore(tmp_path / "records", evidence_resolver=resolver)

    async def body():
        async with Client(create_server(store)) as client:
            result = await client.call_tool("save_experiment_brief", {"brief": {
                "question": "Compare <script>alert(1)</script> **methods**", "references": refs}})
            assert not result.is_error
            row = result.structured_content
            assert "<script>" not in row["markdown"]
            assert "&lt;script&gt;" in row["markdown"]
            assert "artifact_value_verified" in row["markdown"]
            assert "semantic support remains unreviewed" in row["markdown"]
            null_value = await client.call_tool("save_experiment_brief", {"brief": {
                "question": "What was the recorded cost?", "references": [
                    {**refs[1], "json_pointer": "/unknown_cost"}]}})
            assert not null_value.is_error
            assert "> null" in null_value.structured_content["markdown"]
            (exports / "run.json").unlink()
            fetched = (await client.call_tool("get_experiment", {
                "experiment_id": row["experiment_id"]})).structured_content
            assert fetched["evidence_receipts"] == row["evidence_receipts"]

    asyncio.run(body())


def test_real_stdio_client_can_save_and_reopen_from_other_working_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/ml_experiment_mcp.py"
    root = tmp_path / "records"
    library, exports, _, refs = sources(tmp_path)

    async def body():
        parameters = StdioServerParameters(command=sys.executable,
            args=[str(script), "--store-root", str(root),
                  "--library-root", str(library.root), "--artifact-root", str(exports)],
            cwd=str(tmp_path))
        async with Client(parameters) as client:
            assert {t.name for t in (await client.list_tools()).tools} == TOOLS
            result = await client.call_tool("save_experiment_brief", {
                "brief": {"question": "Which latency method should I investigate?"}})
            assert not result.is_error
            key = result.structured_content["experiment_id"]
            revised = await client.call_tool("save_experiment_brief", {
                "brief": {**planned_brief(), "references": refs},
                "experiment_id": key, "expected_revision": 1})
            assert not revised.is_error
            frozen = await client.call_tool("freeze_experiment_plan", {
                "experiment_id": key, "expected_revision": 2})
            assert not frozen.is_error
            plan_hash = frozen.structured_content["plan_sha256"]
            outcome = paired_result()
            measurement_path = attach_measurements(exports, outcome)
            args = {"experiment_id": key, "plan_sha256": plan_hash, "result": outcome}
            recorded = await client.call_tool("record_experiment_result", args)
            assert not recorded.is_error
            expected = recorded.structured_content["selected_result"]
            assert expected["assessment"]["overall"] == "criteria_met"
            assert "Synthetic data" in recorded.structured_content["markdown"]
            assert "25.0 percent" in recorded.structured_content["markdown"]
        (exports / "run.json").unlink()
        measurement_path.unlink()
        async with Client(parameters) as restarted:
            saved = await restarted.call_tool("get_experiment", {
                "experiment_id": key, "run_id": "synthetic_run"})
            assert not saved.is_error
            assert saved.structured_content["plan_sha256"] == plan_hash
            assert saved.structured_content["evidence_receipts"][1]["value"]["passed"] == 3
            assert saved.structured_content["selected_result"] == expected
            retry = await restarted.call_tool("record_experiment_result", args)
            assert not retry.is_error and retry.structured_content["selected_result"] == expected
            args["result"]["candidate"]["measurements"][0]["value"] = 1.0
            assert (await restarted.call_tool("record_experiment_result", args)).is_error

    asyncio.run(body())
