"""Exercise automatic export through real MCP requests, without models or network."""

import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

from scripts.ml_experiment_mcp import create_server
from src.product.ml_experiment_store import ExperimentStore
from src.product.obsidian_refresh import ObsidianRefresh
from tests.ml_experiment_fixtures import complete_brief


def test_successful_writes_refresh_but_reads_errors_and_retries_do_not(tmp_path):
    store = ExperimentStore(tmp_path / "records")
    vault = tmp_path / "vault"
    server = create_server(store)
    ObsidianRefresh(vault, [store.files.root]).install(server)

    async def exercise():
        async with Client(server) as client:
            await client.call_tool("list_experiments", {})
            rejected = await client.call_tool("save_experiment_brief", {"brief": {}})
            assert rejected.is_error and not vault.exists()
            saved = await client.call_tool("save_experiment_brief", {"brief": complete_brief()})
            assert saved.meta["obsidian_refresh"]["status"] == "updated"
            key = saved.structured_content["experiment_id"]
            assert (vault / "Experiments" / f"{key}.md").exists()
            frozen = await client.call_tool("freeze_experiment_plan", {
                "experiment_id": key, "expected_revision": 1})
            assert frozen.meta["obsidian_refresh"]["status"] == "updated"
            before = (vault / ".envoy-export.json").read_bytes()
            retry = await client.call_tool("freeze_experiment_plan", {
                "experiment_id": key, "expected_revision": 1})
            assert retry.meta["obsidian_refresh"]["status"] == "unchanged"
            await client.call_tool("get_experiment", {"experiment_id": key})
            assert (vault / ".envoy-export.json").read_bytes() == before

    asyncio.run(exercise())


def test_export_failure_preserves_saved_record_and_warns_client(tmp_path):
    store = ExperimentStore(tmp_path / "records")
    vault = tmp_path / "vault"
    server = create_server(store)
    ObsidianRefresh(vault, [store.files.root]).install(server)

    async def exercise():
        async with Client(server) as client:
            saved = await client.call_tool("save_experiment_brief", {
                "brief": {"question": "Can we improve evidence support?"}})
            key = saved.structured_content["experiment_id"]
            (vault / "Home.md").write_text("Personal edit")
            revised = await client.call_tool("save_experiment_brief", {
                "brief": {"question": "Can we verify evidence support?"},
                "experiment_id": key, "expected_revision": 1})
            assert not revised.is_error
            assert revised.meta["obsidian_refresh"]["status"] == "failed"
            assert "Record saved, but Obsidian refresh failed" in revised.content[-1].text
            assert store.get(key)["revision"] == 2
            assert (vault / "Home.md").read_text() == "Personal edit"

    asyncio.run(exercise())


def test_concurrent_saves_eventually_export_both_records(tmp_path):
    store = ExperimentStore(tmp_path / "records")
    vault = tmp_path / "vault"
    server = create_server(store)
    ObsidianRefresh(vault, [store.files.root]).install(server)

    async def exercise():
        async with Client(server) as client:
            results = await asyncio.gather(*[
                client.call_tool("save_experiment_brief", {"brief": {"question": question}})
                for question in ["Can retrieval help?", "Can verification help?"]])
            for result in results:
                assert not result.is_error
                assert result.meta["obsidian_refresh"]["status"] in {"updated", "unchanged"}
                key = result.structured_content["experiment_id"]
                assert (vault / "Experiments" / f"{key}.md").exists()

    asyncio.run(exercise())


def test_companion_stdio_refreshes_question_notes(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/research_justification_mcp.py"
    vault = tmp_path / "vault"
    params = StdioServerParameters(command=sys.executable, args=[str(script),
        "--store-root", str(tmp_path / "records"), "--library-root", str(tmp_path / "library"),
        "--obsidian-vault", str(vault)], cwd=str(tmp_path))

    async def exercise():
        async with Client(params) as client:
            result = await client.call_tool("start_research_question", {
                "question": "Can a small verifier flag unsupported paper answers?"})
            assert not result.is_error
            assert result.meta["obsidian_refresh"]["status"] == "updated"
            key = result.structured_content["question_id"]
            assert (vault / "Questions" / f"{key}.md").exists()

    asyncio.run(exercise())
