"""A cross-paper decision packet preserves evidence and visible uncertainty."""

import asyncio
import shutil
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from scripts.research_justification_mcp import create_server
from src.product.ml_experiment_evidence import EvidenceResolver
from src.product.ml_experiment_store import ExperimentStore
from tests.arxiv_fakes import make_library
from tests.test_arxiv_research_mcp import FakeDiscovery

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research_justification_mcp.py"


def setup(tmp_path):
    def page(result):
        paragraphs = "".join(f"<p>Method and sample description {i}.</p>" for i in range(4))
        return f"<article><h2>Results</h2>{paragraphs}<p>{result}</p></article>".encode()

    pages = {
        "2609.10001v1": {"html": page("Domain mixing beats the raw mixture.")},
        "2609.10002v1": {"html": page("Quality filtering beats the raw mixture.")},
    }
    library, _ = make_library(tmp_path, pages)
    refs = []
    for index, paper in enumerate(pages, 1):
        library.open(paper)
        text = library.document(paper)["text"]
        quote = ("Domain mixing beats the raw mixture." if index == 1
                 else "Quality filtering beats the raw mixture.")
        start = text.index(quote)
        refs.append({"kind": "paper_span", "reference_id": f"paper{index}",
                     "arxiv_id": paper, "start": start, "end": start + len(quote),
                     "quote": quote})
    store = ExperimentStore(tmp_path / "records",
                            evidence_resolver=EvidenceResolver(library_root=library.root))
    experiment_id = store.save_brief({"question": "Should we mix domains?"})["experiment_id"]
    note = {"note_id": "mixing_comparison", "question": "Should we mix domains?",
            "references": refs, "uncertainties": ["Transfer to Qwen SFT is unknown."],
            "justification": {
                "candidate_method": "Domain mixing", "simpler_baseline": "Quality filtering",
                "compared_findings": [
                    {"reference_id": "paper1", "relation": "supports",
                     "reported_finding": "Mixing improved the raw mixture.",
                     "intervention": "Domain mixing", "comparator": "Raw mixture",
                     "setting": "Synthetic pretraining fixture",
                     "transfer_limit": "Not a Qwen SFT result.",
                     "independence_note": "Overlap with the other paper is unknown."},
                    {"reference_id": "paper2", "relation": "qualifies",
                     "reported_finding": "Filtering also improved the raw mixture.",
                     "intervention": "Quality filtering", "comparator": "Raw mixture",
                     "setting": "Synthetic pretraining fixture",
                     "transfer_limit": "Does not directly compare with mixing.",
                     "independence_note": "Overlap with the other paper is unknown."}],
                "transfer_gaps": ["Different model, data and training objective."],
                "cheapest_matched_test": (
                    "Compare quality-only to quality plus mixing on fixed data."),
                "decision_rule": "Adopt mixing only if support improves at matched compute.",
                "user_reported_context": "A user says data quality mattered more."}}
    return library, store, experiment_id, note


def test_saved_justification_survives_restart_without_source_cache(tmp_path):
    library, store, experiment_id, note = setup(tmp_path)
    saved = store.record_research(experiment_id, 1, note)
    selected = saved["selected_research_note"]
    assert selected["note"]["semantic_support"] == "not_reviewed"

    async def exercise():
        async with Client(create_server(store, library, FakeDiscovery())) as client:
            response = await client.call_tool("get_research_justification", {
                "experiment_id": experiment_id, "note_id": note["note_id"]})
            assert not response.is_error
            packet = response.structured_content
            assert packet["research_note_sha256"] == selected["sha256"]
            assert [entry["relation"] for entry in packet["compared_findings"]] == [
                "supports", "qualifies"]
            assert [entry["quote"] for entry in packet["compared_findings"]] == [
                ref["quote"] for ref in note["references"]]
            assert all(entry["source_sha256"] for entry in packet["compared_findings"])
            assert [(entry["span_start"], entry["span_end"])
                    for entry in packet["compared_findings"]] == [
                        (ref["start"], ref["end"]) for ref in note["references"]]
            assert packet["semantic_review"] == "not_reviewed"
            assert "not a consensus score" in packet["warning"]
        shutil.rmtree(library.papers_dir)
        async with Client(create_server(ExperimentStore(tmp_path / "records"),
                                        library, FakeDiscovery())) as client:
            repeated = await client.call_tool("get_research_justification", {
                "experiment_id": experiment_id, "note_id": note["note_id"]})
            assert repeated.structured_content == packet

    asyncio.run(exercise())


def test_companion_stdio_server_reads_a_saved_packet(tmp_path):
    library, store, experiment_id, note = setup(tmp_path)
    store.record_research(experiment_id, 1, note)
    parameters = StdioServerParameters(command=sys.executable,
        args=[str(SCRIPT), "--store-root", str(tmp_path / "records"),
              "--library-root", str(library.root)], cwd=str(tmp_path))

    async def exercise():
        async with Client(parameters) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            assert {"record_experiment_research", "get_research_justification"} <= tools
            packet = await client.call_tool("get_research_justification", {
                "experiment_id": experiment_id, "note_id": note["note_id"]})
            assert not packet.is_error
            assert packet.structured_content["simpler_baseline"] == "Quality filtering"

    asyncio.run(exercise())


def test_companion_loads_nebius_key_file_without_making_a_request(tmp_path):
    library, _, _, _ = setup(tmp_path)
    env_file = tmp_path / "private.env"
    env_file.write_text("NEBIUS_API_KEY=fixture-key\n")
    parameters = StdioServerParameters(command=sys.executable,
        args=[str(SCRIPT), "--store-root", str(tmp_path / "records"),
              "--library-root", str(library.root), "--enable-nemotron",
              "--nebius-env-file", str(env_file)], cwd=str(tmp_path))

    async def exercise():
        async with Client(parameters) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            assert {"get_research_justification", "draft_research_question"} <= tools

    asyncio.run(exercise())


def test_companion_exposes_prior_runs_and_their_saved_source(tmp_path):
    library, store, _, _ = setup(tmp_path)

    async def exercise():
        async with Client(create_server(store, library, FakeDiscovery())) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert {"find_prior_experiments", "get_prior_experiment",
                    "read_prior_experiment_source"} <= names
            found = await client.call_tool("find_prior_experiments", {"query": "hybrid"})
            assert not found.is_error
            ids = {row["atlas_id"] for row in found.structured_content["matches"]}
            assert "hybrid-v5-live" in ids
            selected = await client.call_tool("get_prior_experiment", {
                "atlas_id": "hybrid-v5-live"})
            assert not selected.is_error
            assert selected.structured_content["sources"][0]["path"].startswith("examples/")
            source = await client.call_tool("read_prior_experiment_source", {
                "atlas_id": "hybrid-v5-live", "source_index": 0, "length": 500})
            assert not source.is_error
            assert "V5 with matched hybrid-selected passages" in source.structured_content["text"]
            rejected = await client.call_tool("read_prior_experiment_source", {
                "atlas_id": "../escape", "source_index": 0})
            assert rejected.is_error

    asyncio.run(exercise())


@pytest.mark.parametrize("change", ["same_paper", "external", "all_support", "fake_review",
                                    "unknown_reference", "missing_baseline"])
def test_unsupported_comparison_is_rejected_without_write(tmp_path, change):
    _, store, experiment_id, note = setup(tmp_path)
    if change == "same_paper":
        note["references"][1]["arxiv_id"] = note["references"][0]["arxiv_id"]
    elif change == "external":
        note["references"][1] = {"kind": "external_reference", "reference_id": "paper2",
                                 "url": "https://example.org/paper", "description": "Unverified"}
    elif change == "all_support":
        note["justification"]["compared_findings"][1]["relation"] = "supports"
    elif change == "fake_review":
        note["justification"]["semantic_review"] = "human_verified"
    elif change == "unknown_reference":
        note["justification"]["compared_findings"][1]["reference_id"] = "missing"
    else:
        note["justification"]["simpler_baseline"] = ""
    with pytest.raises(ValueError):
        store.record_research(experiment_id, 1, note)
    assert store.get(experiment_id)["research_notes"] == []
