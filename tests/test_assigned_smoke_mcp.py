"""Frozen MCP execution path: host-visible output, budgets and retained raw evidence."""
# ruff: noqa: F811
import json
import os

import pytest
from mcp import Client

from src.eval.assigned_smoke_mcp import FrozenPaperLibrary, NoDiscovery, investigate_through_mcp
from src.eval.assigned_smoke_plan import prepare
from src.eval.assigned_smoke_run import run_cases
from src.policies.runpod_queue import RunpodQueueTransport
from tests.test_assigned_guidance_recovery import candidate
from tests.test_assigned_reader import INSPECT, SUBMIT
from tests.test_assigned_smoke_plan import Tokenizer, cached  # noqa: F401
from tests.test_reader_gaps import GAP


def test_frozen_mcp_adapter_cannot_discover_fetch_or_select_other_papers(tmp_path, monkeypatch):
    _, _, _, snapshot = candidate(tmp_path, [])
    def forbidden(*args, **kwargs):
        raise AssertionError("no external source fetching allowed")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    library = FrozenPaperLibrary(snapshot)
    version = library.assignment["arxiv_id"]
    assert library.open(version)["arxiv_id"] == version
    assert library.view([version]) == snapshot
    with pytest.raises(ValueError, match="frozen paper"):
        library.open("2609.12345")
    with pytest.raises(ValueError, match="frozen one-paper"):
        library.view([version, "2609.54321v1"])
    with pytest.raises(ValueError, match="outside"):
        NoDiscovery().search("anything")


@pytest.mark.parametrize("field", ["evidence", "research_gaps", "candidate_answers",
                                   "answer_changed", "model_identity", "paper_assignment"])
def test_corrupt_host_visible_result_fails_but_preserves_raw_evidence(tmp_path, monkeypatch, field):
    reader, _, _, snapshot = candidate(tmp_path, [INSPECT, GAP, SUBMIT], steps=3)
    call = Client.call_tool
    async def corrupt(self, *args, **kwargs):
        reply = await call(self, *args, **kwargs)
        if field == "answer_changed":
            reply.structured_content["candidate_answers"][0]["text"] = "Changed answer"
        elif field in {"model_identity", "paper_assignment"}:
            reply.structured_content["papers"][0][field] = {}
        else:
            reply.structured_content[field] = []
        return reply
    monkeypatch.setattr(Client, "call_tool", corrupt)
    prefix = tmp_path / "case"
    with pytest.raises(ValueError, match="MCP result"):
        investigate_through_mcp(reader, "Memory?", snapshot, prefix)
    raw = json.loads(prefix.with_suffix(".packet.json").read_text())
    assert raw["evidence"] and raw["research_gaps"]
    assert prefix.with_suffix(".mcp.json").exists()


IMAGE = os.environ.get("ENVOY_ASSIGNED_TEST_DOCKER_IMAGE")


@pytest.mark.skipif(not IMAGE, reason="set ENVOY_ASSIGNED_TEST_DOCKER_IMAGE")
def test_candidate_executor_uses_real_mcp_docker_and_six_bounded_scripted_requests(cached):
    library, _, prepared = cached
    plan = prepare(library.root, prepared, sandbox_image=IMAGE, tokenizer=Tokenizer(),
                   reader_guidance="focused-v1")
    doc = library.document(plan["paper_version"])
    did, start = doc["doc_id"], doc["text"].index("We measure walltime")
    actions = iter([f'print(search_within("{did}", "batch size"))',
        f'submit(answer="1", citations=["{did}"], evidence=[{{"doc_id":"{did}",'
        f'"start":{start},"end":{start+80}}}])',
        f'print(passage("{did}", 0, 200))',
        f'print(search_within("{did}", "batch size"))',
        'report_gap(missing="Target setup not established.", follow_up_query="target setup")',
        'submit(answer="Unanswerable", citations=[], evidence=[])'])
    posts = []
    def factory(endpoint, key, tokenizer, audit, **kwargs):
        def http(method, path, payload=None):
            assert method == "POST" and path == "/run"
            posts.append(payload)
            return {"id": f"job-{len(posts)}", "status": "COMPLETED", "output": {
                "model": "envoy", "choices": [{"message": {"content": next(actions)}}]}}
        return RunpodQueueTransport(endpoint, key, tokenizer, audit, http=http, **kwargs)
    output = prepared.parent / "run"
    output.mkdir()
    summary = run_cases(prepared, output, tokenizer=Tokenizer(), key="fake",
                        transport_factory=factory)
    assert len(posts) == summary["model_requests_reserved"] == 6
    assert [c["status"] for c in summary["cases"]] == ["evidence_found", "no_evidence"]
    assert all(c["mcp_verification"] == c["packet_verification"] == "passed"
               for c in summary["cases"])
    packet = json.loads((output / "assigned-batch-size-v1.packet.json").read_text())
    assert packet["execution"] == "docker_only" and packet["reader_guidance"] == "focused-v1"
    reply = json.loads((output / "assigned-a5000-batch8-v1.mcp.json").read_text())
    assert reply["result"]["research_gaps"][0]["follow_up_query"] == "target setup"
