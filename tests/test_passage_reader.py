"""Scoped source access, uninspected citations, durable attempts and MCP attachment."""
import asyncio
import copy
import hashlib
import os

import pytest
from mcp import Client

from scripts.experiment_mode_mcp import create_server
from src.product.assigned_actions import displayed_tool_call
from src.product.assigned_reader import build_assigned_reader
from src.product.ml_experiment_evidence import EvidenceResolver
from src.product.ml_experiment_store import ExperimentStore
from src.product.packet_verifier import verify_investigation_packet
from src.product.passage_reader_service import PassageReaderService
from src.product.passage_scope import PASSAGES, checked_read, make_scope
from src.product.qwen_investigator import QwenInvestigator
from src.research.agent import load_snapshot
from tests.arxiv_fakes import article_page, make_library
from tests.test_arxiv_research_mcp import FakeDiscovery
from tests.test_qwen_investigator import FakePolicy, FakeREPL

ID = "arxiv_2609_12345v2"


class ScopedREPL(FakeREPL):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def execute(self, code, **kwargs):
        self.executed.append(code)
        scope = {"doc_id": ID, "windows": [{"start": 0, "end": len(self.text)}]}
        w = checked_read(displayed_tool_call(code), scope)
        return repr({"text": self.text[w["start"]:w["end"]], "start": w["start"]})


def fixture(tmp_path, actions=None):
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    library.open("2609.12345v2")
    text = library.document("2609.12345v2")["text"]
    quote = "Eviction saves 40 percent of memory"
    start = text.index(quote)
    windows = [{"start": start, "end": start + len(quote)}]
    read = f'print(passage("{ID}", {start}, {len(quote)}))'
    submit = (f'submit(answer="Eviction saves 40 percent of memory.", citations=["{ID}"], '
              f'evidence=[{{"doc_id":"{ID}","start":{start},"end":{start + len(quote)}}}])')
    repl = ScopedREPL(text)
    policy = FakePolicy(actions or [read, submit])
    reader = QwenInvestigator(lambda: policy, {"checkpoint": "scripted"},
        source_domain="public_papers", harness_version="v2", reader_mode="assigned",
        reader_guidance=PASSAGES, max_steps=4, repl_factory=lambda _: repl)
    store = ExperimentStore(tmp_path / "records", evidence_resolver=EvidenceResolver(
        library_root=library.root))
    key = store.save_brief({"question": "Memory savings?"})["experiment_id"]
    service = PassageReaderService(store, library, lambda: (reader, None), tmp_path / "reports",
                                   data_kind="synthetic")
    return library, store, key, reader, service, windows, read, submit, repl


def test_mcp_reader_attaches_actual_source_receipts_and_resumes_without_rereading(tmp_path):
    library, store, key, reader, service, windows, _, _, repl = fixture(tmp_path)
    args = dict(experiment_id=key, expected_revision=1, note_id="read_one",
                question="Memory savings?", arxiv_id="2609.12345v2", windows=windows)

    async def run():
        server = create_server(store, library, FakeDiscovery(), passage_reader=service)
        async with Client(server) as client:
            tool = next(t for t in (await client.list_tools()).tools
                        if t.name == "read_experiment_passages")
            assert tool.annotations.open_world_hint and not tool.annotations.read_only_hint
            returned = await client.call_tool("read_experiment_passages", args)
            assert not returned.is_error
            result = returned.structured_content
            assert result["status"] == "evidence_found" and result["note_attached"]
            assert result["semantic_support"] == "not_reviewed"
            resume = await client.call_tool("resume_experiment", {"experiment_id": key})
            assert resume.structured_content["latest_research_note_id"] == "read_one"
            invalid = await client.call_tool("read_experiment_passages", {
                **args, "windows": [{"start": True, "end": 10}]})
            assert invalid.is_error
        restarted = PassageReaderService(store, library,
            lambda: (reader, None), tmp_path / "reports", data_kind="synthetic")
        assert restarted.read(**args) == result
        assert len(repl.executed) == 1
        note = store.get(key)["selected_research_note"]
        assert note["note"]["origin"] == "qwen_research"
        assert note["evidence_receipts"][0]["verification"] == "paper_span_verified"
        with pytest.raises(ValueError, match="different passage request"):
            restarted.read(**{**args, "question": "Changed"})
    asyncio.run(run())


@pytest.mark.parametrize("action", [f'print(read("{ID}"))',
    f'print(passage("{ID}", 0, 1))', 'import os\nos.system("echo unsafe")'])
def test_outside_or_arbitrary_actions_are_rejected_before_docker_execution(tmp_path, action):
    library, _, _, reader, _, windows, read, submit, repl = fixture(tmp_path)
    policy = FakePolicy([action, read, submit])
    reader.policy_factory = lambda: policy
    packet = reader.investigate("Memory?", library.view(["2609.12345v2"]),
                                passage_windows=windows)
    assert packet["status"] == "evidence_found" and repl.executed == [read]
    assert "Action rejected" in packet["trajectory"][0]["observation"]


def test_uninspected_evidence_rejected_even_inside_an_assigned_window(tmp_path):
    library, _, _, reader, _, windows, read, submit, _ = fixture(tmp_path)
    start = windows[0]["start"]
    reader.policy_factory = lambda: FakePolicy([
        f'print(passage("{ID}", {start}, 5))', submit, read, submit])
    snapshot = library.view(["2609.12345v2"])
    packet = reader.investigate("Memory?", snapshot, passage_windows=windows)
    assert "actually inspected" in packet["trajectory"][1]["observation"]
    manifest, docs = load_snapshot(snapshot)
    kwargs = dict(allowed_prompt_sha256s=frozenset({hashlib.sha256(
        reader.system_prompt.encode()).hexdigest()}), expected_reader_mode="assigned",
        expected_reader_guidance=PASSAGES, expected_passage_windows=windows)
    assert verify_investigation_packet(packet, manifest, docs, **kwargs)["evidence"]
    bad = copy.deepcopy(packet)
    bad["inspected_passage_windows"] = []
    with pytest.raises(ValueError, match="uninspected"):
        verify_investigation_packet(bad, manifest, docs, **kwargs)


def test_stale_revision_retains_checked_result_and_does_not_repeat_inference(tmp_path):
    library, store, key, reader, service, windows, read, submit, repl = fixture(tmp_path)
    original = reader.investigate
    def revise(*args, **kwargs):
        packet = original(*args, **kwargs)
        store.save_brief({"question": "Changed setup"}, key, 1)
        return packet
    reader.investigate = revise
    args = dict(experiment_id=key, expected_revision=1, note_id="raced",
                question="Memory?", arxiv_id="2609.12345v2", windows=windows)
    result = service.read(**args)
    assert result["evidence"] and not result["note_attached"] and result["attachment_error"]
    assert service.read(**args) == result and repl.executed == [read]
    assert store.get(key)["research_notes"] == []


def test_invalid_ranges_fail_before_policy_or_request_reservation(tmp_path):
    library, store, key, reader, service, windows, _, _, repl = fixture(tmp_path)
    with pytest.raises(ValueError, match="ordered"):
        service.read(key, 1, "bad_range", "Memory?", "2609.12345v2",
                     [{"start": 0, "end": 1000000}])
    assert not repl.started
    with pytest.raises(ValueError):
        make_scope({ID: library.document("2609.12345v2")}, [{"start": True, "end": 10}])


def test_interrupted_attempt_cannot_dispatch_again(tmp_path):
    _, _, key, reader, service, windows, _, _, _ = fixture(tmp_path)
    count = 0
    def interrupt(*args, **kwargs):
        nonlocal count
        count += 1
        raise SystemExit("simulated process interruption")
    reader.investigate = interrupt
    args = (key, 1, "interrupted", "Memory?", "2609.12345v2", windows)
    with pytest.raises(SystemExit):
        service.read(*args)
    result = service.read(*args)
    assert result["status"] == "previous_attempt_incomplete" and count == 1


@pytest.mark.skipif(not os.environ.get("ENVOY_ASSIGNED_TEST_DOCKER_IMAGE"),
                    reason="set ENVOY_ASSIGNED_TEST_DOCKER_IMAGE")
def test_real_docker_read_and_submission_use_the_same_effective_prompt(tmp_path):
    library, _, _, _, _, windows, read, submit, _ = fixture(tmp_path)
    actions, sent = iter([read, submit]), []
    def transport(url, payload, headers, timeout):
        sent.append(payload)
        return {"choices": [{"message": {"content": next(actions)}}]}
    reader = build_assigned_reader("qwen", qwen_endpoint="http://scripted",
        qwen_model="fake", qwen_checkpoint="fixture", reader_guidance=PASSAGES,
        max_steps=4, sandbox_image=os.environ["ENVOY_ASSIGNED_TEST_DOCKER_IMAGE"],
        policy_transport=transport)
    packet = reader.investigate("Memory?", library.view(["2609.12345v2"]),
                                passage_windows=windows)
    assert packet["execution"] == "docker_only" and packet["status"] == "evidence_found"
    assert sent[0]["messages"][0]["content"] == reader.system_prompt
    expected = hashlib.sha256(reader.system_prompt.encode()).hexdigest()
    assert packet["system_prompt_sha256"] == expected
