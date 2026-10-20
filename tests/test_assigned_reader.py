"""Assigned IDs are metadata, not inspection; the default discovery gate stays on."""
import copy
import hashlib

import pytest

from src.product.assigned_reader import build_assigned_reader
from src.product.packet_verifier import verify_investigation_packet
from src.product.paper_assignment import paper_assignment
from src.product.qwen_investigator import QwenInvestigator, effective_system_prompt
from src.research.agent import load_snapshot
from tests.arxiv_fakes import article_page, make_library
from tests.test_qwen_investigator import FakePolicy, FakeREPL


def setup_reader(tmp_path, actions, *, mode="assigned", repl=None, steps=4):
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    library.open("2609.12345v2")
    snapshot = library.view(["2609.12345v2"])
    policy = FakePolicy(actions)
    repl = repl or FakeREPL()
    reader = QwenInvestigator(lambda: policy, {"checkpoint": "fake"},
                              source_domain="public_papers", harness_version="v2",
                              reader_mode=mode, max_steps=steps, repl_factory=lambda _: repl)
    return reader, policy, repl, snapshot


DOC_ID = "arxiv_2609_12345v2"
INSPECT = f'print(search_within("{DOC_ID}", "memory"))'
SUBMIT = (f'submit(answer="A proposal.", citations=["{DOC_ID}"], '
          f'evidence=[{{"doc_id":"{DOC_ID}","start":0,"end":10}}])')


def verify(reader, packet, snapshot):
    manifest, docs = load_snapshot(snapshot)
    return verify_investigation_packet(packet, manifest, docs,
        allowed_prompt_sha256s=frozenset({hashlib.sha256(reader.system_prompt.encode()).hexdigest()}))


def test_assigned_id_is_supplied_and_first_action_can_inspect(tmp_path):
    reader, policy, repl, snapshot = setup_reader(tmp_path, [INSPECT, SUBMIT], steps=2)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "evidence_found"
    assert DOC_ID in policy.observations[0]
    assert "first action must call search()" not in reader.system_prompt
    assert "metadata, not inspected evidence" in reader.system_prompt
    assert packet["inspection_steps"] == packet["document_tool_steps"] == 1
    assert [action for action, _ in repl.executed] == [INSPECT] and repl.killed
    compact = verify(reader, packet, snapshot)
    assert compact["paper_assignment"]["doc_id"] == DOC_ID
    assert compact["paper_assignment"]["counts_as_source_inspection"] is False


def test_assignment_does_not_allow_immediate_citation_or_abstention(tmp_path):
    reader, _, repl, snapshot = setup_reader(tmp_path, [SUBMIT,
        'submit(answer="Unanswerable", citations=[], evidence=[])'], steps=2)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "incomplete" and packet["evidence"] == []
    assert packet["inspection_steps"] == packet["document_tool_steps"] == 0
    assert "inspect source text" in packet["trajectory"][0]["observation"]
    assert "two distinct successful" in packet["trajectory"][1]["observation"]
    assert repl.executed == []


def test_default_gate_rejects_paper_scoped_search_before_discovery(tmp_path):
    reader, _, repl, snapshot = setup_reader(tmp_path, [INSPECT, SUBMIT],
                                             mode="discovery", steps=2)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "incomplete" and repl.executed == []
    assert "first call search" in packet["trajectory"][0]["observation"]
    assert "paper_assignment" not in packet


def test_missing_id_result_is_not_source_inspection(tmp_path):
    class Missing(FakeREPL):
        def execute(self, code, **kwargs):
            return "ERROR: Document 'invented' not found"
    reader, _, _, snapshot = setup_reader(tmp_path,
        ['print(read("invented"))', SUBMIT], repl=Missing(), steps=2)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "incomplete" and packet["inspection_steps"] == 0


@pytest.mark.parametrize("field", ["paper_assignment", "assignment_sha256"])
def test_verifier_rejects_altered_assignment(tmp_path, field):
    reader, _, _, snapshot = setup_reader(tmp_path, [INSPECT, SUBMIT], steps=2)
    packet = reader.investigate("Memory?", snapshot)
    bad = copy.deepcopy(packet)
    bad[field] = {} if field == "paper_assignment" else "bad"
    with pytest.raises(ValueError, match="assignment"):
        verify(reader, bad, snapshot)


def test_assignment_needs_one_pinned_real_paper(tmp_path):
    reader, _, _, snapshot = setup_reader(tmp_path, [], steps=2)
    manifest, docs = load_snapshot(snapshot)
    with pytest.raises(ValueError, match="exactly one"):
        paper_assignment(manifest, {**docs, "invented": next(iter(docs.values()))})
    bad = copy.deepcopy(docs)
    bad[DOC_ID]["metadata"]["arxiv_id"] = "2609.12345"
    with pytest.raises(ValueError, match="pinned version"):
        paper_assignment(manifest, bad)
    with pytest.raises(ValueError, match="requires public papers and harness v2"):
        effective_system_prompt("public_papers", "default", "v1", reader_mode="assigned")


@pytest.mark.parametrize("kind", ["qwen", "nemotron"])
def test_policy_investigator_and_packet_share_assigned_prompt(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    replies = iter([INSPECT, SUBMIT])
    sent = []
    def transport(url, payload, headers, timeout):
        sent.append(payload)
        return {"choices": [{"message": {"content": next(replies)}}]}
    reader = build_assigned_reader(kind, qwen_endpoint="http://test", qwen_model="fake",
        qwen_checkpoint="fake-pinned", policy_transport=transport, max_steps=2)
    _, _, _, snapshot = setup_reader(tmp_path, [], steps=2)
    reader.repl_factory = lambda _: FakeREPL()
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "evidence_found"
    assert reader.system_prompt == reader.policy_factory().system_prompt
    assert sent[0]["messages"][0]["content"] == reader.system_prompt
    assert packet["system_prompt_sha256"] == verify(reader, packet, snapshot)["prompt_hash"]
    assert "Never call submit()" not in reader.system_prompt


def test_assigned_reader_still_requires_docker(tmp_path, monkeypatch):
    reader, policy, _, snapshot = setup_reader(tmp_path, [], steps=2)
    reader.repl_factory = None
    monkeypatch.setattr("src.product.qwen_investigator.PersistentREPL._docker_available",
                        lambda _: False)
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "unavailable" and policy.observations == []
    assert packet["execution"] == "docker_only"
