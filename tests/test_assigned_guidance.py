"""Candidate prompt identity and baseline compatibility; no model requests."""
import hashlib
from pathlib import Path

import pytest

from src.product.assigned_guidance import FOCUSED
from src.product.assigned_reader import build_assigned_reader
from src.product.qwen_investigator import effective_system_prompt
from src.research.agent import load_snapshot
from tests.test_assigned_reader import DOC_ID, INSPECT, SUBMIT, SourceREPL, setup_reader, verify


def test_baseline_assigned_prompt_matches_saved_live_bytes():
    saved = Path("examples/assigned-reader-smoke-prepared/system-prompt.txt").read_text()
    assert effective_system_prompt("public_papers", "default", "v2",
                                   reader_mode="assigned") == saved
    assert hashlib.sha256(saved.encode()).hexdigest() == (
        "721d7dc8bb9c4c8bef05b53923f9daa581715b6750e206c34d78e722cac96e56")


@pytest.mark.parametrize("kind", ["qwen", "nemotron"])
def test_opt_in_candidate_policy_investigator_and_packet_share_prompt(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    replies, sent = iter([INSPECT, SUBMIT]), []
    def transport(url, payload, headers, timeout):
        sent.append(payload)
        return {"choices": [{"message": {"content": next(replies)}}]}
    reader = build_assigned_reader(kind, reader_guidance=FOCUSED,
        qwen_endpoint="http://test", qwen_model="fake", qwen_checkpoint="fake-pinned",
        policy_transport=transport, max_steps=2)
    _, _, _, snapshot = setup_reader(tmp_path, [], steps=2)
    _, docs = load_snapshot(snapshot)
    reader.repl_factory = lambda _: SourceREPL(docs[DOC_ID]["text"])
    packet = reader.investigate("Memory?", snapshot)
    assert packet["status"] == "evidence_found" and packet["reader_guidance"] == FOCUSED
    assert reader.system_prompt == reader.policy_factory().system_prompt
    assert sent[0]["messages"][0]["content"] == reader.system_prompt
    assert packet["system_prompt_sha256"] == verify(reader, packet, snapshot)["prompt_hash"]
    assert verify(reader, packet, snapshot)["reader_guidance"] == FOCUSED
    packet.pop("reader_guidance")
    with pytest.raises(ValueError, match="reader guidance"):
        verify(reader, packet, snapshot)
    assert "you may\nuse one standalone control action" not in reader.system_prompt
    assert "you must\nuse one standalone control action" in reader.system_prompt
    assert "Never call submit()" not in reader.system_prompt
    assert "a real quote can contradict your claim" in reader.system_prompt
    assert "A5000" not in reader.system_prompt and "batch size of 1" not in reader.system_prompt
    assert "2211.17192" not in reader.system_prompt


@pytest.mark.parametrize("mode,guidance", [("assigned", "unknown"), ("discovery", FOCUSED)])
def test_candidate_cannot_be_selected_for_an_unknown_or_discovery_protocol(mode, guidance):
    with pytest.raises(ValueError, match="reader_guidance|requires the assigned"):
        effective_system_prompt("public_papers", "default", "v2",
                                reader_mode=mode, reader_guidance=guidance)
