"""Original traces must reproduce before fixed-action corrections can be interpreted."""
import copy
import json
import os
from pathlib import Path

import pytest

from src.eval import reader_correction_replay as runner
from src.eval.reader_improvement import digest, load, write

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "examples/qasper-hybrid-v5"
QUEUE = ROOT / "examples/reader-improvement-loop"
REVIEW = QUEUE / "implementing-assistant-review.json"
DRAFTS = QUEUE / "correction-drafts.json"


def test_recorded_transport_requires_identical_input_and_has_no_network_fallback():
    payload = {"model": "test", "messages": [{"role": "user", "content": "actual observation"}]}
    response = {"usage": {"prompt_tokens": 1}, "choices": [{"message": {"content": "old"}}]}
    transport = runner.RecordedTransport([payload], [response], replacement="reviewed correction")
    with pytest.raises(ValueError, match="historical"):
        transport("unused", {"model": "changed"}, {}, 1)
    actual = transport("unused", payload, {}, 1)
    assert actual["choices"][0]["message"]["content"] == "reviewed correction"
    assert "usage" not in actual  # Do not relabel historical usage as a new generation.
    assert response["choices"][0]["message"]["content"] == "old"
    with pytest.raises(ValueError):
        transport("unused", payload, {}, 1)
    assert len(transport.receipts) == 1


def test_original_comparison_ignores_only_run_uuid():
    packet = {"status": "evidence_found", "retriever_run_id": "one",
              "trajectory": [{"action": "read", "observation": "actual"}]}
    other = copy.deepcopy(packet)
    other["retriever_run_id"] = "two"
    assert runner.original_matches(packet, other)
    other["trajectory"][0]["observation"] = "changed feedback"
    assert not runner.original_matches(packet, other)


def test_inputs_and_corrections_are_bound_to_the_original_run(tmp_path):
    queue, drafts = runner.validate_inputs(BUNDLE, QUEUE, REVIEW, DRAFTS)
    assert len(queue["records"]) == 3 and len(drafts["correction_drafts"]) == 2
    changed = copy.deepcopy(drafts)
    changed["correction_drafts"][0]["training_eligible"] = True
    path = tmp_path / "changed.json"
    write(path, changed)
    with pytest.raises(ValueError, match="drafts differ"):
        runner.validate_inputs(BUNDLE, QUEUE, REVIEW, path)


def test_current_runtime_change_cannot_silently_enter_a_frozen_replay(monkeypatch):
    real = runner.file_hash
    monkeypatch.setattr(runner, "file_hash", lambda p: "changed" if str(p).endswith(
        "/src/env/repl.py") else real(p))
    with pytest.raises(ValueError, match="runtime implementation changed"):
        runner.validate_inputs(BUNDLE, QUEUE, REVIEW, DRAFTS)


def test_original_mismatch_blocks_corrections_and_preserves_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.PersistentREPL, "_docker_available", lambda _: True)
    calls = []

    def fail_original(*args, candidate=None):
        calls.append(candidate)
        raise ValueError("original mismatch")
    monkeypatch.setattr(runner, "replay_one", fail_original)
    out = tmp_path / "replay"
    with pytest.raises(ValueError, match="original mismatch"):
        runner.replay(BUNDLE, QUEUE, REVIEW, DRAFTS, out)
    assert calls == [None]
    assert load(out / "summary.json")["status"] == "failed"
    assert load(out / "cleanup.json")["all_absent"] is True


def test_every_original_precedes_corrections_and_training_stays_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.PersistentREPL, "_docker_available", lambda _: True)
    calls = []

    def fixed_fixture(*args, candidate=None):
        calls.append(candidate is not None)
        return {"passed": True, "training_eligible": False}
    monkeypatch.setattr(runner, "replay_one", fixed_fixture)
    out = tmp_path / "replay"
    result = runner.replay(BUNDLE, QUEUE, REVIEW, DRAFTS, out)
    assert calls == [False, False, False, True, True]
    assert result["new_model_calls"] == result["training_ready"] == 0
    assert result["human_review"] is None
    with pytest.raises(FileExistsError):
        runner.replay(BUNDLE, QUEUE, REVIEW, DRAFTS, out)


def test_missing_docker_never_runs_local_or_creates_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.PersistentREPL, "_docker_available", lambda _: False)
    out = tmp_path / "replay"
    with pytest.raises(ValueError, match="no host fallback"):
        runner.replay(BUNDLE, QUEUE, REVIEW, DRAFTS, out)
    assert not out.exists()


def test_cleanup_attempts_other_owned_containers_after_an_inspection_failure(monkeypatch):
    inspected = []

    def inspect(container):
        inspected.append(container)
        if container == "first":
            raise RuntimeError("daemon read failed")
        return None
    monkeypatch.setattr(runner, "inspect_container", inspect)
    proof = runner.cleanup(["first", "second"])
    assert "second" in inspected
    assert proof[0]["absent"] is False and proof[1]["absent"] is True


def test_real_docker_originals_and_reviewed_corrections(tmp_path):
    image = os.environ.get("ENVOY_ASSIGNED_TEST_DOCKER_IMAGE")
    if not image:
        pytest.skip("set ENVOY_ASSIGNED_TEST_DOCKER_IMAGE for real Docker replay")
    assert image == load(QUEUE / "queue.json")["baseline"]["plan"]["sandbox_image"]
    out = tmp_path / "replay"
    result = runner.replay(BUNDLE, QUEUE, REVIEW, DRAFTS, out)
    assert result["status"] == "passed_fixed_action_regression_replay"
    assert len(result["originals"]) == 3 and len(result["corrections"]) == 2
    assert all(r["original_packet_matched_except_uuid"] for r in result["originals"])
    assert [r["status"] for r in result["corrections"]] == ["evidence_found", "no_evidence"]
    assert sum(r["request_payloads_matched"] for r in result["originals"] +
               result["corrections"]) == 15
    cleanup = load(out / "cleanup.json")
    assert cleanup["all_absent"] and len(cleanup["owned_containers"]) == 5
    for row in cleanup["owned_containers"]:
        assert runner.inspect_container(row["id"]) is None
    queue = load(QUEUE / "queue.json")
    for record in queue["records"]:
        prefix = out / "originals" / record["case_id"]
        packet = load(prefix / "packet.json")
        assert packet["trajectory"] == record["original_packet"]["trajectory"]
        receipts = load(prefix / "requests.json")["receipts"]
        assert digest(record["actual_final_request"]) == receipts[-1]["payload_sha256"]
        proof, = prefix.glob("*.sandbox.json")
        assert json.loads(proof.read_text())["network_mode"] == "none"
