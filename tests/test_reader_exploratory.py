"""Saved-step-nine amendment gates; synthetic weights and no locked references."""

import copy
import shutil
import tempfile
from pathlib import Path

import pytest

from src.eval.reader_improvement import file_hash, load
from src.training import reader_evaluation as ev
from src.training.prime_run import write

PARENT = Path("examples/reader-rl-pilot-v2/protocol.json")
PROTOCOL = Path("examples/reader-step9-eval-20261026/protocol.json")


@pytest.mark.parametrize(
    "mutation", ["step", "hash", "questions", "order", "caps", "result", "proof"]
)
def test_explicit_amendment_rejects_candidate_or_protocol_drift(mutation):
    protocol = copy.deepcopy(load(PROTOCOL))
    plan = {"pilot_protocol": str(PARENT), "pilot_protocol_sha256": file_hash(PARENT)}
    result = {
        "verified_updates": 9,
        "attempted_update_steps": list(range(1, 10)),
        "unverified_attempts_may_have_updated_weights": False,
        "all_recorded_containers_absent": True,
    }
    assert ev.exploratory_contract(protocol, plan, result) == load(PARENT)
    if mutation == "step":
        protocol["selected_checkpoint_step"] = 8
    if mutation == "hash":
        protocol["selected_checkpoint_sha256"] = ev.V5_SHA
    if mutation == "questions":
        protocol["evaluation_question_ids"].reverse()
    if mutation == "order":
        protocol["phases"]["paired_evaluation"]["arm_order"].reverse()
    if mutation == "caps":
        protocol["phases"]["paired_evaluation"]["max_generation_requests"] += 1
    if mutation == "result":
        result["unverified_attempts_may_have_updated_weights"] = True
    if mutation == "proof":
        protocol["training_evidence_sha256"] = {}
    with pytest.raises(ValueError, match="amended"):
        ev.exploratory_contract(protocol, plan, result)


def test_portable_exploratory_freeze_preserves_original_result_and_has_correct_labels(monkeypatch):
    # Protocols must live inside the checkout. All ephemeral fixtures are under ignored out/.
    with tempfile.TemporaryDirectory(dir=ev.ROOT / "out", prefix="test-step9-") as path:
        root = Path(path)
        training = root / "training"
        training.mkdir()
        baseline = root / "baseline"
        baseline.mkdir()
        (baseline / "adapter_model.safetensors").write_bytes(b"fixture-original-v5")
        write(baseline / "adapter_config.json", {"fixture": True})
        v5 = file_hash(baseline / "adapter_model.safetensors")
        monkeypatch.setattr(ev, "V5_SHA", v5)
        parent = load(PARENT)
        phase = parent["phases"]["training"]
        write(
            training / "plan.json",
            {
                "limits": {
                    "updates": 10,
                    "requests": phase["max_generation_requests"],
                    "reserved_tokens": phase["max_reserved_tokens"],
                    "seconds": phase["max_work_seconds"],
                },
                "v5_sha256": v5,
                "train_case_ids": parent["training_case_ids"],
                "pilot_protocol": str(PARENT),
                "pilot_protocol_sha256": file_hash(PARENT),
            },
        )
        write(training / "started.json", {"synthetic": True})
        write(
            training / "result.json",
            {
                "verified_updates": 9,
                "attempted_update_steps": list(range(1, 10)),
                "unverified_attempts_may_have_updated_weights": False,
                "all_recorded_containers_absent": True,
            },
        )
        previous = v5
        optimizer = "initial"
        for n in range(1, 10):
            d = training / f"cycle-{n}"
            (d / "adapter").mkdir(parents=True)
            (d / "adapter/adapter_model.safetensors").write_bytes(f"fixture-step-{n}".encode())
            write(d / "adapter/adapter_config.json", {"fixture": True})
            (d / "microbatches.bin").write_bytes(b"fixture-batch")
            write(
                d / "verified.json",
                {
                    "step": n,
                    "optimizer_steps": [n],
                    "optimizer_sha256": f"optimizer-{n}",
                    "reload_matches_export": True,
                    "startup_matches_sampled_policy": True,
                    "adapter_sha256": file_hash(d / "adapter/adapter_model.safetensors"),
                },
            )
            write(
                d / "loaded.json",
                {
                    "version": n - 1,
                    "sha256": previous,
                    "load_rpc_completed": True,
                    "engine_reported_path_matches": True,
                },
            )
            write(
                d / "prepared.json",
                {
                    "version": n - 1,
                    "scorer_passed": True,
                    "optimizer_sha256": optimizer,
                    "batch_sha256": file_hash(d / "microbatches.bin"),
                },
            )
            write(d / "scorer.json", {"tokens": 5, "mean": 0.001, "max": 0.01})
            previous = file_hash(d / "adapter/adapter_model.safetensors")
            optimizer = f"optimizer-{n}"
        protocol = load(PROTOCOL)
        protocol["selected_checkpoint_sha256"] = previous
        protocol["training_evidence_sha256"] = {
            name: file_hash(training / name) for name in protocol["training_evidence_sha256"]
        }
        pp = root / "protocol.json"
        write(pp, protocol)
        original = (training / "result.json").read_bytes()
        session = ev.freeze(training, root / "session", exploratory_protocol=pp, baseline=baseline)
        assert ev.evaluation_arms(session) == ev.EXPLORATORY_ARMS
        assert "final_rl_step_10" not in session["arms"]
        assert (training / "result.json").read_bytes() == original
        assert not (root / "session/reference-pack.json").exists()
        shutil.copytree(root / "session", root / "moved")
        assert ev.validate_session(root / "session") == ev.validate_session(root / "moved")
        (root / "moved/final_rl_step_9/adapter_model.safetensors").write_bytes(b"changed")
        with pytest.raises(ValueError, match="artifact changed"):
            ev.validate_session(root / "moved")
        # The historical entry point still requires full completion; no global gate relaxation.
        monkeypatch.setattr(ev, "validate_plan", lambda output: load(training / "plan.json"))
        with pytest.raises(ValueError, match="ten-update"):
            ev.freeze(training, root / "old-gate")
