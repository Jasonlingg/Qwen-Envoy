"""Paired-evaluation gates with synthetic records; never decode locked references."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.reader_improvement import file_hash, load
from src.training import reader_evaluation as ev
from src.training.prime_budget import loopback_engine
from src.training.prime_run import prepare, write
from src.training.reader_data import load_bundle


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8000",
        "http://localhost:8000",
        "http://example.com:8000",
        "http://u:p@127.0.0.1:8000",
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1",
        "http://127.0.0.1:8000?x=1",
        "http://127.0.0.1:8000#x",
    ],
)
def test_native_connection_only_permits_explicit_loopback_forward(url):
    with pytest.raises(ValueError):
        loopback_engine(url)


def test_loopback_forward_can_use_nondefault_ssh_port():
    assert loopback_engine("http://127.0.0.1:18000/") == "http://127.0.0.1:18000"


def test_missing_candidate_cannot_open_locked_references(tmp_path, monkeypatch):
    original = ev.load

    def reject_locked(path):
        assert "model-evaluation-locked" not in str(path)
        return original(path)

    monkeypatch.setattr(ev, "load", reject_locked)
    with pytest.raises(FileNotFoundError):
        ev.open_references(tmp_path)
    run = tmp_path / "integration"
    prepare(run, tmp_path / "assets", updates=2)
    write(run / "result.json", {"verified_updates": 2})
    with pytest.raises(ValueError, match="ten-update"):
        ev.freeze(run, tmp_path / "session")
    assert not (tmp_path / "reference-pack.json").exists()


@pytest.fixture
def synthetic_pair(tmp_path, monkeypatch):
    """Mock the unopened inventory gate only; exercise real pack/reducer code."""
    control = load(next(Path("examples/verifiers-port/controls").glob("*.json")))
    packet = control["packet"]
    _, roles, _, _ = load_bundle(ev.BUNDLE)
    case = copy.deepcopy(roles["train"][0])
    cases = [{**case, "question_id": f"synthetic_question_{i}"} for i in range(12)]
    write(tmp_path / "session.json", {"fixture": "not a real frozen candidate"})
    monkeypatch.setattr(ev, "validate_session", lambda output: ({}, cases))
    monkeypatch.setattr(ev, "reference_review", lambda output, cases: "synthetic_review")
    monkeypatch.setattr(ev, "source_windows", lambda c: [{"quote": "fixture source"}])
    for arm in ev.ARMS:
        (tmp_path / arm).mkdir()
        write(tmp_path / arm / "complete.json", {"episodes": 12})
        for i, c in enumerate(cases):
            episode = {"traces": [{}], "docker_episode": {"packet": packet}}
            path = tmp_path / arm / f"episode-{i:02}.json"
            write(path, episode)
            write(
                tmp_path / arm / f"measure-{i:02}.json",
                {
                    "question_id": c["question_id"],
                    "seed": 42 + i,
                    "seconds": 1.0,
                    "reserved_tokens": 100,
                    "episode_sha256": file_hash(path),
                },
            )
    ev.export_review(tmp_path)
    write(tmp_path / "billing-receipt.json", {"synthetic_fixture_not_a_real_bill": True})
    write(tmp_path / "lifecycle.json", {
        "session_sha256": file_hash(tmp_path / "session.json"), "cost_kind": "provider_billed",
        "billing_receipt_sha256": file_hash(tmp_path / "billing-receipt.json"),
        "actual_spend_usd": 0, "gpu_lifetime_seconds": 1,
        "startup_seconds_by_arm": {a: 0 for a in ev.ARMS}})
    return tmp_path


def grade_fixture(output):
    review = load(output / "output-review.json")
    mapping = load(output / "private-arm-map.json")
    review.update(
        reviewer_kind="model_assisted",
        reviewer_identity="scripted_test_fixture",
        reviewed_at="2026-10-24",
        independent_human_review=None,
    )
    for row in review["rows"]:
        key = mapping[row["review_id"]]
        passed = key["arm"] == ev.ARMS[1] or key["index"] >= 3
        row.update(
            complete_correct_supported=passed,
            false_refusal=False,
            unresolved=False,
            claim_inventory_complete=True, requested_fact_review="synthetic completeness review",
            notes="synthetic judgment, not model evaluation",
            material_claims=[
                {
                    "claim": "fixture claim",
                    "supported": True,
                    "evidence_ids": [row["evidence"][0]["evidence_id"]],
                    "notes": "fixture source judgment",
                }
            ],
        )
    write(output / "output-review.json", review)
    return review


def test_masked_pack_has_no_arm_and_unfilled_review_is_not_a_gain(synthetic_pair):
    review = load(synthetic_pair / "output-review.json")
    assert len(review["rows"]) == 24
    assert all(r["complete_correct_supported"] is None for r in review["rows"])
    assert "unchanged_v5" not in json.dumps(review)
    assert "final_rl_step_10" not in json.dumps(review)
    with pytest.raises(ValueError, match="attributed"):
        ev.summarize(synthetic_pair)
    grade_fixture(synthetic_pair)
    result = ev.summarize(synthetic_pair)
    assert (result["wins"], result["losses"], result["ties"]) == (3, 0, 9)
    assert result["decision"] == "provisional_pilot_signal"
    assert result["independent_human_review"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "changed_source",
        "duplicate",
        "uncertain",
        "uncited",
        "fake_human",
        "latency",
        "new_unsupported",
    ],
)
def test_review_rejects_bad_bindings_or_fails_predeclared_gate(synthetic_pair, mutation):
    review = grade_fixture(synthetic_pair)
    row = review["rows"][0]
    if mutation == "changed_source":
        row["question"] = "edited after outcomes"
    elif mutation == "duplicate":
        review["rows"][1] = copy.deepcopy(row)
    elif mutation == "uncertain":
        row["unresolved"] = True
    elif mutation == "uncited":
        row["material_claims"][0]["evidence_ids"] = []
    elif mutation == "fake_human":
        review["independent_human_review"] = "invented"
    elif mutation == "latency":
        path = synthetic_pair / ev.ARMS[1] / "measure-00.json"
        measure = load(path)
        measure["seconds"] = 100.0
        write(path, measure)
    else:
        mapping = load(synthetic_pair / "private-arm-map.json")
        row = next(
            r
            for r in review["rows"]
            if mapping[r["review_id"]] == {**mapping[r["review_id"]], "arm": ev.ARMS[1], "index": 0}
        )
        row["complete_correct_supported"] = False
        row["material_claims"][0]["supported"] = False
    write(synthetic_pair / "output-review.json", review)
    if mutation == "new_unsupported":
        assert ev.summarize(synthetic_pair)["decision"] == "threshold_not_met"
    else:
        with pytest.raises(ValueError):
            ev.summarize(synthetic_pair)


def test_task_session_cannot_change_question_prompt_or_training_role(tmp_path, monkeypatch):
    _, roles, _, _ = load_bundle(ev.BUNDLE)
    case = roles["train"][0]  # Explicit fixture, never a real locked case.
    write(tmp_path / "session.json", {"fixture": True})
    session = {"bundle_sha256": "manifest_fixture"}
    monkeypatch.setattr(ev, "validate_session", lambda output: (session, [case]))
    data = SimpleNamespace(
        evaluation_session_sha256=file_hash(tmp_path / "session.json"),
        manifest_sha256="manifest_fixture",
        bundle=ev.BUNDLE,
        case_id=case["case_id"],
        id=case["question_id"],
        prompt=case["question"],
        system_prompt=ev.PROMPT,
    )
    assert ev.evaluation_case(tmp_path, data)[1] == case
    data.prompt = "altered"
    with pytest.raises(ValueError):
        ev.evaluation_case(tmp_path, data)


def test_second_arm_cannot_run_before_v5_and_partial_arm_cannot_restart(tmp_path, monkeypatch):
    session_path = tmp_path / "session.json"
    write(session_path, {"fixture": True})
    session = {"arms": {a: "synthetic_sha" for a in ev.ARMS}}
    monkeypatch.setattr(ev, "validate_session", lambda output: (session, []))
    monkeypatch.setattr(ev, "reference_review", lambda output, cases: "ref_sha")
    monkeypatch.setattr(ev, "check_tokenizer", lambda path: None)
    tokenizer = tmp_path / ev.BASE_REVISION
    tokenizer.mkdir()
    backend = tmp_path / "backend.json"
    receipt = {
        "session_sha256": file_hash(session_path),
        "arm": ev.ARMS[1],
        "adapter_sha256": "synthetic_sha",
        "load_rpc_completed": True,
        "gpu": "scripted fixture",
        "versions": {"fixture": True},
        "adapter_root": "/fixture",
    }
    write(backend, receipt)
    with pytest.raises(ValueError, match="V5 must complete"):
        asyncio.run(ev.run_arm(tmp_path, ev.ARMS[1], "http://127.0.0.1:18000", tokenizer, backend))
    arm = tmp_path / ev.ARMS[0]
    arm.mkdir()
    write(arm / "attempted.json", {"partial": True})
    receipt["arm"] = ev.ARMS[0]
    write(backend, receipt)
    with pytest.raises(FileExistsError):
        asyncio.run(ev.run_arm(tmp_path, ev.ARMS[0], "http://127.0.0.1:18000", tokenizer, backend))


@pytest.mark.parametrize("protocol_path", [None, Path("examples/reader-rl-pilot-v2/protocol.json")])
def test_candidate_freeze_is_portable_and_rejects_checkpoint_or_source_drift(
    tmp_path, monkeypatch, protocol_path
):
    import hashlib
    import shutil

    assets, training = tmp_path / "assets", tmp_path / "training"
    (assets / "v5").mkdir(parents=True)
    (assets / "v5/adapter_model.safetensors").write_bytes(b"synthetic-v5")
    write(assets / "v5/adapter_config.json", {"fixture": True})
    sha = hashlib.sha256(b"synthetic-v5").hexdigest()
    monkeypatch.setattr(ev, "V5_SHA", sha)
    phase = load(protocol_path or ev.PROTOCOL)["phases"]["training"]
    plan = prepare(training, assets, updates=10, requests=1024, tokens=8388608,
                   seconds=phase["max_work_seconds"], pilot_protocol=protocol_path)
    plan["v5_sha256"] = sha
    # Explicit synthetic training evidence; does not assert a real optimizer update.
    monkeypatch.setattr(ev, "validate_plan", lambda path: plan)
    write(training / "started.json", {"fixture": True})
    write(training / "result.json", {"verified_updates": 10,
        "attempted_update_steps": list(range(1, 11)),
        "unverified_attempts_may_have_updated_weights": False,
        "all_recorded_containers_absent": True})
    previous_sha, previous_optimizer = sha, "synthetic_initial_optimizer"
    for step in range(1, 11):
        cycle = training / f"cycle-{step}"
        (cycle / "adapter").mkdir(parents=True)
        (cycle / "adapter/adapter_model.safetensors").write_bytes(f"synthetic-{step}".encode())
        write(cycle / "adapter/adapter_config.json", {"fixture": True})
        write(cycle / "verified.json", {"step": step, "optimizer_steps": [step],
                "optimizer_sha256": f"synthetic_optimizer_{step}",
            "reload_matches_export": True, "startup_matches_sampled_policy": True,
            "adapter_sha256": file_hash(cycle / "adapter/adapter_model.safetensors")})
        (cycle / "microbatches.bin").write_bytes(b"synthetic_batch")
        write(cycle / "scorer.json", {"tokens": 1, "mean": .001, "max": .01})
        write(cycle / "loaded.json", {"version": step - 1, "sha256": previous_sha,
              "load_rpc_completed": True, "engine_reported_path_matches": True})
        write(cycle / "prepared.json", {"version": step - 1, "scorer_passed": True,
              "optimizer_sha256": previous_optimizer,
              "batch_sha256": file_hash(cycle / "microbatches.bin")})
        previous_sha = file_hash(cycle / "adapter/adapter_model.safetensors")
        previous_optimizer = f"synthetic_optimizer_{step}"
    output = tmp_path / "session"
    ev.freeze(training, output)
    assert ev.session_protocol(output)["schema_version"] == load(
        protocol_path or ev.PROTOCOL)["schema_version"]
    assert not (output / "reference-pack.json").exists()
    moved = tmp_path / "copied-controller-session"
    shutil.copytree(output, moved)
    assert ev.validate_session(output) == ev.validate_session(moved)
    (moved / ev.ARMS[1] / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        ev.validate_session(moved)
    verified = load(training / "cycle-10/verified.json")
    verified["optimizer_steps"] = [1]
    write(training / "cycle-10/verified.json", verified)
    with pytest.raises(ValueError, match="continuity"):
        ev.freeze(training, tmp_path / "bad-session")


def test_missing_actual_cost_cannot_become_positive_signal(synthetic_pair):
    grade_fixture(synthetic_pair)
    (synthetic_pair / "lifecycle.json").unlink()
    result = ev.summarize(synthetic_pair)
    assert result["answer_and_episode_resource_thresholds_met"] is True
    assert result["decision"] == "inconclusive_missing_lifecycle_cost"
