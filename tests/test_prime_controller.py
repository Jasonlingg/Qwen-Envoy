"""File-handoff gates, no GPU, paid calls, locked references or optimizer."""

from __future__ import annotations

import copy
import time

import pytest

from src.eval.reader_improvement import file_hash, load
from src.training import prime_controller as bridge
from src.training.prime_run import prepare, validate_plan, write


@pytest.fixture
def handoff(tmp_path):
    controller, gpu = tmp_path / "controller", tmp_path / "gpu"
    local = prepare(controller, tmp_path / "local-assets", updates=2)
    plan = prepare(
        gpu,
        tmp_path / "gpu-assets",
        updates=2,
        controller_plan_sha256=file_hash(controller / "plan.json"),
    )
    assert bridge.contract(local) == bridge.contract(plan)
    cycle = gpu / "cycle-1"
    cycle.mkdir()
    write(gpu / "started.json", {"deadline": time.time() + 60})
    write(
        cycle / "loaded.json",
        {
            "version": 0,
            "sha256": "fixture",
            "load_rpc_completed": True,
            "engine_reported_path_matches": True,
        },
    )
    job = bridge.request(plan, 1)
    write(cycle / "microbatches.bin", {"synthetic": True})
    candidate = {
        "case_id": plan["train_case_ids"][0],
        "policy": job["policy"],
        "batch_sha256": file_hash(cycle / "microbatches.bin"),
        "rewards": [0.0, 1.0, 0.0, 1.0],
        "training_ready": False,
    }
    write(cycle / "candidate.json", candidate)
    response = {
        "job_digest": bridge.digest(job),
        "controller_plan_sha256": plan["controller_plan_sha256"],
        "healthy_original_episodes": 4,
        "all_recorded_containers_absent": True,
        "batch_sha256": candidate["batch_sha256"],
    }
    return controller, gpu, plan, job, response


def test_handoff_preserves_contract_and_accepts_only_bound_complete_mixed_group(handoff):
    _, _, plan, job, response = handoff
    bridge.accept(plan, 1, job, response)


@pytest.mark.parametrize(
    "mutation",
    [
        "cleanup",
        "job",
        "controller",
        "batch",
        "partial",
        "uniform",
        "evaluation",
        "policy",
        "source",
    ],
)
def test_changed_or_unhealthy_handoff_cannot_reach_update(handoff, mutation):
    _, gpu, plan, job, original = handoff
    response = copy.deepcopy(original)
    candidate = load(gpu / "cycle-1/candidate.json")
    if mutation == "cleanup":
        response["all_recorded_containers_absent"] = False
    elif mutation == "job":
        response["job_digest"] = "different job"
    elif mutation == "controller":
        response["controller_plan_sha256"] = "different controller"
    elif mutation == "batch":
        response["batch_sha256"] = "different batch"
    elif mutation == "partial":
        response["healthy_original_episodes"] = 3
    elif mutation == "uniform":
        candidate["rewards"] = [0.0] * 4
    elif mutation == "evaluation":
        candidate["case_id"] = "locked_evaluation_fixture"
    elif mutation == "policy":
        candidate["policy"] = {"sha256": "stale"}
    else:
        job["contract"]["prompt_sha256"] = "changed"
    write(gpu / "cycle-1/candidate.json", candidate)
    with pytest.raises(ValueError):
        bridge.accept(plan, 1, job, response)


def test_controller_wait_uses_existing_deadline_without_reset(handoff):
    _, _, plan, job, _ = handoff
    job["deadline"] = time.time() - 1
    # No response is accepted after expiration, even if already on disk.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(bridge, "request", lambda plan, step: job)
        with pytest.raises(TimeoutError):
            bridge.wait_for_controller(plan, 1)


@pytest.mark.parametrize("value", ["short", "z" * 64])
def test_plan_rejects_invalid_controller_identity(tmp_path, value):
    with pytest.raises(ValueError):
        prepare(tmp_path / "run", tmp_path / "assets", controller_plan_sha256=value)


def test_ordinary_plan_keeps_local_docker_default(tmp_path):
    root = tmp_path / "run"
    prepare(root, tmp_path / "assets")
    assert validate_plan(root)["controller_plan_sha256"] is None
