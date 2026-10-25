"""File handoff between the owned GPU launcher and the existing Docker collector.

SSH/file transfer is deployment glue, not a new trainer or model protocol. GPU
waits never execute generated Python; local collection still owns frozen Docker.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from pathlib import Path

from src.eval.reader_improvement import digest, file_hash
from src.training.prime_run import validate_plan, write


def contract(plan):
    return {
        k: v
        for k, v in plan.items()
        if k not in {"output", "assets", "bundle", "controller_plan_sha256"}
    }


def request(plan, step):
    root = Path(plan["output"])
    cycle = root / f"cycle-{step}"
    return {
        "schema": "reader-external-collection-v1",
        "step": step,
        "nonce": secrets.token_hex(16),
        "contract": contract(plan),
        "controller_plan_sha256": plan["controller_plan_sha256"],
        "gpu_plan_sha256": file_hash(root / "plan.json"),
        "deadline": json.loads((root / "started.json").read_text())["deadline"],
        "policy": json.loads((cycle / "loaded.json").read_text()),
    }


def accept(plan, step, job, response):
    cycle = Path(plan["output"]) / f"cycle-{step}"
    candidate = json.loads((cycle / "candidate.json").read_text())
    rewards = candidate.get("rewards", [])
    if (
        job["contract"] != contract(plan)
        or job["step"] != step
        or response.get("job_digest") != digest(job)
        or response.get("controller_plan_sha256") != plan["controller_plan_sha256"]
        or response.get("all_recorded_containers_absent") is not True
        or response.get("healthy_original_episodes", 0) < 4
        or response.get("batch_sha256") != file_hash(cycle / "microbatches.bin")
        or candidate.get("batch_sha256") != response["batch_sha256"]
        or candidate.get("policy") != job["policy"]
        or candidate.get("case_id") not in plan["train_case_ids"]
        or len(rewards) != 4
        or set(rewards) != {0.0, 1.0}
        or candidate.get("training_ready") is not False
    ):
        raise ValueError("unbound, unhealthy or incomplete external collection; no update")


def wait_for_controller(plan, step):
    root = Path(plan["output"])
    job = request(plan, step)
    cycle = root / f"cycle-{step}"
    write(cycle / "collection-request.json", job)
    while time.time() < job["deadline"]:
        response = cycle / "collection-complete.json"
        if response.exists():
            accept(plan, step, job, json.loads(response.read_text()))
            return
        time.sleep(0.25)
    raise TimeoutError("external collector did not finish before the unchanged run deadline")


def collect(output, job_path, engine):
    """On Docker controller; caller transfers only the completed, bound result."""
    from src.training.prime_native import collect_group

    output = Path(output).resolve()
    plan, job = validate_plan(output), json.loads(Path(job_path).read_text())
    step = job["step"]
    if (
        job.get("schema") != "reader-external-collection-v1"
        or job["controller_plan_sha256"] != file_hash(output / "plan.json")
        or job["contract"] != contract(plan)
        or type(step) is not int
        or not 1 <= step <= plan["limits"]["updates"]
        or not time.time() < job["deadline"] <= time.time() + plan["limits"]["seconds"]
        or job["policy"].get("version") != step - 1
        or job["policy"].get("load_rpc_completed") is not True
        or job["policy"].get("engine_reported_path_matches") is not True
    ):
        raise ValueError("external job differs from the frozen controller plan")
    started = output / "started.json"
    if not started.exists():
        if step != 1:
            raise ValueError("external collection must begin at step one")
        with started.open("x") as stream:
            json.dump(
                {
                    "deadline": job["deadline"],
                    "plan_sha256": file_hash(output / "plan.json"),
                    "gpu_plan_sha256": job["gpu_plan_sha256"],
                },
                stream,
            )
    state = json.loads(started.read_text())
    if state["deadline"] != job["deadline"] or state["gpu_plan_sha256"] != job["gpu_plan_sha256"]:
        raise ValueError("controller cannot attach to another run or reset its budget")
    cycle = output / f"cycle-{step}"
    cycle.mkdir(exist_ok=False)
    write(cycle / "collection-request.json", job)
    write(cycle / "loaded.json", job["policy"])
    asyncio.run(collect_group(plan, step, engine=engine))
    episodes = list(cycle.glob("episode-*.json"))
    if len(episodes) < 4:
        raise ValueError("external collection has no complete original group")
    for path in episodes:
        episode = json.loads(path.read_text())
        if not episode.get("ok") or len(episode.get("traces", [])) != 1:
            raise ValueError("unhealthy original external episode")
        trace = episode["traces"][0]
        if not trace.get("ok") or trace["agent"]["trainable"] is not False:
            raise ValueError("original external traces must stay healthy and untrainable")
    proofs = list((cycle / "rollouts").glob("*/cleanup.json"))
    if len(proofs) != len(episodes) or any(
        json.loads(p.read_text()).get("all_absent") is not True for p in proofs
    ):
        raise ValueError("external Docker cleanup proof is missing")
    response = {
        "job_digest": digest(job),
        "controller_plan_sha256": file_hash(output / "plan.json"),
        "healthy_original_episodes": len(episodes),
        "all_recorded_containers_absent": True,
        "batch_sha256": file_hash(cycle / "microbatches.bin"),
        "raw_episode_sha256": {p.name: file_hash(p) for p in episodes},
        "cleanup_sha256": {str(p.relative_to(cycle)): file_hash(p) for p in proofs},
    }
    write(cycle / "collection-complete.json", response)
    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("job", type=Path)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()
    collect(args.output, args.job, args.engine)
