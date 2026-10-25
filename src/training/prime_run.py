"""Bounded sequential launch glue; optimization and checkpointing belong to Prime-RL.

Preparation is CPU-only. Execution requires a separately provisioned Linux GPU,
local pinned assets and explicit --execute-local-gpu. It never rents hardware.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.eval.passage_pilot_replay import PROMPT_SHA, frozen_runtime
from src.eval.reader_improvement import file_hash, inside
from src.training.reader_data import (
    PILOT_REVIEW,
    ROOT,
    load_training_bundle,
)

BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
V5_REVISION = "27b912a863ff914ad45baa03624d8911dc1e17fb"
V5_SHA = "7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3"
PRIME_REVISION = "9e15490dcab1d506ce3462e69b8aa5dde45c6b3a"
RENDERER_REVISION = "5d870d8eaa65114f5bef8a4cffc146fbcb418ec4"


def check_limits(limits):
    for key, low, high in (
        ("updates", 1, 20),
        ("requests", 1, 4096),
        ("reserved_tokens", 8192, 33554432),
        ("seconds", 60, 14400),
    ):
        value = limits[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError("run limit outside the supported range")


def write(path, value):
    """Atomic local receipt: reservations must be durable before dispatch."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def prepare(output, assets, *, updates=2, requests=256, tokens=2097152, seconds=3600,
            controller_plan_sha256=None, pilot_protocol=None):
    limits = {
        "updates": updates,
        "requests": requests,
        "reserved_tokens": tokens,
        "seconds": seconds,
    }
    check_limits(limits)
    if controller_plan_sha256 is not None and (
            len(controller_plan_sha256) != 64
            or any(c not in "0123456789abcdef" for c in controller_plan_sha256)):
        raise ValueError("external collector requires an exact controller plan SHA-256")
    output, assets = Path(output).resolve(), Path(assets).resolve()
    bundle = ROOT / "data/rl/reader-facts-v1"
    protocol_path = inside(ROOT, pilot_protocol) if pilot_protocol is not None else None
    protocol = json.loads(protocol_path.read_text()) if protocol_path else None
    revision = protocol["identities"].get("reward_revision") if protocol else None
    revision_path = inside(ROOT, revision) if revision is not None else None
    manifest, _, _, _, cases, reward_version = load_training_bundle(bundle, revision_path)
    random.Random(42).shuffle(cases)
    files = [
        Path(__file__),
        ROOT / "src/training/prime_native.py",
        ROOT / "src/training/prime_budget.py",
        ROOT / "src/training/prime_controller.py",
        ROOT / "src/training/verifiers_adapter.py",
        ROOT / "src/training/reader_data.py",
        ROOT / "src/training/reader_rollout.py",
        ROOT / "src/eval/reader_rl_validation.py",
        ROOT / "src/eval/reader_rl_audit.py",
        ROOT / "examples/verifiers-port/prime-rl-smoke/native-source-proof.json",
        ROOT / "examples/verifiers-port/prime-base-files.json",
        PILOT_REVIEW,
    ]
    files += [p for p in (protocol_path, revision_path) if p is not None]
    plan = {
        "schema": "reader-prime-sequential-v1",
        "output": str(output),
        "assets": str(assets),
        "controller_plan_sha256": controller_plan_sha256,
        "bundle": str(bundle),
        "seed": 42,
        "limits": limits,
        "group_size": 4,
        "train_case_ids": cases,
        "base_revision": BASE_REVISION,
        "v5_revision": V5_REVISION,
        "v5_sha256": V5_SHA,
        "prime_revision": PRIME_REVISION,
        "renderer_revision": RENDERER_REVISION,
        "reward_version": reward_version,
        "source_revision": manifest["source_revision"],
        "bundle_sha256": file_hash(bundle / "manifest.json"),
        "training_review_sha256": file_hash(PILOT_REVIEW),
        "prompt_sha256": PROMPT_SHA,
        "frozen_runtime": frozen_runtime(),
        "implementation_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in files},
        "hypothesis": "On-policy mixed groups survive consecutive native updates and reloads",
        "decision_rule": "Stop on any identity, infrastructure, scoring, reload or budget failure",
        "quality_improvement_measured": False,
        "independent_human_review": None,
    }
    if protocol_path:
        plan.update(pilot_protocol=str(protocol_path.relative_to(ROOT)),
                    pilot_protocol_sha256=file_hash(protocol_path),
                    reward_revision=revision,
                    reward_revision_sha256=file_hash(revision_path) if revision_path else None)
        plan["hypothesis"] = protocol["hypothesis"]
        plan["decision_rule"] = protocol["stopping_rule"]
        check_pilot_contract(plan)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    return plan


def check_pilot_contract(plan):
    """A prepared quality pilot must match its prospective protocol exactly."""
    path = inside(ROOT, plan["pilot_protocol"])
    protocol = json.loads(path.read_text())
    phase, identities = protocol["phases"]["training"], protocol["identities"]
    if (file_hash(path) != plan["pilot_protocol_sha256"]
            or plan["limits"] != {"updates": phase["optimizer_updates"],
                                   "requests": phase["max_generation_requests"],
                                   "reserved_tokens": phase["max_reserved_tokens"],
                                   "seconds": phase["max_work_seconds"]}
            or set(plan["train_case_ids"]) != set(protocol["training_case_ids"])
            or phase["initialization"] != "original_pinned_v5_not_the_integration_adapter"
            or identities.get("reward_revision") != plan.get("reward_revision")
            or identities.get("reward_revision_sha256") != plan.get("reward_revision_sha256")
            or any(identities[k] != plan[k] for k in (
                "base_revision", "v5_revision", "v5_sha256", "prime_revision",
                "renderer_revision", "prompt_sha256", "reward_version", "source_revision"))
            or identities["bundle_manifest_sha256"] != plan["bundle_sha256"]
            or identities["training_review_sha256"] != plan["training_review_sha256"]):
        raise ValueError("training plan differs from the prospective pilot protocol")
    return protocol


def verify_base_assets(base):
    """Check streamed local bytes against the pinned public Hub Git/LFS identities."""
    import hashlib

    catalog = json.loads((ROOT / "examples/verifiers-port/prime-base-files.json").read_text())
    if catalog["revision"] != BASE_REVISION:
        raise ValueError("wrong base identity catalog")
    hashes = {}
    for name, expected in catalog["files"].items():
        path = base / name
        digest = hashlib.sha256() if expected["lfs_sha256"] else hashlib.sha1()
        if not expected["lfs_sha256"]:
            digest.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
        if hashes[name] != (expected["lfs_sha256"] or expected["git_blob_id"]):
            raise ValueError("base asset differs from pinned Hub revision: " + name)
    return hashes


def validate_plan(output):
    output = Path(output).resolve()
    plan = json.loads((output / "plan.json").read_text())
    if (output / "started.json").exists():
        started = json.loads((output / "started.json").read_text())
        if started["plan_sha256"] != file_hash(output / "plan.json"):
            raise ValueError("plan changed after execution started")
    check_limits(plan["limits"])
    controller = plan.get("controller_plan_sha256")
    if controller is not None and (not isinstance(controller, str) or len(controller) != 64
            or any(c not in "0123456789abcdef" for c in controller)):
        raise ValueError("invalid external controller binding")
    if plan["schema"] != "reader-prime-sequential-v1" or plan["output"] != str(output):
        raise ValueError("wrong run or relocated output")
    bundle = Path(plan["bundle"])
    revision = plan.get("reward_revision")
    revision_path = inside(ROOT, revision) if revision is not None else None
    if (plan.get("reward_revision_sha256") !=
            (file_hash(revision_path) if revision_path else None)):
        raise ValueError("training reward revision changed")
    _, _, _, _, selected, reward_version = load_training_bundle(bundle, revision_path)
    random.Random(42).shuffle(selected)
    if (
        plan["bundle_sha256"] != file_hash(bundle / "manifest.json")
        or plan.get("training_review_sha256") != file_hash(PILOT_REVIEW)
        or plan["frozen_runtime"] != frozen_runtime()
        or plan["prompt_sha256"] != PROMPT_SHA
        or plan["reward_version"] != reward_version
        or (
            plan["base_revision"],
            plan["v5_revision"],
            plan["v5_sha256"],
            plan["prime_revision"],
            plan["renderer_revision"],
            plan["group_size"],
        )
        != (BASE_REVISION, V5_REVISION, V5_SHA, PRIME_REVISION, RENDERER_REVISION, 4)
        or plan["seed"] != 42
        or plan["train_case_ids"] != selected
    ):
        raise ValueError("source, role, runtime or reward drift")
    for name, expected in plan["implementation_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("implementation changed after preparation: " + name)
    if plan.get("pilot_protocol") is not None:
        check_pilot_contract(plan)
    elif revision is not None:
        raise ValueError("a revised reward requires its prospective pilot protocol")
    return plan


class Processes:
    """Own only launched process groups, and bound every wait by one deadline."""

    def __init__(self, deadline):
        self.deadline, self.children = deadline, []

    def remaining(self):
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("whole-run wall clock exhausted")
        return remaining

    def start(self, command, log, *, distributed=False):
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")
        if distributed:
            env.update(
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT="29571",
                RANK="0",
                LOCAL_RANK="0",
                WORLD_SIZE="1",
                LOCAL_WORLD_SIZE="1",
            )
        self.remaining()
        with Path(log).open("xb") as stream:
            child = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
                cwd=ROOT,
            )
        self.children.append(child)
        return child

    def wait(self, child):
        code = child.wait(timeout=self.remaining())
        self.stop(child)
        if code:
            raise RuntimeError(f"owned process {child.pid} exited {code}; inspect run logs")

    def stop(self, child):
        if child not in self.children:
            return
        # A launcher can exit before its engine children. Always signal its group.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
        self.children.remove(child)

    def close(self):
        for child in list(reversed(self.children)):
            self.stop(child)


def execute(output):
    plan = validate_plan(output)
    output = Path(output).resolve()
    # Exclusive creation also prevents restart from silently resetting budgets.
    with (output / "started.json").open("x") as stream:
        json.dump(
            {"started": time.time(), "deadline": time.time() + plan["limits"]["seconds"],
             "plan_sha256": file_hash(output / "plan.json")}, stream
        )
    deadline = json.loads((output / "started.json").read_text())["deadline"]
    processes = Processes(deadline)
    command = [sys.executable, "-m", "src.training.prime_native", str(output)]
    completed = 0
    attempted_updates = []
    try:

        def stage(mode, step=0, distributed=False):
            child = processes.start(
                command + [mode, str(step)],
                output / f"{step:02}-{mode}.log",
                distributed=distributed,
            )
            processes.wait(child)

        stage("preflight")
        for step in range(1, plan["limits"]["updates"] + 1):
            (output / f"cycle-{step}").mkdir()
            stage("configure", step)
            inference = processes.start(
                command + ["inference", str(step)], output / f"{step:02}-inference.log"
            )
            try:
                stage("load", step)
                if plan.get("controller_plan_sha256") is None:
                    stage("collect", step)
                else:
                    from src.training.prime_controller import wait_for_controller

                    wait_for_controller(plan, step)
            finally:
                processes.stop(inference)
            stage("prepare", step, distributed=True)
            receiver = processes.start(
                command + ["weights", str(step)], output / f"{step:02}-weights.log"
            )
            sender = processes.start(
                command + ["batch", str(step)], output / f"{step:02}-batch.log"
            )
            attempted_updates.append(step)
            trainer = processes.start(
                command + ["train", str(step)], output / f"{step:02}-train.log", distributed=True
            )
            for child in (trainer, receiver, sender):
                processes.wait(child)
            stage("verify", step, distributed=True)
            completed = step
    finally:
        processes.close()
        # The controller may have been killed before its finally block. Remove
        # only container IDs recorded by this run, never account-wide resources.
        from src.eval.reader_correction_replay import cleanup

        tracked = []
        for path in output.glob("cycle-*/rollouts/*/owned-containers.json"):
            tracked.extend(json.loads(path.read_text()))
        proof = cleanup(tracked) if tracked else []
        write(
            output / "result.json",
            {
                "verified_updates": completed,
                "attempted_update_steps": attempted_updates,
                "unverified_attempts_may_have_updated_weights": len(attempted_updates) > completed,
                "quality_improvement_measured": False,
                "cleanup": proof,
                "all_recorded_containers_absent": all(p["absent"] for p in proof),
            },
        )
        if any(not p["absent"] for p in proof):
            raise RuntimeError("owned Docker cleanup could not be verified; inspect result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--seconds", type=int)
    parser.add_argument("--controller-plan-sha256")
    parser.add_argument("--pilot-protocol", type=Path)
    parser.add_argument("--execute-local-gpu", action="store_true")
    args = parser.parse_args()
    if args.execute_local_gpu:
        if any(value is not None for value in (
                args.assets, args.updates, args.requests, args.tokens, args.seconds,
                args.controller_plan_sha256, args.pilot_protocol)):
            parser.error("execution uses the saved plan; set paths and limits during preparation")
        def stop_requested(signum, frame):
            raise KeyboardInterrupt("termination requested; cleaning up owned processes")

        previous = signal.signal(signal.SIGTERM, stop_requested)
        try:
            execute(args.output)
        finally:
            signal.signal(signal.SIGTERM, previous)
    else:
        if args.assets is None:
            parser.error("preparation requires --assets (local base/<revision> and v5 directories)")
        prepare(
            args.output,
            args.assets,
            updates=2 if args.updates is None else args.updates,
            requests=256 if args.requests is None else args.requests,
            tokens=2097152 if args.tokens is None else args.tokens,
            seconds=3600 if args.seconds is None else args.seconds,
            controller_plan_sha256=args.controller_plan_sha256,
            pilot_protocol=args.pilot_protocol,
        )
        print("Prepared only; no model, GPU, provider or infrastructure calls.")


if __name__ == "__main__":
    main()
