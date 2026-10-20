"""Replay saved actions in Docker; no model transport, weights or GPU activation."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env.repl import PersistentREPL
from src.product.assigned_guidance import BASELINE, FOCUSED
from src.product.packet_verifier import verify_investigation_packet
from src.product.qwen_investigator import QwenInvestigator
from src.research.agent import load_snapshot

CHECK_FIELDS = ("status", "candidate_claims", "evidence", "research_gaps", "trajectory",
    "document_tool_steps", "inspection_steps", "model_requests_attempted",
    "system_prompt_sha256", "tool_preamble_sha256", "harness_shim_sha256",
    "paper_assignment", "assignment_sha256", "max_steps")


class RecordedActionsExhaustedError(RuntimeError):
    pass


class RecordedPolicy:
    def __init__(self, actions):
        self.actions, self.observations = actions, []

    def reset(self):
        self.observations = []

    def act(self, observation):
        index = len(self.observations)
        self.observations.append(observation)
        if index >= len(self.actions):
            raise RecordedActionsExhaustedError("no recorded response to the new feedback")
        return self.actions[index]


def require_baseline(packet, policy, original, requests):
    for field in CHECK_FIELDS:
        if packet.get(field) != original.get(field):
            raise ValueError(f"baseline differs from recorded run: {field}")
    expected = [row["payload"]["messages"][-1]["content"] for row in requests]
    if policy.observations != expected:
        raise ValueError("baseline differs from recorded model-input observations")


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def replay(prepared: Path, recorded: Path, output: Path):
    plan_bytes = (prepared / "plan.json").read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != (prepared / "plan.sha256").read_text().strip():
        raise ValueError("frozen plan hash mismatch")
    plan = json.loads(plan_bytes)
    manifest, docs = load_snapshot(Path(plan["snapshot"]))
    if manifest["corpus_hash"] != plan["corpus_hash"]:
        raise ValueError("snapshot differs from the recorded corpus")
    if not PersistentREPL._docker_available(plan["sandbox_image"]):
        raise ValueError("pinned Docker image required; no host execution fallback")
    output.mkdir(parents=True, exist_ok=False)
    report = {"mode": "offline_fixed_action_replay", "actual_model_requests": 0,
        "gpu_activations": 0, "sandbox_image": plan["sandbox_image"],
        "prepared_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "corpus_hash": manifest["corpus_hash"], "baseline_match": False, "cases": [],
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [*plan["implementation_sha256"], "scripts/replay_assigned_guidance.py",
                         "src/product/assigned_guidance.py"]},
        "interpretation": "Recorded actions are fixed. Replay cannot measure how Qwen would "
            "respond to new feedback, answer quality improvement, or fine-tuning benefit."}
    originals = {}
    inputs = json.loads((prepared / "model_inputs.json").read_text())
    for case, initial in zip(plan["cases"], inputs, strict=True):
        cid = case["case_id"]
        original = json.loads((recorded / f"{cid}.packet.json").read_text())
        ledger = [json.loads(line) for line in
                  (recorded / f"{cid}.requests.jsonl").read_text().splitlines()]
        requests = [row for row in ledger if row["event"] == "reserved"]
        if requests[0]["payload"] != initial["payload"]:
            raise ValueError("recorded initial request differs from frozen model input")
        originals[cid] = original, requests

    # Both baseline cases must reproduce before any candidate action executes.
    for guidance in (BASELINE, FOCUSED):
        for case in plan["cases"]:
            cid = case["case_id"]
            original, requests = originals[cid]
            policy = RecordedPolicy([trace["action"] for trace in original["trajectory"]])
            reader = QwenInvestigator(lambda: policy, original["model_identity"],
                max_steps=original["max_steps"], image=plan["sandbox_image"],
                source_domain="public_papers", harness_version="v2", reader_mode="assigned",
                reader_guidance=guidance)
            packet = reader.investigate(case["question"], Path(plan["snapshot"]))
            if guidance == BASELINE:
                try:
                    require_baseline(packet, policy, original, requests)
                except ValueError as exc:
                    report["error"] = str(exc)
                    write(output / "replay.json", report)
                    raise
            exhausted = packet.get("error", "").startswith("RecordedActionsExhaustedError:")
            checked = verify_investigation_packet(packet, manifest, docs,
                allowed_prompt_sha256s=frozenset({
                    hashlib.sha256(reader.system_prompt.encode()).hexdigest()}),
                expected_reader_mode="assigned", expected_reader_guidance=guidance)
            packet["replay"] = {"actual_model_requests": 0, "fixed_actions": True,
                                "recorded_actions_exhausted": exhausted}
            write(output / f"{cid}.{guidance}.json", packet)
            (output / f"{guidance}.prompt.txt").write_text(reader.system_prompt)
            report["cases"].append({"case_id": cid, "guidance": guidance,
                "status": packet["status"], "verified_status": checked["status"],
                "recorded_actions_exhausted": exhausted,
                "note": ("Fixed actions ended after new feedback; no model continuation "
                         "was generated." if exhausted else "No new model response generated."),
                "rejected_steps": [t["step"] for t in packet["trajectory"] if t.get(
                    "observation", "").startswith(("Action rejected:", "Submission rejected:"))],
                "prompt_sha256": packet["system_prompt_sha256"]})
        if guidance == BASELINE:
            report["baseline_match"] = True
    write(output / "replay.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "recorded", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.prepared, args.recorded, args.output), indent=2))


if __name__ == "__main__":
    main()
