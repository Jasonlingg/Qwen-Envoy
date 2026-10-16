"""Exact, Docker-only reproduction of the saved October Nemotron environment.

Only hash-checked repository infrastructure is imported on the host. Recorded
model actions are passed exclusively to DockerREPL.execute().
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.env.corpus import Corpus
from src.env.repl import DockerREPL

HISTORICAL_COMMIT = "1976690"
ENV_SHA256 = "fa017c0a73adcf212361171bfedc7d8264db9fc17127ce0c7d70a5fae4acfbc4"
STATE_SHA256 = "69789ccd04a08b84a3dc50e01bba8ced2e71a333ebe756404da2b83df9a62dbe"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _historical_module(root: Path, relative: str, digest: str):
    source = subprocess.check_output(
        ["git", "show", f"{HISTORICAL_COMMIT}:{relative}"], cwd=root,
    )
    if hashlib.sha256(source).hexdigest() != digest:
        raise ValueError(f"Historical source hash mismatch: {relative}")
    name = "_envoy_replay_" + Path(relative).stem
    with tempfile.TemporaryDirectory(prefix="envoy-replay-source-") as directory:
        path = Path(directory) / Path(relative).name
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def historical_environment(root: Path, manifest: dict):
    """Restore recorded infrastructure; never import Python from the run JSON."""
    identities = manifest["implementation_sha256"]
    if identities["src/env/document_env.py"] != ENV_SHA256:
        raise ValueError("This replay supports only the recorded October environment")
    for relative in ("src/env/repl.py", "src/env/tools.py", "src/env/reward.py"):
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != identities[relative]:
            raise ValueError(f"Recorded infrastructure differs: {relative}")
    module = _historical_module(root, "src/env/document_env.py", ENV_SHA256)
    state = _historical_module(root, "src/env/evidence_state.py", STATE_SHA256)
    for name in ("EscalationEvent", "EvidenceState", "VerifierEvent",
                 "format_escalation", "format_feedback"):
        setattr(module, name, getattr(state, name))
    return module


def inspect_image(image: str) -> dict:
    info = json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]
    if (info.get("Config", {}).get("Labels") or {}).get("org.envoybench.sandbox") != "v1":
        raise ValueError("Replay image lacks the org.envoybench.sandbox=v1 label")
    return {"image_id": info["Id"], "repo_digests": info.get("RepoDigests", [])}


def compact_report(report: dict) -> dict:
    """Keep every action/result reviewable without duplicating megabytes of text."""
    summary = {**report, "questions": []}
    for question in report["questions"]:
        compact = {**question, "actions": [],
                   "initial_observation_sha256": sha256(question["initial_observation"])}
        del compact["initial_observation"]
        for item in question["actions"]:
            action = {}
            for key, value in item.items():
                if key.endswith("observation"):
                    action[key + "_sha256"] = sha256(value) if value is not None else None
                    action[key + "_chars"] = len(value) if value is not None else None
                else:
                    action[key] = value
            compact["actions"].append(action)
        summary["questions"].append(compact)
    return summary


def replay_baseline(module, row: dict, question: dict, corpus: Path,
                    manifest: dict, image: str) -> dict:
    """Replay all environment decisions and require exact saved observations."""
    executions: list[str] = []

    class RecordingDockerREPL(DockerREPL):
        def execute(self, code, timeout=30):
            observation = super().execute(code, timeout=timeout)
            executions.append(observation)
            return observation

    def docker_only(*, use_docker, corpus_path):
        if use_docker is not True:
            raise ValueError("Replay requires Docker; host execution is forbidden")
        return RecordingDockerREPL(image=image, corpus_path=corpus_path)

    module.PersistentREPL = docker_only
    loaded_corpus = Corpus(corpus_path=str(corpus))
    loaded_corpus.load(build_index=False)
    env = module.DocumentExplorationEnv(
        corpus=loaded_corpus, questions=[question],
        corpus_path=str(corpus), use_docker=True, max_steps=manifest["max_steps"],
        require_evidence=manifest["require_evidence"], include_preamble=True,
        evidence_verifier=manifest["evidence_verifier"],
        verifier_feedback_budget=manifest["verifier_feedback_budget"],
        escalate_after_verifier_failure=manifest["escalate_after_verifier_failure"],
    )
    results = []
    try:
        initial = env.reset()
        for saved in row["trajectory"]:
            before = len(executions)
            env.step(saved["action"])
            current = env.get_trajectory()[-1].model_dump()
            exact = all(current[key] == saved[key]
                        for key in ("step", "action", "observation", "reward", "done"))
            results.append({
                "step": saved["step"], "action": saved["action"],
                "recorded_observation": saved["observation"],
                "v1_observation": current["observation"], "v1_exact_match": exact,
                "v1_executed": len(executions) > before,
                "v1_tool_observation": executions[-1] if len(executions) > before else None,
                "v1_reward": current["reward"], "recorded_reward": saved["reward"],
                "v1_done": current["done"], "recorded_done": saved["done"],
                "normalization": "none",
            })
    finally:
        env.close()
    return {"question_id": row["question_id"], "question": row["question"],
            "initial_observation": initial, "v1_exact_match": all(
                action["v1_exact_match"] for action in results), "actions": results}
