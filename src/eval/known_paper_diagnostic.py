"""Isolated v5 training-protocol diagnostic; never changes product discovery gates.

Uses the original Python preamble and citation-only submission format. The
caller reviews answer support separately; no training reward is calculated.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from src.env.repl import PersistentREPL
from src.env.reward import parse_submission_details
from src.product.qwen_investigator import _used_inspection_tool
from src.research.agent import load_snapshot

TRAINING_PROMPT_SHA256 = "7b5ae820827750b2dd93f5a2d1c67bacf2c1d13c3def62b17aec6d0e433ddc8a"


def training_prompt(path: Path) -> str:
    prompts = {next(m["content"] for m in row["messages"] if m["role"] == "system")
               for row in map(json.loads, path.read_text().splitlines())}
    if len(prompts) != 1:
        raise ValueError("diagnostic requires one pinned v5 training prompt")
    prompt = prompts.pop()
    if hashlib.sha256(prompt.encode()).hexdigest() != TRAINING_PROMPT_SHA256:
        raise ValueError("v5 training prompt changed")
    return prompt


def _feedback(output: str, step: int, total: int) -> str:
    # Match the legacy environment's step feedback, with this run's smaller cap.
    remaining = total - step
    if remaining <= 1:
        return output + ('\n\n*** FINAL STEP — you MUST respond with: '
                         'SUBMIT: <answer> CITATIONS: ["doc_id"] ***')
    if remaining <= 5:
        return output + (f"\n\n[Step {step}/{total} — {remaining} steps left. "
                         "Run sufficiency check: can you answer now? If yes → SUBMIT immediately.]")
    return output + f"\n\n[Step {step}/{total}/{total}]"


def run_trial(policy, snapshot: Path, question: str, doc_id: str, *, image: str,
              max_steps: int = 6, repl_factory=None) -> dict:
    if type(max_steps) is not int or not 2 <= max_steps <= 6:
        raise ValueError("diagnostic supports 2..6 steps")
    manifest, docs = load_snapshot(snapshot)
    if len(docs) != 1 or doc_id not in docs:
        raise ValueError("diagnostic needs exactly its named paper")
    result = {"status": "incomplete", "question": question, "doc_id": doc_id,
              "corpus_hash": manifest["corpus_hash"], "max_steps": max_steps,
              "trajectory": [], "answer": None, "citations": [], "evidence": [],
              "source_inspection_actions": 0, "model_requests_attempted": 0,
              "execution": "docker_only" if repl_factory is None else "injected_test_repl",
              "semantic_support": "not_reviewed", "reward": "not_computed"}
    started = time.monotonic()
    repl = None
    try:
        if repl_factory is None:
            if not PersistentREPL._docker_available(image):
                raise RuntimeError("pinned Docker sandbox is unavailable")
            repl = PersistentREPL(use_docker=True, image=image,
                                  corpus_path=str(snapshot / "corpus"))
        else:
            repl = repl_factory(snapshot / "corpus")
        repl.start_session()
        policy.reset()
        observation = f"Question: {question}\n"
        for step in range(1, max_steps + 1):
            result["model_requests_attempted"] += 1
            action = policy.act(observation)
            trace = {"step": step, "action": action}
            result["trajectory"].append(trace)
            if not isinstance(action, str) or not action or len(action) > 8000:
                raise ValueError("empty, non-text or oversized action")
            parsed = parse_submission_details(action)
            if parsed is not None:
                answer, citations, evidence = parsed
                result.update(status="submitted", answer=answer,
                              citations=citations, evidence=evidence)
                trace["observation"] = "Submitted under the original citation-only protocol."
                break
            output = repl.execute(action, timeout=20)
            if not isinstance(output, str):
                raise ValueError("sandbox returned a non-text observation")
            execution_error = "Traceback" in output or "SyntaxError" in output
            inspected = not execution_error and _used_inspection_tool(action, output)
            result["source_inspection_actions"] += int(inspected)
            trace.update(observation=output, inspection_action=inspected,
                         execution_error=execution_error)
            observation = _feedback(output, step, max_steps)
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__, error=str(exc)[:500])
    finally:
        if repl is not None:
            try:
                repl.kill_session()
            except Exception:
                pass
        result["elapsed_seconds"] = time.monotonic() - started
    return result
