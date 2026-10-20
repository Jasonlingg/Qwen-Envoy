"""Offline verification of a frozen assigned-reader smoke preparation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.eval.assigned_smoke_candidate import candidate_contract
from src.eval.assigned_smoke_plan import (
    CASES,
    LIMITS,
    MODEL_IDENTITY,
    ROOT,
    SOURCE_FILES,
    digest,
    initial_payload,
    source_digest,
)
from src.product.assigned_guidance import BASELINE, FOCUSED, validate_guidance
from src.product.paper_assignment import assignment_hash, paper_assignment
from src.product.qwen_investigator import effective_system_prompt
from src.research.agent import load_snapshot


def check(output: Path, *, tokenizer) -> dict:
    raw = (output / "plan.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != (output / "plan.sha256").read_text().strip():
        raise ValueError("prepared plan changed")
    plan = json.loads(raw)
    guidance = plan.get("reader_guidance", BASELINE)
    validate_guidance(guidance, "assigned")
    if guidance == FOCUSED and any(plan.get(k) != v for k, v in candidate_contract().items()):
        raise ValueError("candidate execution contract changed")
    if guidance == BASELINE and plan.get("execution_path") is not None:
        raise ValueError("unknown baseline execution path")
    if (plan.get("schema_version") != "assigned-reader-smoke-plan-v1"
            or plan.get("limits") != LIMITS or plan.get("cases") != CASES
            or plan.get("activation", {}).get("approved") is not False
            or plan.get("status") != "prepared_not_run" or plan.get("model_calls") != 0):
        raise ValueError("unknown preparation contract or altered limits/approval")
    if any(plan.get("model", {}).get(k) != v for k, v in MODEL_IDENTITY.items()):
        raise ValueError("prepared model or decoding pin changed")
    if set(plan["implementation_sha256"]) != set(SOURCE_FILES):
        raise ValueError("preparation implementation file set changed")
    for name, expected in plan["implementation_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"implementation changed after prepare: {name}")
    artifacts = {}
    for name in ["model_inputs.json", "references.json", "claim-review-template.json"]:
        artifacts[name] = json.loads((output / name).read_text())
        if digest(artifacts[name]) != plan["artifact_sha256"].get(name):
            raise ValueError(f"prepared artifact changed: {name}")
    prompt = (output / "system-prompt.txt").read_text()
    if (hashlib.sha256(prompt.encode()).hexdigest() != plan["system_prompt_sha256"]
            or prompt != effective_system_prompt("public_papers", "default", "v2",
                reader_mode="assigned", reader_guidance=guidance)):
        raise ValueError("prepared effective prompt changed")
    manifest, docs = load_snapshot(Path(plan["snapshot"]))
    if manifest["corpus_hash"] != plan["corpus_hash"]:
        raise ValueError("prepared corpus changed")
    assignment = paper_assignment(manifest, docs)
    if (assignment != plan["paper_assignment"]
            or assignment_hash(assignment) != plan["assignment_sha256"]
            or source_digest(next(iter(docs.values()))) != plan["source_sha256"]):
        raise ValueError("prepared source or assignment changed")
    inputs = artifacts["model_inputs.json"]
    if not isinstance(inputs, list) or len(inputs) != len(CASES):
        raise ValueError("prepared input count changed")
    token_counts = []
    for case, item in zip(CASES, inputs):
        payload = initial_payload(prompt, assignment, case["question"])
        if item.get("case_id") != case["case_id"] or item.get("payload") != payload:
            raise ValueError("model input differs from the actual reader initial payload")
        tokens = tokenizer.apply_chat_template(payload["messages"], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if (not isinstance(tokens, list) or any(type(t) is not int for t in tokens)
                or len(tokens) > LIMITS["input_tokens_per_request"]
                or item.get("initial_prompt_tokens") != len(tokens)):
            raise ValueError("prepared input token count changed or exceeds the cap")
        token_counts.append(len(tokens))
    context = artifacts["references.json"]["source_context"]
    source = next(iter(docs.values()))["text"]
    if (type(context["start"]) is not int or type(context["end"]) is not int
            or not 0 <= context["start"] < context["end"] <= len(source)
            or context["text"] != source[context["start"]:context["end"]]):
        raise ValueError("reference context is not exact cached source text")
    return {"status": "prepared_not_run", "cases": len(CASES), "model_calls": 0,
            "initial_prompt_tokens": token_counts, "limits": LIMITS,
            "reference_review": "assistant_prepared; human review not performed",
            "activation_approved": False, "reader_guidance": guidance,
            "execution_path": plan.get("execution_path", "direct_reader")}
