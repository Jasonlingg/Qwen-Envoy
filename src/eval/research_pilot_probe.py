"""A real Docker boundary check with scripted actions, never a Qwen answer."""

from __future__ import annotations

import json

from src.product.packet_verifier import verify_investigation_packet
from src.product.qwen_investigator import QwenInvestigator
from src.research.agent import load_snapshot


def probe_sandbox(library, version, image):
    document = library.document(version)
    doc_id, end = document["doc_id"], min(100, len(document["text"]))
    actions = [
        'print(search("executable code actions", top_k=1))',
        f"print(passage({doc_id!r}, start=0, length=300))",
        "SUBMIT: Scripted source inspection probe. CITATIONS: " + json.dumps([doc_id])
        + " EVIDENCE: " + json.dumps([{"doc_id": doc_id, "start": 0, "end": end}]),
    ]
    class ScriptedPolicy:
        def __init__(self):
            self.actions = iter(actions)
        def act(self, observation):
            return next(self.actions)
    investigator = QwenInvestigator(policy_factory=ScriptedPolicy,
        model_identity={"backend": "scripted_probe", "model": "none"}, image=image,
        source_domain="public_papers", harness_version="v2", max_steps=4)
    view = library.view([version])
    packet = investigator.investigate("Inspect a source excerpt for a plumbing check.", view)
    manifest, docs = load_snapshot(view)
    import hashlib
    verified = verify_investigation_packet(packet, manifest, docs,
        allowed_prompt_sha256s=frozenset({hashlib.sha256(
            investigator.system_prompt.encode()).hexdigest()}))
    return {"execution_kind": "scripted_docker_probe", "actual_model_requests": 0,
            "semantic_support": "not_reviewed", "docker_image_digest": image,
            "status": verified["status"], "scripted_actions": len(packet["trajectory"]),
            "exact_evidence_spans": len(verified["evidence"]),
            "warning": "Scripted integration check; no live Qwen or answer-quality result.",
            "packet": packet}
