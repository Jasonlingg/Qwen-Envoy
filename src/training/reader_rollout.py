"""Run the frozen investigator in Docker; Verifiers owns generation and traces."""

from pathlib import Path

from src.eval.passage_pilot_replay import PROMPT, PROMPT_SHA, frozen_runtime
from src.eval.reader_correction_replay import cleanup, track
from src.eval.reader_improvement import write
from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.product.harness_v2 import clean_v2_action
from src.product.packet_verifier import verify_investigation_packet
from src.product.passage_scope import PASSAGES_V2
from src.product.qwen_investigator import QwenInvestigator
from src.research.agent import load_snapshot

TEMPERATURE = 1.0
MAX_ACTION_TOKENS = 1024


def make_policy(transport):
    return OpenAICompatiblePolicy(
        endpoint="http://127.0.0.1/local-rl-no-http",
        api_key="offline-no-credential",
        model="envoy",
        transport=transport,
        system_prompt=PROMPT,
        action_cleaner=clean_v2_action,
        temperature=TEMPERATURE,
        max_tokens=MAX_ACTION_TOKENS,
        extra_body={"top_p": 1.0, "chat_template_kwargs": {"enable_thinking": False}},
    )


def collect(case, snapshot, transport, output: Path, model_identity, *, max_steps=4):
    """Do not accept repl injection here: every real action uses the frozen Docker path."""
    plan = frozen_runtime()
    output.mkdir(parents=True, exist_ok=False)
    reader = QwenInvestigator(
        lambda: make_policy(transport),
        model_identity,
        max_steps=max_steps,
        image=plan["sandbox_image"],
        source_domain="public_papers",
        harness_version="v2",
        reader_mode="assigned",
        reader_guidance=PASSAGES_V2,
    )
    containers = []
    track(reader, output, containers)
    try:
        packet = reader.investigate(
            case["question"], snapshot, passage_windows=case["scope"]["windows"]
        )
    finally:
        proofs = cleanup(containers)
        clean = bool(proofs) and all(p["absent"] for p in proofs)
        write(output / "cleanup.json", {"containers": proofs, "all_absent": clean})
    manifest, docs = load_snapshot(snapshot)
    verified = None
    if packet["status"] not in {"error", "unavailable"}:
        verified = verify_investigation_packet(
            packet,
            manifest,
            docs,
            allowed_prompt_sha256s=frozenset({PROMPT_SHA}),
            expected_reader_mode="assigned",
            expected_reader_guidance=PASSAGES_V2,
            expected_passage_windows=case["scope"]["windows"],
        )
    infrastructure_ok = clean and packet["status"] not in {"error", "unavailable"}
    result = {
        "case_id": case["case_id"],
        "question_id": case["question_id"],
        "packet": packet,
        "verified": verified,
        "infrastructure_ok": infrastructure_ok,
        "training_eligible": False,
    }
    write(output / "rollout.json", result)
    write(output / "payloads.json", transport.payloads)
    return result
