"""Reviewed demonstrations through the frozen Docker reader; no model inference."""
from __future__ import annotations

import copy
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone

from src.env.repl import PersistentREPL
from src.eval.passage_pilot_data import ROOT, check_prepared, reviewed_rejections
from src.eval.qasper_reward import parse_strict
from src.eval.reader_correction_replay import cleanup, track
from src.eval.reader_improvement import digest, file_hash, inside, load, write
from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.product.harness_v2 import clean_v2_action
from src.product.packet_verifier import MAX_SPAN_CHARS, verify_investigation_packet
from src.product.paper_assignment import assignment_observation, paper_assignment
from src.product.passage_scope import PASSAGES_V2, contains
from src.product.qwen_investigator import QwenInvestigator, effective_system_prompt
from src.product.reader_gaps import parse_gap_submission
from src.research.agent import load_snapshot

BASELINE = ROOT / "examples/qasper-hybrid-v5/frozen/plan.json"
PROMPT = effective_system_prompt("public_papers", "default", "v2", reader_mode="assigned",
                                 reader_guidance=PASSAGES_V2)
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()
IDENTITY = {"data_kind": "scripted_reviewed_demonstration", "model_inference": False,
            "author": "implementing assistant; annotation-assisted, source-reviewed"}
SOURCES = ("src/eval/passage_pilot_data.py", "src/eval/passage_pilot_replay.py",
           "scripts/prepare_passage_pilot.py", "scripts/replay_passage_pilot.py",
           "scripts/train_sft.py", "src/policies/openai_compatible.py",
           "src/eval/reader_correction_replay.py")


def frozen_runtime():
    plan = load(BASELINE)
    hashes = plan["implementation_sha256"] | plan["additional_implementation_sha256"]
    if any(file_hash(inside(ROOT, p)) != sha for p, sha in hashes.items()):
        raise ValueError("frozen runtime changed; pilot data preparation is blocked")
    return plan


def check_review(prepared, review):
    manifest, cases = check_prepared(prepared)
    if (review.get("cases_sha256") != manifest["cases_sha256"]
            or review.get("reviewer_kind") not in {"human", "model_assisted"}
            or not review.get("reviewer_identity") or not review.get("reviewed_at")):
        raise ValueError("explicit case-bound source review attribution required")
    rows = review.get("rows", [])
    if (len(rows) != len(cases) or {r.get("case_id") for r in rows}
            != {c["case_id"] for c in cases}):
        raise ValueError("review must cover all cases exactly once")
    by_id = {r["case_id"]: r for r in rows}
    for case in cases:
        row = by_id[case["case_id"]]
        if (row.get("case_sha256") != digest(case) or row.get("decision") != "accept"
                or row.get("source_support_reviewed") is not True or not row.get("notes")
                or not row.get("requested_fact_review")):
            raise ValueError("accepted source review and requested-fact notes required")
        spans = row.get("supporting_spans", [])
        doc = load(inside(prepared, case["snapshot"] + "/corpus/" + case["doc_id"] + ".json"))
        if not isinstance(spans, list) or not 1 <= len(spans) <= 5:
            raise ValueError("bounded source-review excerpts required")
        for s in spans:
            if (set(s) != {"doc_id", "start", "end", "quote"} or s["doc_id"] != case["doc_id"]
                    or not contains(case["scope"]["windows"], s["start"], s["end"])
                    or s["end"] - s["start"] > MAX_SPAN_CHARS
                    or s["quote"] != doc["text"][s["start"]:s["end"]]):
                raise ValueError("review span is not an exact bounded in-scope excerpt")
        action = row.get("final_action")
        if not isinstance(action, str) or len(action) > 8000:
            raise ValueError("bounded literal terminal action required")
        if case["scope_answerability"] == "present":
            parsed = parse_strict(action)
            expected = [{k: s[k] for k in ("doc_id", "start", "end")} for s in spans]
            if (parsed is None or "\n" in action or parsed[0].casefold() == "unanswerable"
                    or parsed[1] != [case["doc_id"]] or parsed[2] != expected):
                raise ValueError("present case requires supported SUBMIT with reviewed spans")
        elif parse_gap_submission(action) is None:
            raise ValueError("missing case requires a lone literal terminal submit_gap")
    return cases, by_id


class DemonstrationTransport:
    """Capture real formatter payloads, returning a bounded local action sequence."""

    def __init__(self, actions):
        self.actions, self.payloads = list(actions), []

    def __call__(self, url, payload, headers, timeout):
        index = len(self.payloads)
        if index >= len(self.actions) or payload["messages"][0] != {
                "role": "system", "content": PROMPT}:
            raise ValueError("demonstration exceeds fixed actions or changes effective prompt")
        self.payloads.append(copy.deepcopy(payload))
        return {"model": payload["model"], "choices": [{"message": {
            "content": self.actions[index]}, "finish_reason": "stop"}]}


def verify_case(prepared, case, row, packet, payloads):
    snapshot = inside(prepared, case["snapshot"])
    manifest, docs = load_snapshot(snapshot)
    verified = verify_investigation_packet(packet, manifest, docs,
        allowed_prompt_sha256s=frozenset({PROMPT_SHA}), expected_reader_mode="assigned",
        expected_reader_guidance=PASSAGES_V2, expected_passage_windows=case["scope"]["windows"])
    actions = case["scope"]["read_calls"] + [row["final_action"]]
    trace = packet["trajectory"]
    if (packet["execution"] != "docker_only" or packet["inspection_steps"] != 2
            or packet["model_identity"] != IDENTITY or packet["question"] != case["question"]
            or packet["model_requests_attempted"] != 3 or len(payloads) != 3 or len(trace) != 3
            or [t["action"] for t in trace] != actions
            or any(t.get("source_inspection") is not True for t in trace[:2])
            or any(t.get("format_retries") or t.get("error") for t in trace)):
        raise ValueError("demonstration did not complete two Docker reads and reviewed terminal")
    expected_status = ("evidence_found" if case["scope_answerability"] == "present"
                       else "no_evidence")
    if packet["status"] != expected_status:
        raise ValueError("demonstration terminal status differs from reviewed target")
    if expected_status == "evidence_found":
        spans = [{k: s[k] for k in ("doc_id", "start", "end", "quote")}
                 for s in packet["evidence"]]
        if (spans != row["supporting_spans"]
                or verified["candidate_answer"] != parse_strict(row["final_action"])[0]):
            raise ValueError("reviewed answer/evidence changed in replay")
    else:
        gaps = verified["research_gaps"]
        fields = parse_gap_submission(row["final_action"])
        if (len(gaps) != 1 or any(gaps[0][k] != v for k, v in fields.items())
                or gaps[0]["absence_proven"] is not False or verified["candidate_answer"]):
            raise ValueError("reviewed gap lost its scoped-uncertainty semantics")
    # Replay the policy formatter only, not Python, to prove every saved prefix
    # and every next observation equals the actual recorded runtime feedback.
    initial = (f"{PROMPT}\nQuestion: {case['question']}\n" +
               assignment_observation(paper_assignment(manifest, docs)) +
               "\nController passage ranges (metadata only): " + json.dumps(case["scope"]))
    if payloads[0]["messages"] != [{"role": "system", "content": PROMPT},
                                    {"role": "user", "content": initial}]:
        raise ValueError("initial request differs from case question, source assignment or scope")
    captured = DemonstrationTransport(actions)
    policy = make_policy(captured)
    for i, (payload, action) in enumerate(zip(payloads, actions)):
        observation = (payload["messages"][-1]["content"] if i == 0
                       else trace[i - 1]["observation"])
        if policy.act(observation) != action or captured.payloads[-1] != payload:
            raise ValueError("saved conversation differs from real formatter/observation prefix")
    return verified


def make_policy(transport):
    return OpenAICompatiblePolicy(endpoint="http://127.0.0.1/offline-demonstration",
        api_key="offline-no-credential", model="envoy", max_tokens=1024, temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_p": 1.0, "seed": 42},
        transport=transport, system_prompt=PROMPT, action_cleaner=clean_v2_action)


def replay_case(prepared, case, row, image, output, containers):
    output.mkdir(parents=True)
    transport = DemonstrationTransport(case["scope"]["read_calls"] + [row["final_action"]])
    reader = QwenInvestigator(lambda: make_policy(transport), IDENTITY, max_steps=4, image=image,
        source_domain="public_papers", harness_version="v2", reader_mode="assigned",
        reader_guidance=PASSAGES_V2)
    if reader.system_prompt != PROMPT:
        raise ValueError("policy and investigator effective prompts differ")
    track(reader, output, containers)
    packet = reader.investigate(case["question"], inside(prepared, case["snapshot"]),
                                passage_windows=case["scope"]["windows"])
    write(output / "packet.json", packet)
    write(output / "payloads.json", transport.payloads)
    verified = verify_case(prepared, case, row, packet, transport.payloads)
    write(output / "verified.json", verified)
    receipt = {"case_id": case["case_id"], "role": case["role"], "passed": True,
               "status": packet["status"], "source_inspections": 2, "local_policy_turns": 3,
               "new_model_calls": 0, "review_origin": "see bound source review",
               "semantic_support": "reviewer_attestation_not_mechanically_proven",
               "actions": [{"action": t["action"], "source_inspection": t.get(
                   "source_inspection", False), "observation_sha256": hashlib.sha256(
                       t["observation"].encode()).hexdigest()} for t in packet["trajectory"]]}
    write(output / "receipt.json", receipt)
    return receipt


def outside_inputs(output, *inputs):
    if any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(
            output.resolve()) for p in inputs):
        raise ValueError("output must be separate from frozen inputs")


def replay(prepared, review_path, output):
    outside_inputs(output, prepared)
    plan = frozen_runtime()
    review = load(review_path)
    cases, rows = check_review(prepared, review)
    rejected = reviewed_rejections(load(prepared / "selection.json")["data_review_rejections"])
    if {c["doc_id"] for c in cases} & {r["doc_id"] for r in rejected}:
        raise ValueError("source-review rejected paper reappeared")
    image = plan["sandbox_image"]
    if not image.startswith("sha256:") or not PersistentREPL._docker_available(image):
        raise ValueError("pinned Docker image unavailable; no host fallback")
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "data_kind": "reviewed_fixed_action_docker_demonstration",
               "new_model_calls": 0, "training_performed": False, "human_review": None,
               "started_at": datetime.now(timezone.utc).isoformat(),
               "preparation_hardware": {"platform": platform.platform(),
                                        "machine": platform.machine(),
                                        "python": platform.python_version()},
               "cases": [], "limitation": "Fixed actions do not measure Qwen choosing actions, "
               "responding to feedback, or improving. Quotes establish provenance; source support "
               "is a separately attributed review, not a mechanical proof."}
    containers = []
    try:
        for case in cases:
            summary["cases"].append(replay_case(prepared, case, rows[case["case_id"]], image,
                output / "cases" / case["case_id"], containers))
        summary["status"] = "passed_demonstration_replay"
    except Exception as exc:
        summary.update(status="failed", error_kind=type(exc).__name__, error=str(exc))
        raise
    finally:
        proofs = cleanup(containers)
        closed = len(proofs) == len(cases) and all(p["absent"] for p in proofs)
        write(output / "cleanup.json", {"containers": proofs, "all_absent": closed})
        if not closed:
            summary["status"] = "failed"
        write(output / "summary.json", summary)
        write(output / "manifest.json", {"schema_version": "passage-pilot-replay-v1",
            "prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
            "review_sha256": file_hash(review_path), "prompt_sha256": PROMPT_SHA,
            "sandbox_image": image, "frozen_runtime": plan["implementation_sha256"] |
                plan["additional_implementation_sha256"],
            "implementation_sha256": {p: file_hash(ROOT / p) for p in SOURCES},
            "artifact_sha256": {str(p.relative_to(output)): file_hash(p)
                                for p in sorted(output.rglob("*.json"))}})
    if summary["status"] != "passed_demonstration_replay":
        raise ValueError("Docker cleanup did not verify every owned container absent")
    return summary


def export(prepared, review_path, replay_dir, output):
    """Train-only export, requiring source review, replay and exact token prefixes."""
    outside_inputs(output, prepared, replay_dir)
    frozen_runtime()
    review = load(review_path)
    cases, rows = check_review(prepared, review)
    manifest = load(replay_dir / "manifest.json")
    if (manifest["schema_version"] != "passage-pilot-replay-v1"
            or manifest["prepared_manifest_sha256"] != file_hash(prepared / "manifest.json")
            or manifest["review_sha256"] != file_hash(review_path)
            or manifest["prompt_sha256"] != PROMPT_SHA
            or any(file_hash(inside(ROOT, p)) != sha for p, sha in manifest[
                "implementation_sha256"].items())):
        raise ValueError("replay inputs or implementation changed")
    required = {"summary.json", "cleanup.json"} | {f"cases/{c['case_id']}/{name}.json"
        for c in cases for name in ("packet", "payloads", "verified", "receipt")}
    hashes = manifest["artifact_sha256"]
    if not required <= hashes.keys() or any(file_hash(inside(replay_dir, p)) != sha
                                           for p, sha in hashes.items()):
        raise ValueError("incomplete or changed replay artifacts")
    summary, cleanup_proof = load(replay_dir / "summary.json"), load(replay_dir / "cleanup.json")
    if (summary["status"] != "passed_demonstration_replay" or summary["new_model_calls"] != 0
            or not cleanup_proof["all_absent"] or len(cleanup_proof["containers"]) != 36
            or not all(p["absent"] for p in cleanup_proof["containers"])
            or len({p["id"] for p in cleanup_proof["containers"]}) != 36
            or {c["case_id"] for c in summary["cases"]} != {c["case_id"] for c in cases}):
        raise ValueError("incomplete replay or owned-container cleanup")
    train, train_cases = [], []
    for case in cases:
        base = replay_dir / "cases" / case["case_id"]
        packet, payloads = load(base / "packet.json"), load(base / "payloads.json")
        verified = verify_case(prepared, case, rows[case["case_id"]], packet, payloads)
        if verified != load(base / "verified.json"):
            raise ValueError("saved verifier result differs")
        if case["role"] == "train":
            train.append({"messages": payloads[-1]["messages"] + [{"role": "assistant",
                           "content": rows[case["case_id"]]["final_action"]}]})
            train_cases.append(case)
    from scripts.train_sft import (
        BASE_MODEL,
        BASE_REVISION,
        _expand_per_action,
        _tokenize_per_action,
    )
    from src.product.experiment_tokenizer import load_tokenizer
    tokenizer = load_tokenizer(BASE_MODEL, BASE_REVISION)
    examples = _expand_per_action(train)
    tokenized, stats = _tokenize_per_action(tokenizer, examples, 8192, "passage-pilot-train")
    allocation = Counter()
    for case, group in zip(train_cases, [tokenized[i:i + 3] for i in range(0, len(tokenized), 3)]):
        for index, item in enumerate(group):
            key = case["scope_answerability"] + ("_read" if index < 2 else "_terminal")
            allocation[key] += sum(x != -100 for x in item["labels"])
    if len(train) != 24 or len(examples) != 72:
        raise ValueError("expected only 24 train trajectories and 72 next actions")
    output.mkdir(parents=True, exist_ok=False)
    (output / "train.jsonl").write_text("".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in train))
    development = [{**c, "review": rows[c["case_id"]], "usage": "scoring_only_never_training"}
                   for c in cases if c["role"] == "development"]
    write(output / "development-scoring-only.json", development)
    result = {"schema_version": "passage-pilot-export-v1", "training_ready_cases": 24,
        "development_scoring_only": 12, "human_review": None, "new_model_calls": 0,
        "training_performed": False, "tokenizer": BASE_MODEL, "tokenizer_revision": BASE_REVISION,
        "token_checks": stats, "supervised_token_allocation": dict(allocation),
        "train_case_ids": [c["case_id"] for c in train_cases],
        "development_case_ids": [c["case_id"] for c in development],
        "prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
        "replay_manifest_sha256": file_hash(replay_dir / "manifest.json"),
        "review_sha256": file_hash(review_path), "effective_prompt_sha256": PROMPT_SHA,
        "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
        "data_origin": "Official QASPER annotations plus implementing-assistant source review; "
        "not paid teacher inference, independent human review, or private reasoning traces.",
        "initialization_if_later_authorized": load(BASELINE)["model_identity"],
        "proposed_training_not_executed": {"continue_separate_v5_copy": True,
            "rank": 4, "alpha": 8, "dropout": 0.05, "learning_rate": 2e-4,
            "microbatch": 1, "accumulation": 4, "sequence_limit": 8192, "seed": 42,
            "epochs": 1, "maximum_optimizer_updates": 24},
        "authorization": "Data preparation only; zero GPU/training/inference allowance."}
    write(output / "manifest.json", result)
    return result
