"""Fixed-action Docker replay; never inference, training, or a semantic reward."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

from src.env.repl import PersistentREPL
from src.eval.reader_improvement import (
    build_queue,
    check_review,
    digest,
    file_hash,
    inside,
    load,
    write,
)
from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.product.harness_v2 import clean_v2_action
from src.product.packet_verifier import verify_investigation_packet
from src.product.qwen_investigator import QwenInvestigator
from src.product.reader_gaps import parse_gap_submission
from src.research.agent import load_snapshot

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "reader-correction-replay-v1"


class RecordedTransport:
    """Compare actual policy inputs with originals; return only fixed local replies."""

    def __init__(self, requests, responses, *, replacement=None):
        self.requests = requests
        self.responses = responses
        self.replacement = replacement
        self.receipts = []

    def __call__(self, url, payload, headers, timeout):
        index = len(self.receipts)
        if index >= len(self.requests) or payload != self.requests[index]:
            raise ValueError("replay request differs from historical model input")
        response = copy.deepcopy(self.responses[index])
        replaced = self.replacement is not None and index == len(self.requests) - 1
        if replaced:
            # Do not attach historical usage/logprobs to a reviewer-authored reply.
            response = {"model": payload["model"], "choices": [{"message": {
                "content": self.replacement}, "finish_reason": "stop"}]}
        self.receipts.append({"step": index + 1, "payload_sha256": digest(payload),
                              "historical_payload_matched": True,
                              "reply_origin": "reviewed_correction" if replaced else "recorded",
                              "content": response["choices"][0]["message"]["content"]})
        return response


def validate_inputs(bundle, queue_dir, review_path, drafts_path):
    queue = load(queue_dir / "queue.json")
    if queue != build_queue(bundle):
        raise ValueError("queue differs from frozen original run")
    drafts = check_review(queue, load(review_path))
    if drafts != load(drafts_path):
        raise ValueError("drafts differ from attributed, source-bound review")
    plan = queue["baseline"]["plan"]
    frozen = plan["implementation_sha256"] | plan.get("additional_implementation_sha256", {})
    if any(file_hash(inside(ROOT, p)) != sha for p, sha in frozen.items()):
        raise ValueError("frozen runtime implementation changed")
    records, candidates = queue["records"], drafts["correction_drafts"]
    if not records or not candidates or len(records) + len(candidates) > 5:
        raise ValueError("one to five bounded episodes with at least one correction required")
    if any(r["original_packet"]["max_steps"] != 4 for r in records):
        raise ValueError("this replay requires the original four-step action limit")
    return queue, drafts


def request_groups(bundle, records):
    ledger = [json.loads(line) for line in (bundle / "session.requests.jsonl").read_text()
              .splitlines() if line.strip()]
    requests = [r for r in ledger if r.get("event") == "reserved"]
    responses = {r["request"]: r["response"] for r in ledger if r.get("event") == "completed"}
    offset, groups = 0, {}
    for record in records:
        count = record["original_packet"]["model_requests_attempted"]
        subset = requests[offset:offset + count]
        groups[record["case_id"]] = ([r["payload"] for r in subset],
                                      [responses[r["request"]] for r in subset])
        offset += count
    return groups


def inspect_container(container):
    result = subprocess.run(["docker", "inspect", container], capture_output=True,
                            text=True, timeout=15)
    if result.returncode:
        if any(s in result.stderr.lower() for s in ("no such object", "no such container")):
            return None
        raise RuntimeError("could not verify owned Docker container state")
    value = json.loads(result.stdout)[0]
    return {"id": value["Id"], "image": value["Image"],
            "network_mode": value["HostConfig"]["NetworkMode"],
            "mounts": [{k: m[k] for k in ("Source", "Destination", "RW")}
                       for m in value["Mounts"]]}


def track(reader, output, containers):
    original = reader._repl

    def create(corpus):
        repl = original(corpus)
        start = repl.start_session

        def tracked_start():
            try:
                start()
            finally:
                container = getattr(repl._impl, "_container_id", None)
                if container:
                    containers.append(container)
                    write(output / "owned-containers.json", containers)
            proof = inspect_container(container) if container else None
            if (proof is None or proof["image"] != reader.image
                    or proof["network_mode"] != "none"
                    or not any(m["Source"] == str(corpus.resolve()) and m["RW"] is False
                               for m in proof["mounts"])):
                raise ValueError("replay sandbox image, network or read-only corpus differs")
            write(output / f"{container}.sandbox.json", proof)
        repl.start_session = tracked_start
        return repl
    reader._repl = create


def cleanup(containers):
    proofs = []
    for container in containers:
        try:
            if inspect_container(container) is not None:
                subprocess.run(["docker", "rm", "-f", container], capture_output=True,
                               text=True, timeout=15, check=True)
            proofs.append({"id": container, "absent": inspect_container(container) is None})
        except Exception as exc:
            # Attempt every owned container even if one removal/read fails.
            proofs.append({"id": container, "absent": False,
                           "error_kind": type(exc).__name__})
    return proofs


def original_matches(packet, original):
    # UUID is the sole nondeterministic packet field in this adapter.
    def comparable(value):
        return {k: v for k, v in value.items() if k != "retriever_run_id"}
    return comparable(packet) == comparable(original)


def replay_one(bundle, record, group, image, output, containers, *, candidate=None):
    original = record["original_packet"]
    requests, responses = group
    replacement = candidate["proposed_final_action"] if candidate else None
    transport = RecordedTransport(requests, responses, replacement=replacement)
    first = requests[0]
    extra = {k: v for k, v in first.items()
             if k not in {"messages", "model", "max_tokens", "temperature"}}
    prompt = first["messages"][0]["content"]

    def policy():
        return OpenAICompatiblePolicy(endpoint="http://127.0.0.1/offline-replay",
            api_key="offline-no-credential", model=first["model"],
            max_tokens=first["max_tokens"], temperature=first["temperature"], extra_body=extra,
            transport=transport, system_prompt=prompt, action_cleaner=clean_v2_action)

    reader = QwenInvestigator(policy, original["model_identity"], max_steps=4, image=image,
        source_domain=original["source_domain"], protocol_variant=original["protocol_variant"],
        harness_version=original["harness_version"], reader_mode="assigned",
        reader_guidance=original["reader_guidance"])
    if reader.system_prompt != prompt:
        raise ValueError("effective replay prompt differs from original policy")
    track(reader, output, containers)
    cases = {c["case_id"]: c for c in load(bundle / "frozen/cases.json")}
    case = cases[record["case_id"]]
    snapshot = inside(bundle, "frozen/" + case["snapshot"])
    packet = reader.investigate(record["question"], snapshot, passage_windows=case["windows"])
    write(output / "packet.json", packet)
    write(output / "requests.json", {"data_kind": "fixed_local_replies_no_inference",
                                     "receipts": transport.receipts,
                                     "historical_usage_is_not_new_inference_usage": True})
    manifest, docs = load_snapshot(snapshot)
    verified = verify_investigation_packet(packet, manifest, docs,
        allowed_prompt_sha256s=frozenset({hashlib.sha256(prompt.encode()).hexdigest()}),
        expected_reader_mode="assigned", expected_reader_guidance=original["reader_guidance"],
        expected_passage_windows=case["windows"])
    write(output / "verified.json", verified)
    if (packet["execution"] != "docker_only" or len(transport.receipts) != len(requests)
            or len(packet["trajectory"]) != len(original["trajectory"])):
        raise ValueError("replay did not execute the complete bounded Docker trajectory")
    if candidate is None:
        if not original_matches(packet, original):
            raise ValueError("original replay differs; corrections are blocked")
    else:
        if (packet["trajectory"][:-1] != original["trajectory"][:-1]
                or packet["trajectory"][-1]["action"] != candidate["proposed_final_action"]):
            raise ValueError("correction changed the original source-inspection prefix")
        if candidate["kind"] == "answer":
            expected = [{k: s[k] for k in ("doc_id", "start", "end", "quote")}
                        for s in candidate["supporting_spans"]]
            actual = [{k: s[k] for k in ("doc_id", "start", "end", "quote")}
                      for s in packet["evidence"]]
            if packet["status"] != "evidence_found" or actual != expected:
                raise ValueError("corrected answer did not retain reviewed exact evidence")
        else:
            fields = parse_gap_submission(candidate["proposed_final_action"])
            gaps = verified.get("research_gaps", [])
            if (packet["status"] != "no_evidence" or len(gaps) != 1
                    or any(gaps[0][k] != v for k, v in fields.items())
                    or gaps[0]["absence_proven"] is not False or verified["candidate_answer"]):
                raise ValueError("corrected terminal gap was not retained as scoped uncertainty")
    receipt = {"case_id": record["case_id"], "kind": candidate["kind"] if candidate else "original",
               "status": packet["status"], "passed": True,
               "source_inspection_steps": packet["inspection_steps"],
               "request_payloads_matched": len(transport.receipts),
               "original_packet_matched_except_uuid": candidate is None,
               "source_prefix_unchanged": True, "semantic_support": "not_mechanically_reviewed",
               "training_eligible": False, "training_ready": False}
    receipt["actions"] = [{
        "step": trace["step"], "action": trace["action"],
        "source_inspection": trace.get("source_inspection", False),
        "observation_sha256": hashlib.sha256(trace["observation"].encode()).hexdigest(),
        "original_action_matched": trace["action"] == previous["action"],
        "original_observation_matched": trace["observation"] == previous["observation"],
        "reply_origin": transport.receipts[index]["reply_origin"],
    } for index, (trace, previous) in enumerate(zip(packet["trajectory"], original["trajectory"]))]
    write(output / "receipt.json", receipt)
    return receipt


def replay(bundle, queue_dir, review_path, drafts_path, output):
    queue, drafts = validate_inputs(bundle, queue_dir, review_path, drafts_path)
    if any(output.resolve().is_relative_to(p.resolve()) for p in (bundle, queue_dir)):
        raise ValueError("replay output must be outside frozen inputs")
    image = queue["baseline"]["plan"]["sandbox_image"]
    if not image.startswith("sha256:") or not PersistentREPL._docker_available(image):
        raise ValueError("pinned Docker image unavailable; no host fallback")
    output.mkdir(parents=True, exist_ok=False)
    records = queue["records"]
    by_id = {r["case_id"]: r for r in records}
    groups = request_groups(bundle, records)
    containers = []
    summary = {"schema_version": SCHEMA, "status": "running", "originals": [], "corrections": [],
               "queue_sha256": digest(queue), "drafts_sha256": digest(drafts),
               "data_kind": "fixed_action_docker_replay", "new_model_calls": 0,
               "human_review": None, "independent_human_review": None, "training_ready": 0,
               "limitation": "Fixed replies cannot measure how Qwen chooses actions or responds "
               "to changed feedback. No training, semantic reward or model improvement is measured."
               }
    write(output / "inputs.json", {"queue_sha256": digest(queue), "drafts_sha256": digest(drafts),
        "sandbox_image": image, "baseline_plan_sha256": queue["baseline"]["frozen_plan_sha256"],
        "implementation_sha256": {p: file_hash(ROOT / p) for p in (
            "src/eval/reader_correction_replay.py", "scripts/replay_reader_corrections.py")}})
    try:
        # All original episodes must match before the first replacement can run.
        for record in records:
            destination = output / "originals" / record["case_id"]
            destination.mkdir(parents=True)
            summary["originals"].append(replay_one(bundle, record, groups[record["case_id"]],
                                                  image, destination, containers))
            write(output / "summary.json", summary)
        for candidate in drafts["correction_drafts"]:
            record = by_id[candidate["case_id"]]
            destination = output / "corrections" / candidate["case_id"]
            destination.mkdir(parents=True)
            summary["corrections"].append(replay_one(bundle, record, groups[record["case_id"]],
                image, destination, containers, candidate=candidate))
            write(output / "summary.json", summary)
        summary["status"] = "passed_fixed_action_regression_replay"
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            proofs = cleanup(containers)
            write(output / "cleanup.json", {"owned_containers": proofs,
                "all_absent": all(p["absent"] for p in proofs)})
            if not all(p["absent"] for p in proofs):
                raise RuntimeError("owned replay container cleanup could not be verified")
        except Exception as exc:
            summary.update(status="failed_cleanup", cleanup_error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write(output / "summary.json", summary)
    return summary
