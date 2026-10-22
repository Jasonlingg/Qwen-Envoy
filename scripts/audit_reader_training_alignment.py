"""Read-only V5 data audit against the frozen passage-reader contract."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_qasper_sft_data import audit as legacy_audit
from scripts.export_qasper_sft_data import _to_conversation
from scripts.train_sft import BASE_MODEL, BASE_REVISION, _expand_per_action, _tokenize_per_action
from src.env.reward import parse_submission_details
from src.eval.reader_improvement import digest, file_hash, inside, load, write
from src.eval.sft_quality import known_paper_id
from src.product.assigned_actions import displayed_tool_call
from src.product.experiment_tokenizer import load_tokenizer
from src.product.passage_scope import PASSAGES_V2, normalize_read_arithmetic
from src.product.qwen_investigator import effective_system_prompt

ROOT = Path(__file__).resolve().parents[1]
UNCERTAINTY = re.compile(
    r"\b(?:cannot|can't|not establish|not specified|not reported|"
    r"does not specify|do not specify|does not provide|do not provide)\b", re.I)


def row_audit(conversation, candidate, current_prompt):
    messages = conversation["messages"]
    question = messages[1]["content"].removeprefix("Question: ")
    if question != candidate["question"]:
        raise ValueError("conversation differs from its reviewed question")
    actions = [m["content"] for m in messages if m["role"] == "assistant"]
    if actions != [t["action"] for t in candidate["trajectory"]]:
        raise ValueError("conversation differs from reviewed actions")
    # The historical exporter strips outer whitespace for this legacy protocol.
    # Reuse that rule explicitly rather than calling a normalization a fake observation.
    expected = _to_conversation(candidate, 32000)
    if conversation != expected:
        raise ValueError("conversation differs from actual recorded observations")
    parsed = parse_submission_details(actions[-1])
    if parsed is None:
        raise ValueError("expected a saved terminal SUBMIT in historical V5 data")
    tools, forms = Counter(), []
    for action in actions[:-1]:
        tree = ast.parse(action)
        tools.update(n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id != "print")
        try:
            call = displayed_tool_call(normalize_read_arithmetic(action))
            compatible = call.func.id == "passage"
        except ValueError:
            compatible = False
        forms.append(compatible)
    answer, citations, evidence = parsed
    return {
        "question_id": candidate["question_id"], "doc_id": known_paper_id(question),
        "expected_answerability": candidate["expected_answerability"],
        "actions": len(actions), "end_by_action_three": len(actions) <= 3,
        "one_tool_action_before_submit": len(actions) == 2,
        "assistant_repaired": bool(candidate.get("curation")),
        "conversation_sha256": digest(conversation), "candidate_sha256": digest(candidate),
        "whitespace_normalized_observations": sum(
            t["observation"] != t["observation"].strip()
            for t in candidate["trajectory"][:-1]),
        "system_prompt_matches_current": messages[0]["content"] == current_prompt,
        "initial_observation_is_question_only": messages[1]["content"] == "Question: " + question,
        "tool_calls": dict(tools), "nonterminal_passage_form_compatible": forms,
        "first_action_passage_form_compatible": forms[0],
        "scope_compatibility": "not_checked_no_controller_ranges_in_old_data",
        "terminal_has_evidence_spans": bool(evidence),
        "terminal_has_citations": bool(citations),
        "terminal_is_bare_unanswerable": answer.strip().lower() == "unanswerable",
        "terminal_is_submit_gap": False,
        "insufficient_label_with_cited_prose": candidate["expected_answerability"] == "insufficient"
        and bool(citations) and answer.strip().lower() != "unanswerable",
        "uncertainty_phrase_flag": bool(UNCERTAINTY.search(answer)),
        "final_action": actions[-1], "semantic_support_review": None,
    }


def split_report(rows):
    counters = Counter({k: 0 for k, v in rows[0].items() if type(v) is bool})
    for row in rows:
        counters.update(k for k, v in row.items() if type(v) is bool and v)
    return {"conversations": len(rows), "assistant_actions": sum(r["actions"] for r in rows),
            "whitespace_normalized_observations": sum(
                r["whitespace_normalized_observations"] for r in rows),
            "answerability": dict(Counter(r["expected_answerability"] for r in rows)),
            "action_count_distribution": dict(sorted(Counter(r["actions"] for r in rows).items())),
            "flags": dict(counters), "rows": rows}


def tokenize(conversations, report):
    tokenizer = load_tokenizer(BASE_MODEL, BASE_REVISION)  # Local files only, never a download.
    result = {"model": BASE_MODEL, "revision": BASE_REVISION, "local_files_only": True,
              "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
              "splits": {}}
    for split, rows in conversations.items():
        features, stats = _tokenize_per_action(tokenizer, _expand_per_action(rows), 8192, split)
        offset, allocation = 0, Counter()
        for conversation, record in zip(rows, report["splits"][split]["rows"], strict=True):
            actions = [m for m in conversation["messages"] if m["role"] == "assistant"]
            counts = [sum(v != -100 for v in f["labels"])
                      for f in features[offset:offset + len(actions)]]
            offset += len(actions)
            record["supervised_tokens_by_action"] = counts
            allocation["tool_action_tokens"] += sum(counts[:-1])
            allocation[record["expected_answerability"] + "_terminal_tokens"] += counts[-1]
        if sum(allocation.values()) != stats["supervised_tokens"]:
            raise ValueError("supervised token allocation differs from trainer labels")
        result["splits"][split] = {**stats, "allocation": dict(allocation)}
    return result


def audit(root=ROOT, *, with_tokenization=False):
    data = root / "data/sft/qasper-v5"
    manifest = load(data / "manifest.json")
    training = load(root / "release/qwen-v5-training/run-manifest.json")
    checked = {}
    for item in (manifest["sources"] + [manifest["benchmark"], manifest["review"]]
                 + manifest["exclusions"]):
        path = inside(root, item["path"])
        if file_hash(path) != item["sha256"]:
            raise ValueError("historical dataset provenance changed")
        checked[item["path"]] = item["sha256"]
    if file_hash(data / "manifest.json") != training["corpus_manifest"]["sha256"]:
        raise ValueError("dataset manifest differs from actual training run")
    candidates = [json.loads(line) for line in (data / "candidates.jsonl").read_text().splitlines()]
    parents = load(inside(root, manifest["sources"][0]["path"]))
    by_id = {r["question_id"]: r for r in candidates}
    if len(by_id) != len(candidates) or by_id != {r["question_id"]: r for r in parents}:
        raise ValueError("candidate records differ from reviewed parents")
    prompt = effective_system_prompt("public_papers", "default", "v2", reader_mode="assigned",
                                     reader_guidance=PASSAGES_V2)
    frozen_inputs = load(root / "examples/qasper-hybrid-v5/frozen/first-inputs.json")
    if any(p["messages"][0]["content"] != prompt for p in frozen_inputs.values()):
        raise ValueError("current prompt differs from frozen reader baseline")
    report = {"schema_version": "reader-training-alignment-audit-v1", "splits": {},
              "current_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "historical_prompt_sha256": manifest["system_prompt_sha256"],
              "checked_source_sha256": checked, "human_review": None,
              "new_model_calls": 0, "training_performed": False,
              "trainer": {"historical_sha256": training["training_script_sha256"],
                          "current_sha256": file_hash(root / "scripts/train_sft.py")},
              "limitations": ["Structural/lexical audit, not fresh semantic grading or causality.",
              "Uncertainty flags are lexical cues, not judgments that a refusal is wrong.",
              "V5 labels/reviews are historical model-assisted curation, "
              "not independent human review."]}
    seen, conversations = set(), {}
    for split in ("train", "val"):
        entry = manifest["splits"][split]
        path = inside(root, entry["path"])
        if file_hash(path) != entry["sha256"]:
            raise ValueError("training split changed")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = entry["question_ids"]
        if len(rows) != len(ids) or seen.intersection(ids):
            raise ValueError("split question identities overlap or differ")
        seen.update(ids)
        report["splits"][split] = split_report([
            row_audit(row, by_id[qid], prompt) for row, qid in zip(rows, ids, strict=True)])
        conversations[split] = rows
        checked[entry["path"]] = entry["sha256"]
    if seen != set(by_id):
        raise ValueError("not all candidate identities were accounted for")
    papers = {s: {r["doc_id"] for r in v["rows"]} for s, v in report["splits"].items()}
    if papers["train"] & papers["val"]:
        raise ValueError("training and validation papers overlap")
    report["train_val_paper_overlap"] = []
    report["evaluation_exclusions"] = []
    for name in ("dev", "test_candidate"):
        path = root / f"benchmarks/envoybench/data/{name}/benchmark.json"
        questions = load(path)["questions"]
        ids = {q["id"] for q in questions}
        # Use paper identifiers only, without loading source text or references.
        docs = {"qasper_" + re.sub(r"[^A-Za-z0-9_-]+", "_", q["source_paper_id"])
                for q in questions}
        report["evaluation_exclusions"].append({"set": name, "sha256": file_hash(path),
            "status": "already_exposed", "question_overlap": sorted(seen & ids),
            "paper_overlap": sorted((papers["train"] | papers["val"]) & docs)})
    path = root / "data/research/qasper-v6-holdout-lock.json"
    lock = load(path)
    report["evaluation_exclusions"].append({"set": "old_v6_holdout_lock",
        "sha256": file_hash(path), "status": "consumed_by_completed_v6_comparison",
        "question_overlap": sorted(seen & set(lock["question_ids"])),
        "paper_overlap": sorted((papers["train"] | papers["val"]) & set(lock["reserved_doc_ids"]))})
    cases = load(root / "examples/qasper-hybrid-v5/frozen/cases.json")
    report["evaluation_exclusions"].append({"set": "current_three_case_regression",
        "status": "already_exposed_regression_only",
        "question_overlap": sorted(seen & {c["case_id"] for c in cases}),
        "paper_overlap": sorted((papers["train"] | papers["val"]) & {c["doc_id"] for c in cases})})
    report["legacy_structural_audit"] = legacy_audit(
        [root / manifest["splits"][s]["path"] for s in ("train", "val")],
        load(inside(root, manifest["benchmark"]["path"])))
    report["tokenization"] = tokenize(conversations, report) if with_tokenization else None
    historical = load(data / "tokenization-check.json")
    report["recorded_tokenization"] = historical
    report["checked_source_sha256"].update({
        "data/sft/qasper-v5/manifest.json": file_hash(data / "manifest.json"),
        "data/sft/qasper-v5/candidates.jsonl": file_hash(data / "candidates.jsonl"),
        "data/sft/qasper-v5/tokenization-check.json": file_hash(data / "tokenization-check.json"),
        "release/qwen-v5-training/run-manifest.json": file_hash(
            root / "release/qwen-v5-training/run-manifest.json")})
    if with_tokenization:
        for split, stats in report["tokenization"]["splits"].items():
            stats["matches_recorded_counts"] = all(stats[k] == historical["splits"][split][k]
                for k in ("actions", "input_tokens", "supervised_tokens", "max_tokens"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenize", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; historical audits cannot be overwritten")
    report = audit(with_tokenization=args.tokenize)
    report["environment"] = {"python": platform.python_version(), "machine": platform.machine(),
        "transformers": importlib.metadata.version("transformers") if args.tokenize else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, report)
    print(json.dumps({s: {k: v[k] for k in ("conversations", "assistant_actions", "answerability",
                      "flags")} for s, v in report["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
