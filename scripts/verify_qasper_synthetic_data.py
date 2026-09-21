"""Independently audit programmatic QASPER code-execution trajectories.

The builder's acceptance flags are deliberately ignored. This auditor recomputes
structural, provenance, leakage, answer, and evidence checks from the frozen
benchmark and corpus. With ``--replay``, it also executes every saved action in a
fresh environment and requires byte-identical actions, observations, and terminal
flags.

Mechanical checks can prove that an answerable trajectory follows the annotated
evidence. They cannot prove that a fact is absent from an entire paper, so every
unanswerable trajectory remains quarantined for semantic review.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.eval.artifacts import content_hash
from src.eval.harness import run_single
from src.eval.qasper_reward import answer_f1, normalize, parse_strict


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|\b(?:SyntaxError|TypeError|NameError):|^ERROR:",
    re.M,
)
GENERIC_QUERY_TOKENS = set(
    "dataset corpus benchmark data evaluation metric score accuracy performance method approach "
    "procedure process limitation challenge future work model architecture baseline system "
    "training train pretrained number total count samples examples experiment experiments results "
    "analysis evidence".split()
)


class FixedActions:
    def __init__(self, actions: list[str]):
        self.actions = actions
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def act(self, observation: str) -> str:
        del observation
        if self.index >= len(self.actions):
            raise RuntimeError("Saved trajectory ended before the replay environment")
        action = self.actions[self.index]
        self.index += 1
        return action


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def known_doc_id(question: str) -> str:
    match = re.search(r'doc_id:\s*"([^"\s]+)"', question)
    if match is None:
        raise ValueError("Known-paper question has no doc_id")
    return match.group(1)


def trajectory_hash(row: dict) -> str:
    payload = {key: row[key] for key in ("question_id", "question", "trajectory")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def search_queries(action: str) -> list[str]:
    tree = ast.parse(action)
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "search_within":
            continue
        query = node.args[1] if len(node.args) > 1 else next(
            (item.value for item in node.keywords if item.arg == "query"), None
        )
        if not isinstance(query, ast.Constant) or not isinstance(query.value, str):
            raise ValueError("search_within query must be a literal string in synthetic data")
        result.append(query.value)
    return result


def _observation_payload(observation: str) -> str:
    return observation.split("\n\n\n[Step ", 1)[0].strip()


def _collect_passages(value: Any) -> list[dict]:
    result = []
    if isinstance(value, dict):
        if {"doc_id", "start", "end", "text"}.issubset(value):
            result.append(value)
        for nested in value.values():
            result.extend(_collect_passages(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            result.extend(_collect_passages(nested))
    return result


def observed_passages(row: dict) -> list[dict]:
    result = []
    for step in row.get("trajectory", [])[:-1]:
        payload = _observation_payload(str(step.get("observation", "")))
        try:
            result.extend(_collect_passages(ast.literal_eval(payload)))
        except (ValueError, SyntaxError):
            continue
    return result


def exact_gold_spans(question: dict) -> set[tuple[str, int, int]]:
    return {
        (span["doc_id"], span["start"], span["end"])
        for annotation in question["answer_annotations"]
        if not annotation["unanswerable"]
        for span in annotation.get("evidence", [])
    }


def accepted_answers(question: dict) -> list[str]:
    return [
        annotation["answer_text"]
        for annotation in question["answer_annotations"]
        if not annotation["unanswerable"]
    ]


def audit_row(row: dict, question: dict, documents: dict[str, dict]) -> list[str]:
    """Return independently recomputed rejection reasons for one trajectory."""
    issues: set[str] = set()
    if row.get("question_id") != question.get("id"):
        issues.add("question_id_mismatch")
    if row.get("question") != question.get("question"):
        issues.add("question_text_mismatch")
    if row.get("expected_answerability") != question.get("expected_answerability"):
        issues.add("answerability_mismatch")
    try:
        doc_id = known_doc_id(question["question"])
    except ValueError:
        return ["missing_known_doc_id"]
    document = documents.get(doc_id)
    if document is None:
        return ["missing_target_document"]

    steps = row.get("trajectory") or []
    if len(steps) < 2:
        issues.add("no_investigation")
    if len(steps) > 6:
        issues.add("over_action_budget")
    if not steps:
        return sorted(issues | {"missing_submission"})
    if any(step.get("done") for step in steps[:-1]):
        issues.add("action_after_done")
    if not steps[-1].get("done"):
        issues.add("submission_not_terminal")
    if not str(steps[-1].get("action", "")).startswith("SUBMIT:"):
        issues.add("missing_submission")

    question_tokens = tokens(question["question"])
    answer_only_tokens = set().union(*(tokens(answer) for answer in accepted_answers(question)))
    answer_only_tokens -= question_tokens | GENERIC_QUERY_TOKENS
    visible_tokens = set(question_tokens)
    successful_document_action = False
    actions_seen = set()
    for step in steps[:-1]:
        action = str(step.get("action", ""))
        observation = str(step.get("observation", ""))
        if action in actions_seen:
            issues.add("repeated_action")
        actions_seen.add(action)
        try:
            tree = ast.parse(action)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in {"search_within", "read", "passage", "extract"}:
                        successful_document_action = True
            for query in search_queries(action):
                query_tokens = tokens(query)
                leaked = query_tokens & answer_only_tokens - visible_tokens
                if leaked:
                    issues.add("gold_answer_query_leak:" + ",".join(sorted(leaked)))
                unobserved = query_tokens - visible_tokens - GENERIC_QUERY_TOKENS
                if unobserved:
                    issues.add("query_uses_unobserved_tokens:" + ",".join(sorted(unobserved)))
        except (SyntaxError, ValueError):
            issues.add("invalid_or_dynamic_action")
        if not observation.strip():
            issues.add("missing_observation")
        if ERROR_RE.search(observation):
            issues.add("execution_error")
        visible_tokens |= tokens(observation)
    if not successful_document_action:
        issues.add("no_successful_document_action")

    parsed = parse_strict(str(steps[-1].get("action", "")))
    if parsed is None:
        return sorted(issues | {"invalid_submission"})
    answer, citations, evidence = parsed
    if set(citations) != {item.get("doc_id") for item in evidence}:
        issues.add("citations_evidence_disagree")

    insufficient = question["expected_answerability"] == "insufficient"
    if insufficient:
        if normalize(answer) != "unanswerable" or citations or evidence:
            issues.add("incorrect_abstention_format")
    else:
        references = accepted_answers(question)
        if not references or max(answer_f1(answer, reference) for reference in references) != 1.0:
            issues.add("answer_does_not_match_annotation")
        if normalize(answer) == "unanswerable":
            issues.add("false_refusal")
        if not evidence:
            issues.add("missing_evidence")

    observed = {
        (item.get("doc_id"), item.get("start"), item.get("end"), item.get("text"))
        for item in observed_passages(row)
    }
    gold = exact_gold_spans(question)
    for item in evidence:
        span_doc = item.get("doc_id")
        start, end = item.get("start"), item.get("end")
        doc = documents.get(span_doc)
        if (
            doc is None
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(doc["text"])
        ):
            issues.add("invalid_evidence_span")
            continue
        quote = doc["text"][start:end]
        if (span_doc, start, end) not in gold:
            issues.add("evidence_not_exact_gold")
        if (span_doc, start, end, quote) not in observed:
            issues.add("evidence_not_observed_before_submit")

    saved_hash = row.get("review", {}).get("trajectory_sha256")
    if saved_hash != trajectory_hash(row):
        issues.add("stale_or_missing_trajectory_hash")
    return sorted(issues)


def replay_issues(
    row: dict,
    question: dict,
    corpus: Corpus,
    corpus_path: Path,
) -> list[str]:
    actions = [step["action"] for step in row["trajectory"]]
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=[question],
        max_steps=10,
        use_docker=False,
        corpus_path=str(corpus_path),
        require_evidence=True,
        include_preamble=False,
    )
    try:
        replay = json.loads(run_single(env, FixedActions(actions), 0).model_dump_json())
    except Exception as error:  # retained in the report, never silently accepted
        return [f"replay_exception:{type(error).__name__}"]
    finally:
        env.close()
    if len(replay["trajectory"]) != len(row["trajectory"]):
        return ["replay_step_count_mismatch"]
    issues = []
    for index, (saved, fresh) in enumerate(zip(row["trajectory"], replay["trajectory"]), 1):
        if saved["action"] != fresh["action"]:
            issues.append(f"replay_action_mismatch:{index}")
        if saved.get("observation") != fresh.get("observation"):
            issues.append(f"replay_observation_mismatch:{index}")
        if bool(saved.get("done")) != bool(fresh.get("done")):
            issues.append(f"replay_done_mismatch:{index}")
    return issues


def _replace_submission(row: dict, answer: str, citations: list[str], evidence: list[dict]) -> None:
    row["trajectory"][-1]["action"] = (
        f"SUBMIT: {answer} CITATIONS: {json.dumps(citations, separators=(',', ':'))} "
        f"EVIDENCE: {json.dumps(evidence, separators=(',', ':'))}"
    )


def mutation_suite(row: dict, question: dict, documents: dict[str, dict]) -> dict:
    """Prove that representative corruptions are rejected by the independent audit."""
    parsed = parse_strict(row["trajectory"][-1]["action"])
    if parsed is None or not parsed[2]:
        raise ValueError("Mutation suite needs an answerable trajectory with evidence")
    answer, citations, evidence = parsed
    answer_terms = sorted(tokens(answer) - tokens(question["question"]) - GENERIC_QUERY_TOKENS)
    leak_token = answer_terms[0] if answer_terms else "private_answer_sentinel"

    mutations: dict[str, dict] = {}
    leaked = copy.deepcopy(row)
    first = leaked["trajectory"][0]["action"]
    tree = ast.parse(first)
    query_node = next(
        node.args[1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "search_within"
        and len(node.args) > 1
    )
    query_node.value = f"{query_node.value} {leak_token}"
    leaked["trajectory"][0]["action"] = ast.unparse(tree)
    mutations["hidden_answer_in_query"] = leaked

    shifted = copy.deepcopy(row)
    shifted_evidence = copy.deepcopy(evidence)
    shifted_evidence[0]["start"] += 1
    _replace_submission(shifted, answer, citations, shifted_evidence)
    mutations["shifted_evidence_offset"] = shifted

    tampered = copy.deepcopy(row)
    tampered["trajectory"][0]["observation"] += "\nTAMPERED"
    mutations["tampered_observation"] = tampered

    early = copy.deepcopy(row)
    early["trajectory"] = [copy.deepcopy(row["trajectory"][-1])]
    mutations["submission_without_investigation"] = early

    wrong_doc = copy.deepcopy(row)
    wrong_evidence = copy.deepcopy(evidence)
    wrong_evidence[0]["doc_id"] = "missing_document"
    _replace_submission(wrong_doc, answer, ["missing_document"], wrong_evidence)
    mutations["missing_evidence_document"] = wrong_doc

    result = {}
    for name, changed in mutations.items():
        issues = audit_row(changed, question, documents)
        result[name] = {"rejected": bool(issues), "issues": issues}
    return result


def conversation_membership(
    path: Path, questions_by_text: dict[str, dict]
) -> tuple[set[str], set[str], list[str]]:
    question_ids, papers, unknown = set(), set(), []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        messages = json.loads(line)["messages"]
        question_text = messages[1]["content"].removeprefix("Question: ").strip()
        question = questions_by_text.get(question_text)
        if question is None:
            unknown.append(question_text)
            continue
        question_ids.add(question["id"])
        papers.add(known_doc_id(question["question"]))
    return question_ids, papers, unknown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--semantic-review", type=Path)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text())
    if benchmark.get("source_split") != "train":
        raise ValueError("Synthetic training data must come from the official train split")
    actual_corpus_hash = content_hash(args.corpus)
    if actual_corpus_hash != benchmark.get("corpus_hash"):
        raise ValueError("Frozen corpus hash does not match the benchmark")
    questions = {item["id"]: item for item in benchmark["questions"]}
    questions_by_text = {item["question"].strip(): item for item in benchmark["questions"]}
    rows = [
        json.loads(line)
        for line in args.candidates.read_text().splitlines()
        if line.strip()
    ]
    corpus = Corpus(corpus_path=str(args.corpus))
    corpus.load(build_index=False)

    semantic_reviews = {}
    semantic_review_metadata: dict[str, Any] = {}
    semantic_review_issues: list[str] = []
    if args.semantic_review:
        review_data = json.loads(args.semantic_review.read_text())
        semantic_review_metadata = {
            "reviewer": review_data.get("reviewer"),
            "reviewer_type": review_data.get("reviewer_type"),
        }
        review_rows = review_data.get("reviews", [])
        review_ids = [item.get("question_id") for item in review_rows]
        duplicate_review_ids = sorted(
            item for item, count in Counter(review_ids).items() if item and count > 1
        )
        if duplicate_review_ids:
            semantic_review_issues.append("duplicate_review_ids")
        if not semantic_review_metadata["reviewer"] or not semantic_review_metadata["reviewer_type"]:
            semantic_review_issues.append("missing_reviewer_metadata")
        allowed_verdicts = {"pass", "reject", "ambiguous"}
        if any(item.get("verdict") not in allowed_verdicts for item in review_rows):
            semantic_review_issues.append("invalid_or_missing_verdict")
        semantic_reviews = {item["question_id"]: item for item in review_rows}

    reports = []
    for index, row in enumerate(rows, 1):
        question = questions.get(row.get("question_id"))
        if question is None:
            reports.append({"question_id": row.get("question_id"), "issues": ["unknown_question"]})
            continue
        issues = audit_row(row, question, corpus._documents)
        if args.replay:
            issues.extend(replay_issues(row, question, corpus, args.corpus))
        semantic_review = semantic_reviews.get(row["question_id"])
        if semantic_review and semantic_review.get("trajectory_sha256") != trajectory_hash(row):
            issues.append("stale_semantic_review")
        reports.append(
            {
                "question_id": row["question_id"],
                "doc_id": known_doc_id(question["question"]),
                "expected_answerability": question["expected_answerability"],
                "trajectory_sha256": trajectory_hash(row),
                "issues": sorted(set(issues)),
                "mechanically_valid": not issues,
                "semantic_status": (
                    semantic_review.get("verdict")
                    if semantic_review
                    else (
                        "requires_full_review"
                        if question["expected_answerability"] == "insufficient"
                        else "annotation_grounded_unreviewed"
                    )
                ),
            }
        )
        print(f"[{index}/{len(rows)}] {row['question_id']}: {len(issues)} issue(s)")

    train_ids, train_papers, train_unknown = conversation_membership(
        args.train, questions_by_text
    )
    val_ids, val_papers, val_unknown = conversation_membership(
        args.validation, questions_by_text
    )
    candidate_ids = {row.get("question_id") for row in rows}
    candidate_papers = {item["doc_id"] for item in reports if item.get("doc_id")}
    duplicate_ids = sorted(
        item for item, count in Counter(row.get("question_id") for row in rows).items() if count > 1
    )
    duplicate_hashes = sorted(
        item for item, count in Counter(trajectory_hash(row) for row in rows).items() if count > 1
    )
    excluded_papers = set(benchmark.get("excluded_doc_ids", []))
    dataset_issues = {
        "duplicate_question_ids": duplicate_ids,
        "duplicate_trajectory_hashes": duplicate_hashes,
        "unknown_train_questions": train_unknown,
        "unknown_validation_questions": val_unknown,
        "train_validation_question_overlap": sorted(train_ids & val_ids),
        "train_validation_paper_overlap": sorted(train_papers & val_papers),
        "candidate_excluded_paper_overlap": sorted(candidate_papers & excluded_papers),
        "split_missing_candidates": sorted(candidate_ids - train_ids - val_ids),
        "split_orphan_questions": sorted((train_ids | val_ids) - candidate_ids),
    }

    mutation_source = next(
        (
            row
            for row in rows
            if questions[row["question_id"]]["expected_answerability"] == "sufficient"
            and not audit_row(row, questions[row["question_id"]], corpus._documents)
        ),
        None,
    )
    if mutation_source is None:
        raise RuntimeError("No valid answerable trajectory available for mutation tests")
    mutation_question = questions[mutation_source["question_id"]]
    mutations = mutation_suite(mutation_source, mutation_question, corpus._documents)
    mechanical_failures = [item for item in reports if item["issues"]]
    insufficient_ids = {
        item["question_id"]
        for item in reports
        if item["expected_answerability"] == "insufficient"
    }
    sufficient_ids = {
        item["question_id"]
        for item in reports
        if item["expected_answerability"] == "sufficient"
    }
    reviewed_insufficient = insufficient_ids & semantic_reviews.keys()
    reviewed_sufficient = sufficient_ids & semantic_reviews.keys()
    semantic_failures = sorted(
        question_id
        for question_id, review in semantic_reviews.items()
        if question_id in candidate_ids
        if review.get("verdict") != "pass"
    )
    required_answerable_spot_checks = min(20, len(sufficient_ids))
    semantic_gate = {
        "all_unanswerable_reviewed": reviewed_insufficient == insufficient_ids,
        "answerable_spot_checks_complete": (
            len(reviewed_sufficient) >= required_answerable_spot_checks
        ),
        "no_semantic_review_failures": not semantic_failures,
        "review_metadata_valid": not semantic_review_issues,
    }
    semantic_gate["complete"] = all(semantic_gate.values())
    gate = {
        "all_rows_mechanically_valid": not mechanical_failures,
        "all_corruptions_rejected": all(item["rejected"] for item in mutations.values()),
        "paper_splits_disjoint": not dataset_issues["train_validation_paper_overlap"],
        "no_reserved_paper_overlap": not dataset_issues["candidate_excluded_paper_overlap"],
        "no_duplicates_or_missing_rows": not any(
            dataset_issues[key]
            for key in (
                "duplicate_question_ids",
                "duplicate_trajectory_hashes",
                "unknown_train_questions",
                "unknown_validation_questions",
                "train_validation_question_overlap",
                "split_missing_candidates",
                "split_orphan_questions",
            )
        ),
        "semantic_review_complete": semantic_gate["complete"],
    }
    gate["ready_to_scale"] = all(gate.values())
    report = {
        "schema_version": "qasper-synthetic-reliability-audit-v1",
        "inputs": {
            "candidates": str(args.candidates),
            "candidates_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
            "benchmark": str(args.benchmark),
            "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
            "corpus": str(args.corpus),
            "corpus_hash": actual_corpus_hash,
            "replay_enabled": args.replay,
            "semantic_review": str(args.semantic_review) if args.semantic_review else None,
            "semantic_review_sha256": (
                hashlib.sha256(args.semantic_review.read_bytes()).hexdigest()
                if args.semantic_review
                else None
            ),
        },
        "counts": {
            "trajectories": len(rows),
            "mechanically_valid": len(rows) - len(mechanical_failures),
            "mechanically_rejected": len(mechanical_failures),
            "answerable_annotation_grounded": len(sufficient_ids),
            "answerable_semantically_reviewed": len(reviewed_sufficient),
            "unanswerable_semantically_reviewed": len(reviewed_insufficient),
            "semantic_review_failures": len(semantic_failures),
        },
        "gate": gate,
        "semantic_gate": semantic_gate,
        "semantic_review_metadata": semantic_review_metadata,
        "semantic_review_issues": semantic_review_issues,
        "semantic_review_failures": semantic_failures,
        "dataset_issues": dataset_issues,
        "mutation_tests": mutations,
        "rows": reports,
        "claim_boundary": (
            "Replay, provenance, leakage, and exact annotation alignment are mechanical. "
            "Unanswerability remains a semantic absence judgment and must be reviewed before scaling."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": report["counts"], "gate": gate}, indent=2))
    if not all(value for key, value in gate.items() if key != "semantic_review_complete" and key != "ready_to_scale"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
