"""Build replayed QASPER code-execution demonstrations from expert annotations.

The teacher is deterministic and deliberately constrained:

* search queries contain only question words plus a small fixed vocabulary;
* an answer is emitted only after the annotated evidence has appeared in a real
  REPL observation;
* exact evidence offsets come from a subsequent ``passage`` call, never from an
  invented observation; and
* papers are split before export so train and validation remain disjoint.

This is a data-construction utility, not an evaluation policy. Gold evidence is
used to select and verify a learnable action path on QASPER's training split.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.eval.artifacts import content_hash
from src.eval.harness import run_single
from src.eval.hashing import sha256
from src.eval.qasper_reward import score_submission
from src.eval.sft_quality import known_paper_id, trajectory_hash, trajectory_issues
from src.policies.code_execution import QASPER_SYSTEM_PROMPT

STOP_WORDS = set(
    "use known paper doc id to answer what which who when where why how do does did "
    "is are was were the a an of in on for from with and or their they this that these "
    "those it its as by be been being have has had can could would should may might "
    "about into than then there any all part based study authors".split()
)

# Fixed answer-form vocabulary. None of these strings comes from a reference
# answer. Candidate selection may use gold evidence, but candidate construction
# cannot leak an unseen answer entity into the student's action.
GENERIC_QUERIES = (
    "dataset corpus benchmark data",
    "evaluation metric score accuracy performance",
    "method approach procedure process",
    "limitation challenge future work",
    "model architecture baseline system",
    "training train pretrained",
    "number total count samples examples",
    "experiment experiments results analysis",
)


class FixedActions:
    def __init__(self, actions: list[str]):
        self.actions = actions
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def act(self, observation: str) -> str:
        del observation
        action = self.actions[self.index]
        self.index += 1
        return action


def question_terms(question: str) -> list[str]:
    prompt = question.split("to answer:", 1)[-1].lower()
    terms = [
        word
        for word in re.findall(r"[a-z0-9][a-z0-9_-]+", prompt)
        if word not in STOP_WORDS and len(word) > 2
    ]
    # Stable de-duplication keeps queries short enough for the student to copy.
    return list(dict.fromkeys(terms))


def query_candidates(question: str) -> list[str]:
    terms = question_terms(question)
    raw = " ".join(terms[:12]) or "method results"
    longest = " ".join(sorted(terms, key=lambda word: (-len(word), word))[:8])
    candidates = [raw]
    if longest and longest != raw:
        candidates.append(longest)
    stem = " ".join(terms[:5])
    candidates.extend(f"{stem} {suffix}".strip() for suffix in GENERIC_QUERIES)
    return list(dict.fromkeys(candidates))


def search_windows(text: str, query: str, top_k: int) -> list[dict]:
    terms = query.lower().split()
    windows = []
    for start in range(0, len(text), 200):
        end = min(start + 500, len(text))
        window = text[start:end]
        score = sum(window.lower().count(term) for term in terms)
        if score > 0:
            windows.append({"text": window, "offset": start, "end": end, "score": score})
    windows.sort(key=lambda item: (-item["score"], item["offset"]))
    return windows[:top_k]


def overlap(window: dict, spans: list[dict]) -> int:
    return sum(
        max(0, min(window["end"], span["end"]) - max(window["offset"], span["start"]))
        for span in spans
    )


def choose_search(question: dict, text: str, spans: list[dict]) -> tuple[str, int, list[dict], bool] | None:
    candidates = query_candidates(question["question"])
    raw_query = candidates[0]
    raw_hits = search_windows(text, raw_query, 3)
    if any(overlap(hit, spans) for hit in raw_hits):
        return raw_query, 3, raw_hits, False

    best = None
    for query in candidates:
        hits = search_windows(text, query, 8)
        coverage = sum(overlap(hit, spans) for hit in hits)
        candidate = (coverage, query, hits)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None or best[0] <= 0:
        return None
    return best[1], 8, best[2], True


def covering_context(hit: dict, span: dict, text_length: int) -> tuple[int, int] | None:
    """Choose a bounded passage using only a visible hit offset plus a fixed look-back."""
    for back in (0, 200, 400, 800, 1200, 1600, 2000):
        start = max(0, hit["offset"] - back)
        end = min(text_length, start + 3000)
        if start <= span["start"] and end >= span["end"]:
            return back, end - start
    return None


def answerable_actions(question: dict, text: str) -> tuple[list[str], str] | None:
    annotation = next((a for a in question["answer_annotations"] if not a["unanswerable"]), None)
    if annotation is None or not annotation.get("evidence"):
        return None
    spans = annotation["evidence"]
    if any(span["doc_id"] != known_paper_id(question["question"]) for span in spans):
        return None

    selected = choose_search(question, text, spans)
    if selected is None:
        return None
    query, top_k, hits, recovery = selected
    doc_id = spans[0]["doc_id"]
    actions = []
    raw_query = query_candidates(question["question"])[0]
    if recovery:
        actions.append(
            f"hits = search_within({doc_id!r}, {raw_query!r}); print(hits)"
        )
    actions.append(
        f"hits = search_within({doc_id!r}, {query!r}, top_k={top_k}); print(hits)"
    )

    # At most two broad contexts keeps the full trajectory within the six-action
    # data gate after recovery search, exact-span verification, and submission.
    context_specs: list[tuple[int, int, int]] = []
    for span in spans:
        choices = sorted(
            ((overlap(hit, [span]), index, hit) for index, hit in enumerate(hits)),
            reverse=True,
        )
        if not choices or choices[0][0] <= 0:
            return None
        _, hit_index, hit = choices[0]
        context = covering_context(hit, span, len(text))
        if context is None:
            return None
        back, length = context
        spec = (hit_index, back, length)
        if spec not in context_specs:
            context_specs.append(spec)
    if len(context_specs) > 2:
        return None

    for number, (hit_index, back, length) in enumerate(context_specs):
        actions.append(
            f"context_{number} = passage({doc_id!r}, "
            f"start=max(0, hits[{hit_index}]['offset'] - {back}), length={length}); "
            f"print(context_{number})"
        )

    # The quoted strings have appeared in the preceding context observation.
    # Re-finding them produces exact, executable offsets instead of fabricating
    # the annotation coordinates in the final answer.
    quotes = [span["text"] for span in spans]
    actions.append(
        f"text = read({doc_id!r}); quotes = {quotes!r}; exact = []; "
        "\nfor quote in quotes:\n"
        "    start = text.find(quote)\n"
        "    assert start >= 0\n"
        f"    exact.append(passage({doc_id!r}, start=start, length=len(quote)))\n"
        "print(exact)"
    )
    evidence = [
        {"doc_id": span["doc_id"], "start": span["start"], "end": span["end"]}
        for span in spans
    ]
    submit = (
        f"SUBMIT: {question['answer'].strip()} "
        f"CITATIONS: {json.dumps([doc_id], separators=(',', ':'))} "
        f"EVIDENCE: {json.dumps(evidence, separators=(',', ':'))}"
    )
    actions.append(submit)
    if len(actions) > 6:
        return None
    return actions, "recovery_top8" if recovery else "direct_top3"


def unanswerable_actions(question: dict) -> tuple[list[str], str]:
    doc_id = known_paper_id(question["question"])
    candidates = query_candidates(question["question"])
    raw = candidates[0]
    refinement = next((candidate for candidate in candidates[1:] if candidate != raw), raw + " evidence")
    return [
        f"hits = search_within({doc_id!r}, {raw!r}); print(hits)",
        f"more_hits = search_within({doc_id!r}, {refinement!r}, top_k=8); print(more_hits)",
        "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []",
    ], "verified_abstention"


def to_conversation(row: dict) -> dict:
    messages = [
        {"role": "system", "content": QASPER_SYSTEM_PROMPT},
        {"role": "user", "content": row["initial_observation"]},
    ]
    for step in row["trajectory"]:
        messages.append({"role": "assistant", "content": step["action"]})
        if not step["action"].startswith("SUBMIT:"):
            messages.append({"role": "user", "content": step["observation"]})
    return {"messages": messages}


def split_by_paper(
    rows: list[dict], *, val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split whole paper groups, keeping every paper on exactly one side."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        doc_id = known_paper_id(row["initial_observation"])
        if not doc_id:
            raise ValueError(f"Cannot determine paper for {row.get('question_id')}")
        groups.setdefault(doc_id, []).append(row)

    papers = sorted(groups)
    random.Random(seed).shuffle(papers)
    target = max(1, round(len(rows) * val_fraction))
    validation: list[dict] = []
    train: list[dict] = []
    for paper in papers:
        group = groups[paper]
        # Fill validation until the next whole-paper group would move it farther
        # from the requested size than leaving that group in training.
        add_distance = abs(target - (len(validation) + len(group)))
        keep_distance = abs(target - len(validation))
        if len(validation) < target and add_distance <= keep_distance:
            validation.extend(group)
        else:
            train.extend(group)
    if not validation or not train:
        raise ValueError("Need at least two paper groups for a train/validation split")
    return validation, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text())
    if benchmark.get("source_split") != "train":
        raise ValueError("Only QASPER training questions may become SFT demonstrations")
    if content_hash(args.corpus) != benchmark["corpus_hash"]:
        raise ValueError("Corpus differs from the frozen benchmark")
    questions = benchmark["questions"][: args.limit]
    corpus = Corpus(corpus_path=str(args.corpus))
    corpus.load(build_index=False)

    accepted, rejected = [], []
    for index, question in enumerate(questions, start=1):
        doc_id = known_paper_id(question["question"])
        text = corpus._documents[doc_id]["text"]
        plan = (
            unanswerable_actions(question)
            if question["expected_answerability"] == "insufficient"
            else answerable_actions(question, text)
        )
        if plan is None:
            rejected.append({"question_id": question["id"], "reason": "no_nonleaking_retrieval_path"})
            continue
        actions, behavior = plan
        env = DocumentExplorationEnv(
            corpus=corpus,
            questions=[question],
            max_steps=10,
            use_docker=False,
            corpus_path=str(args.corpus),
            require_evidence=True,
            include_preamble=False,
        )
        try:
            result = run_single(env, FixedActions(actions), 0)
            row = json.loads(result.model_dump_json())
        finally:
            env.close()
        row.update(
            {
                "question_id": question["id"],
                "expected_answerability": question["expected_answerability"],
                "student_protocol": "qasper-span-v1",
                "initial_observation": f"Question: {question['question']}\n",
                "teacher_model": "deterministic-qasper-annotation-builder-v1",
                "teacher_behavior": behavior,
            }
        )
        for step in row["trajectory"]:
            step["raw_action"] = step["action"]
        score = score_submission(
            row["trajectory"][-1]["action"], question, corpus._documents, investigated=True
        )
        row["qasper_score"] = score
        issues = trajectory_issues(row)
        if not score["valid"] or score["answer_f1"] != 1.0 or issues:
            rejected.append(
                {"question_id": question["id"], "reason": "replay_or_score_failure", "issues": issues, "score": score}
            )
            continue
        row["review"] = {
            "basis": "QASPER expert answer/evidence annotation plus exact fresh REPL replay",
            "replay_verified": True,
            "evidence_checked": True,
            "stopping_checked": True,
            "trajectory_sha256": trajectory_hash(row),
        }
        accepted.append(row)
        print(f"[{index}/{len(questions)}] accepted {question['id']} ({behavior})")

    if not accepted:
        raise RuntimeError("No trajectories passed")

    # QASPER occasionally repeats an identical question annotation within one
    # paper. Repeating the exact same action trace would overweight that one
    # example and makes exported conversations ambiguous to independent audits.
    deduplicated: list[dict] = []
    seen_signatures: set[tuple[str, tuple[str, ...]]] = set()
    for row in sorted(accepted, key=lambda item: item["question_id"]):
        signature = (row["question"], tuple(step["action"] for step in row["trajectory"]))
        if signature in seen_signatures:
            rejected.append({"question_id": row["question_id"], "reason": "duplicate_question_and_trace"})
            continue
        seen_signatures.add(signature)
        deduplicated.append(row)
    accepted = deduplicated

    validation, train = split_by_paper(
        accepted, val_fraction=args.val_fraction, seed=args.seed
    )

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in accepted)
    )
    (args.output / "train.jsonl").write_text(
        "".join(json.dumps(to_conversation(row)) + "\n" for row in train)
    )
    (args.output / "val.jsonl").write_text(
        "".join(json.dumps(to_conversation(row)) + "\n" for row in validation)
    )
    (args.output / "rejected.json").write_text(json.dumps(rejected, indent=2) + "\n")
    manifest = {
        "schema_version": "qasper-annotation-sft-v1",
        "builder": "scripts/build_qasper_annotation_sft.py",
        "builder_sha256": sha256(Path(__file__)),
        "benchmark": {"path": str(args.benchmark), "sha256": sha256(args.benchmark)},
        "corpus": {"path": str(args.corpus), "hash": benchmark["corpus_hash"]},
        "selection_seed": args.seed,
        "val_fraction": args.val_fraction,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "behaviors": dict(Counter(row["teacher_behavior"] for row in accepted)),
        "answerability": dict(Counter(row["expected_answerability"] for row in accepted)),
        "splits": {
            "train": {
                "count": len(train),
                "papers": len({known_paper_id(row["initial_observation"]) for row in train}),
                "sha256": sha256(args.output / "train.jsonl"),
            },
            "val": {
                "count": len(validation),
                "papers": len({known_paper_id(row["initial_observation"]) for row in validation}),
                "sha256": sha256(args.output / "val.jsonl"),
            },
        },
        "claim_boundary": (
            "Training data derived from QASPER train annotations and exact REPL replay; "
            "not an independently authored teacher-policy sample or a held-out result."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
