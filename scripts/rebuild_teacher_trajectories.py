"""Execute explicitly reviewed action edits from scratch; never fabricate observations.

The edit file maps question IDs to {parent_sha256, actions, notes}. Results remain
candidates until a separate replay/source review approves them for SFT export.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.eval.artifacts import content_hash
from src.eval.harness import run_single
from src.eval.sft_quality import trajectory_hash, trajectory_issues


class ReviewedActions:
    def __init__(self, actions):
        self.actions = actions
        self.index = 0

    def reset(self):
        self.index = 0

    def act(self, observation):
        action = self.actions[self.index]
        self.index += 1
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--edits", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    benchmark = json.loads(args.benchmark.read_text())
    edits = json.loads(args.edits.read_text())
    by_id = {r["question_id"]: r for r in source}
    questions = {q["id"]: q for q in benchmark["questions"]}
    if benchmark.get("source_split") != "train":
        raise ValueError("Only official training questions may be repaired for SFT")
    if content_hash(args.corpus) != benchmark["corpus_hash"]:
        raise ValueError("Corpus differs from the recorded benchmark")
    for qid, edit in edits.items():
        if qid not in questions or trajectory_hash(by_id[qid]) != edit["parent_sha256"]:
            raise ValueError(f"Unknown question or stale edit: {qid}")
        if by_id[qid]["question"] != questions[qid]["question"]:
            raise ValueError(f"Question text changed: {qid}")
        actions = edit["actions"]
        if not (edit["notes"].strip() and 2 <= len(actions) <= 6):
            raise ValueError(f"Invalid edit: {qid}")
        if any(a.startswith("SUBMIT:") for a in actions[:-1]):
            raise ValueError(f"Early submission: {qid}")
        if not actions[-1].startswith("SUBMIT:"):
            raise ValueError(f"Missing final submission: {qid}")
    corpus = Corpus(corpus_path=str(args.corpus))
    corpus.load(build_index=False)
    output = []
    for row in source:
        qid = row["question_id"]
        if qid not in edits:
            output.append(row)
            continue
        edit = edits[qid]
        env = DocumentExplorationEnv(
            corpus=corpus, questions=[questions[qid]], max_steps=10,
            use_docker=False, corpus_path=str(args.corpus), require_evidence=False,
        )
        try:
            result = run_single(env, ReviewedActions(edit["actions"]), 0)
            rebuilt = json.loads(result.model_dump_json())
            rebuilt.update({
                "question_id": qid,
                "expected_answerability": questions[qid]["expected_answerability"],
                "teacher_model": row.get("teacher_model"),
                "curation": {"method": "Codex reviewed edit, executed in a fresh episode",
                             "parent_sha256": edit["parent_sha256"], "notes": edit["notes"]},
            })
            issues = trajectory_issues(rebuilt)
            if issues or len(rebuilt["trajectory"]) != len(edit["actions"]):
                raise ValueError(f"Rebuilt episode failed structural checks: {qid}: {issues}")
            output.append(rebuilt)
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    metadata = {"status": "needs_source_review_and_replay", "source": str(args.source),
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "edits": str(args.edits),
                "edits_sha256": hashlib.sha256(args.edits.read_bytes()).hexdigest(),
                "corpus_hash": benchmark["corpus_hash"], "max_steps": 10,
                "edited": len(edits), "total": len(output)}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
