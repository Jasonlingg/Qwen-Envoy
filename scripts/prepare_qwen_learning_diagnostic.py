"""Build two reviewed, replayed training episodes and evidence-use controls.

No teacher API calls. This is a deliberately tiny memorization diagnostic, not
a new generalization benchmark. Validation questions never enter train.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_qasper_grpo import PROMPT, file_hash
from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.eval.artifacts import content_hash
from src.eval.qasper_reward import score_submission


SPECS = [
    ("qasper_train_62a3dc90ba427c5985789001a02825c9434ce67d",
     "clinical notes experiment", "The clinical notes we used for the experiment"),
    ("qasper_train_bfa3776c30cb30e0088e185a5908e5172df79236",
     "classification supervised experiments", "To test whether topic models can be used"),
]


def build(benchmark: Path, corpus_path: Path, output: Path):
    data = json.loads(benchmark.read_text())
    if data["source_split"] != "train":
        raise ValueError("The overfit test must use official training questions")
    if content_hash(corpus_path) != data["corpus_hash"]:
        raise ValueError("Corpus hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(str(corpus_path))
    corpus.load(build_index=False)
    by_id = {q["id"]: q for q in data["questions"]}
    questions = [by_id[identifier] for identifier, _, _ in SPECS]
    env = DocumentExplorationEnv(corpus, questions, max_steps=10, use_docker=False,
                                 corpus_path=str(corpus_path), include_preamble=False)
    rows = []
    try:
        for i, (identifier, query, anchor) in enumerate(SPECS):
            q = questions[i]
            gold = q["answer_annotations"][0]["evidence"]
            assert len(gold) == 1
            span = {key: gold[0][key] for key in ("doc_id", "start", "end")}
            assert corpus._documents[span["doc_id"]]["text"][span["start"]:span["end"]] == gold[0]["text"]
            messages = [{"role": "system", "content": PROMPT},
                        {"role": "user", "content": env.reset(question_idx=i)}]
            first = (f'doc_id = {json.dumps(span["doc_id"])}\n'
                     'text = read(doc_id)\n'
                     f'print(search_within(doc_id, {json.dumps(query)}))')
            obs, _, done, _ = env.step(first)
            assert not done and "Traceback" not in obs and anchor in obs
            messages.extend([{"role": "assistant", "content": first}, {"role": "user", "content": obs}])
            second = (f'start = text.find({json.dumps(anchor)})\n'
                      'assert start >= 0\n'
                      'end = text.find("\\n\\n", start)\n'
                      'if end == -1:\n    end = len(text)\n'
                      'print(passage(doc_id, start=start, length=end-start))')
            obs, _, done, _ = env.step(second)
            assert not done and "Traceback" not in obs
            assert str(span["start"]) in obs and str(span["end"]) in obs
            messages.extend([{"role": "assistant", "content": second}, {"role": "user", "content": obs}])
            final = f'SUBMIT: {q["answer"]} CITATIONS: {json.dumps([span["doc_id"]])} EVIDENCE: {json.dumps([span])}'
            score = score_submission(final, q, corpus._documents, investigated=True)
            assert score["reward"] == 1.0, score
            messages.append({"role": "assistant", "content": final})
            rows.append({"question_id": identifier, "source_split": "train", "messages": messages,
                         "review": "assistant-reviewed against source; tools replayed; not independent human review",
                         "teacher_source": "QASPER annotation plus manually authored executable actions", "score": score})
    finally:
        env.close()
    (output / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    subset = {**data, "questions": questions, "question_ids": [q["id"] for q in questions],
              "purpose": "tiny training-set memorization diagnostic"}
    (output / "benchmark.json").write_text(json.dumps(subset, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({
        "source_benchmark": str(benchmark), "source_sha256": file_hash(benchmark),
        "corpus_hash": data["corpus_hash"], "train_sha256": file_hash(output / "train.jsonl"),
        "question_ids": subset["question_ids"], "actions": 6,
        "scope": "memorization only; no validation/test targets used for updates",
    }, indent=2) + "\n")
    print(f"Replayed and validated {len(rows)} three-action episodes at {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    build(args.benchmark, args.corpus, args.out)
