import json
import sys

from scripts.filter_qasper_synthetic_data import main


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def conversation(question):
    return {
        "messages": [
            {"role": "system", "content": "tools"},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "assistant", "content": "SUBMIT: answer"},
        ]
    }


def test_semantic_rejection_removes_candidate_and_matching_split_row(tmp_path, monkeypatch):
    questions = [
        {
            "id": f"q{index}",
            "question": f'Use paper (doc_id: "paper_{index}") to answer: question {index}',
            "expected_answerability": "sufficient",
        }
        for index in range(1, 4)
    ]
    candidates = tmp_path / "candidates.jsonl"
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "val.jsonl"
    benchmark = tmp_path / "benchmark.json"
    reviews = tmp_path / "reviews.json"
    output = tmp_path / "accepted"
    write_jsonl(
        candidates,
        [
            {"question_id": item["id"], "teacher_behavior": "direct_top3"}
            for item in questions
        ],
    )
    write_jsonl(train, [conversation(questions[0]["question"]), conversation(questions[1]["question"])])
    write_jsonl(validation, [conversation(questions[2]["question"])])
    write_json(benchmark, {"questions": questions})
    write_json(
        reviews,
        {
            "reviewer": "reviewer",
            "reviewer_type": "source-visible",
            "reviews": [
                {
                    "question_id": "q1",
                    "trajectory_sha256": "hash",
                    "verdict": "reject",
                    "notes": "unsupported",
                }
            ],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter_qasper_synthetic_data.py",
            "--candidates",
            str(candidates),
            "--train",
            str(train),
            "--validation",
            str(validation),
            "--benchmark",
            str(benchmark),
            "--semantic-review",
            str(reviews),
            "--output-dir",
            str(output),
        ],
    )

    main()

    accepted = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    accepted_train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "manifest.json").read_text())
    assert [row["question_id"] for row in accepted] == ["q2", "q3"]
    assert len(accepted_train) == 1
    assert "question 2" in accepted_train[0]["messages"][1]["content"]
    assert manifest["accepted"] == 2
    assert manifest["excluded_rows"][0]["question_id"] == "q1"
