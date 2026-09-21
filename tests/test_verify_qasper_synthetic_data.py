import copy
import json

from scripts.verify_qasper_synthetic_data import audit_row, mutation_suite, trajectory_hash


def fixture():
    doc_id = "paper_1"
    text = "Background. The system uses BLEU and ROUGE for evaluation. Conclusion."
    quote = "The system uses BLEU and ROUGE for evaluation."
    start = text.index(quote)
    end = start + len(quote)
    question = {
        "id": "q1",
        "question": f'Use the known paper (doc_id: "{doc_id}") to answer: Which evaluation metrics are used?',
        "expected_answerability": "sufficient",
        "answer_annotations": [{
            "answer_type": "extractive",
            "answer_text": "BLEU and ROUGE",
            "unanswerable": False,
            "evidence": [{"doc_id": doc_id, "start": start, "end": end, "text": quote}],
        }],
    }
    exact_observation = repr([{"doc_id": doc_id, "start": start, "end": end, "text": quote}])
    row = {
        "question_id": "q1",
        "question": question["question"],
        "expected_answerability": "sufficient",
        "trajectory": [
            {
                "action": f"hits = search_within('{doc_id}', 'evaluation metrics'); print(hits)",
                "observation": "[{'text': 'evaluation metrics', 'offset': 0}]",
                "done": False,
            },
            {
                "action": f"print(passage('{doc_id}', start={start}, length={end-start}))",
                "observation": exact_observation,
                "done": False,
            },
            {
                "action": (
                    f'SUBMIT: BLEU and ROUGE CITATIONS: ["{doc_id}"] '
                    f'EVIDENCE: [{{"doc_id":"{doc_id}","start":{start},"end":{end}}}]'
                ),
                "observation": "Submitted.",
                "done": True,
            },
        ],
    }
    row["review"] = {"trajectory_sha256": trajectory_hash(row)}
    documents = {doc_id: {"doc_id": doc_id, "text": text}}
    return row, question, documents


def test_valid_annotation_grounded_trajectory_passes():
    row, question, documents = fixture()
    assert audit_row(row, question, documents) == []


def test_mutation_suite_rejects_every_corruption():
    row, question, documents = fixture()
    result = mutation_suite(row, question, documents)
    assert result
    assert all(item["rejected"] for item in result.values())
    assert any(
        issue.startswith("gold_answer_query_leak:")
        for issue in result["hidden_answer_in_query"]["issues"]
    )
    assert "evidence_not_exact_gold" in result["shifted_evidence_offset"]["issues"]
    assert "no_investigation" in result["submission_without_investigation"]["issues"]
    assert "invalid_evidence_span" in result["missing_evidence_document"]["issues"]


def test_stale_review_and_unobserved_evidence_are_rejected():
    row, question, documents = fixture()
    changed = copy.deepcopy(row)
    changed["trajectory"][1]["observation"] = "[]"
    issues = audit_row(changed, question, documents)
    assert "evidence_not_observed_before_submit" in issues
    assert "stale_or_missing_trajectory_hash" in issues


def test_unanswerable_is_mechanically_valid_but_requires_semantic_review():
    row, question, documents = fixture()
    question["expected_answerability"] = "insufficient"
    question["answer_annotations"] = [{
        "answer_type": "unanswerable",
        "answer_text": "Unanswerable",
        "unanswerable": True,
        "evidence": [],
    }]
    row["expected_answerability"] = "insufficient"
    row["trajectory"][-1]["action"] = "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []"
    row["review"]["trajectory_sha256"] = trajectory_hash(row)
    assert audit_row(row, question, documents) == []
