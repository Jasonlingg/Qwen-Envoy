from rank_bm25 import BM25Okapi

from scripts.prepare_ai_paper_id_diagnostics import (
    add_candidates,
    build_passages,
    retrieve_doc_ids,
    tokenize,
)


def test_retrieved_ids_are_ranked_without_answer_labels() -> None:
    documents = {
        "search-paper": {
            "doc_id": "search-paper",
            "title": "Search Training",
            "text": "Retrieved passage tokens are masked during policy-gradient training.",
        },
        "unrelated-paper": {
            "doc_id": "unrelated-paper",
            "title": "Image Classification",
            "text": "A convolutional network classifies photographs.",
        },
    }
    passages = build_passages(documents)
    index = BM25Okapi([
        tokenize(f'{item["title"]} {item["title"]} {item["text"]}') for item in passages
    ])

    ranked = retrieve_doc_ids("Why mask retrieved tokens?", passages, index, top_docs=2)

    assert ranked[0] == "search-paper"
    assert set(ranked) == set(documents)


def test_candidate_prompt_exposes_exact_ids_without_changing_reference() -> None:
    question = {"id": "q1", "question": "What happened?", "answer": "An event."}
    documents = {"paper-1": {"title": "The Paper"}}

    result = add_candidates(question, ["paper-1"], documents)

    assert 'doc_id: "paper-1"' in result["question"]
    assert result["question_without_candidates"] == question["question"]
    assert result["answer"] == question["answer"]
    assert question["question"] == "What happened?"
