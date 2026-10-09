from scripts.build_qasper_annotation_sft import split_by_paper


def _row(question_id: str, paper: str) -> dict:
    return {
        "question_id": question_id,
        "initial_observation": (
            f'Question: Use the known paper "Example" (doc_id: "{paper}") to answer: test?'
        ),
    }


def test_split_by_paper_keeps_question_groups_together():
    rows = [
        _row("a1", "qasper_1"),
        _row("a2", "qasper_1"),
        _row("b1", "qasper_2"),
        _row("c1", "qasper_3"),
        _row("c2", "qasper_3"),
        _row("d1", "qasper_4"),
    ]

    validation, train = split_by_paper(rows, val_fraction=0.34, seed=7)

    val_papers = {row["initial_observation"].split('doc_id: "', 1)[1].split('"', 1)[0] for row in validation}
    train_papers = {row["initial_observation"].split('doc_id: "', 1)[1].split('"', 1)[0] for row in train}
    assert not val_papers & train_papers
    assert {row["question_id"] for row in validation + train} == {
        row["question_id"] for row in rows
    }
    assert split_by_paper(rows, val_fraction=0.34, seed=7) == (validation, train)
