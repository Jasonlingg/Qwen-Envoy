"""Fixed, annotation-blind retrieval for an exposed QASPER development smoke."""
from __future__ import annotations

import hashlib
import re

from rank_bm25 import BM25Okapi

SELECTION = "sha256(42:<question ID>); first 2 sufficient + 1 insufficient; distinct papers"
RETRIEVAL = {"kind": "BM25Okapi", "k1": 1.5, "b": 0.75, "epsilon": 0.25,
             "window_chars": 1600, "stride_chars": 800, "windows": 2,
             "tokens": "lowercase Unicode word regex", "tie_break": "start ascending",
             "overlap": "forbidden", "query": "source_question only"}


def select_questions(questions):
    selected, papers = [], set()
    ordered = sorted(questions, key=lambda q: hashlib.sha256(
        ("42:" + q["id"]).encode()).hexdigest())
    for label, count in [("sufficient", 2), ("insufficient", 1)]:
        matches = [q for q in ordered if q["expected_answerability"] == label
                   and q["source_paper_id"] not in papers]
        if len(matches) < count:
            raise ValueError("not enough distinct QASPER papers")
        for question in matches:
            if question["source_paper_id"] in papers:
                continue
            selected.append(question)
            papers.add(question["source_paper_id"])
            count -= 1
            if not count:
                break
        if count:
            raise ValueError("not enough distinct QASPER papers")
    return selected


def retrieve(text, question):
    """Read the complete paper; never accept labels or reference annotations."""
    if not isinstance(text, str) or len(text) < 3200:
        raise ValueError("need two disjoint passage windows; no replacement case")
    starts = list(range(0, len(text) - 1599, 800))
    if starts[-1] + 1600 < len(text):
        starts.append(len(text) - 1600)
    windows = [{"start": start, "end": start + 1600} for start in starts]
    tokens = lambda s: re.findall(r"\w+", s.lower())  # noqa: E731
    corpus = [tokens(text[w["start"]:w["end"]]) for w in windows]
    if not all(corpus):
        raise ValueError("empty retrieval window")
    scores = BM25Okapi(corpus).get_scores(tokens(question))
    ranked = sorted(range(len(windows)), key=lambda i: (-float(scores[i]), starts[i]))
    selected = []
    for index in ranked:
        window = windows[index]
        if all(window["end"] <= w["start"] or window["start"] >= w["end"] for w in selected):
            selected.append(window)
            if len(selected) == 2:
                break
    if len(selected) != 2:
        raise ValueError("could not retrieve two disjoint windows; no replacement case")
    return sorted(selected, key=lambda w: w["start"]), [
        {**windows[i], "score": float(scores[i])} for i in ranked]
