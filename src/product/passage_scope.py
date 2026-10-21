"""Controller-owned passage limits for an opt-in assigned reader."""
from __future__ import annotations

import ast
import hashlib
import json

PASSAGES = "passages-v1"
PASSAGE_PROTOCOL = """
Passage-focused reader v1:
The controller supplies exact candidate passage ranges. Your job is to extract
only the facts explicitly requested by the question, including their stated
conditions. The host handles discovery and synthesis. Do not search, read the
whole paper, infer a new hardware result, or propose an experiment here.
Only passage(doc_id, start, length) reads contained in a supplied range are
allowed. Start with print(passage(...)) using a supplied range. Inspect source
text before submitting. Cite exact spans inside text you actually inspected.
Keep the answer compact; bind numbers to the paper's hardware, baseline, task
and workload. A genuine quote may not support your answer. If the requested
measurement is missing, inspect a second distinct subrange, report_gap(), then
submit Unanswerable. A gap applies to these selected passages, not the full
paper or the literature. Never invent a requested target measurement.
"""


def make_scope(docs, windows):
    if len(docs) != 1 or not isinstance(windows, list) or not 1 <= len(windows) <= 3:
        raise ValueError("select one paper and one to three passage windows")
    doc_id, doc = next(iter(docs.items()))
    total, previous, checked = 0, 0, []
    for window in windows:
        if not isinstance(window, dict) or set(window) != {"start", "end"}:
            raise ValueError("passage windows require start and end only")
        start, end = window["start"], window["end"]
        if (type(start) is not int or type(end) is not int
                or not previous <= start < end <= len(doc["text"]) or end - start > 2000):
            raise ValueError("passage windows must be ordered, disjoint and at most 2000 chars")
        checked.append({"start": start, "end": end})
        total, previous = total + end - start, end
    if total > 4000:
        raise ValueError("assigned passages exceed 4000 characters")
    return {"version": PASSAGES, "doc_id": doc_id, "windows": checked,
            "source_text_sha256": hashlib.sha256(doc["text"].encode()).hexdigest(),
            "counts_as_source_inspection": False}


def scope_hash(scope):
    return hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()


def contains(windows, start, end):
    return (type(start) is int and type(end) is int and start < end
            and any(w["start"] <= start < end <= w["end"] for w in windows))


def checked_read(call, scope):
    if call.func.id != "passage" or len(call.args) > 3:
        raise ValueError("only displayed passage reads inside assigned windows are allowed")
    names = ("doc_id", "start", "length")
    fields = dict(zip(names, call.args))
    for keyword in call.keywords:
        if keyword.arg not in names or keyword.arg in fields:
            raise ValueError("passage arguments must be doc_id, start, length")
        fields[keyword.arg] = keyword.value
    if set(fields) != set(names):
        raise ValueError("supply all three passage arguments explicitly")
    values = {k: ast.literal_eval(v) for k, v in fields.items()}
    start, length = values["start"], values["length"]
    if (values["doc_id"] != scope["doc_id"] or type(start) is not int
            or type(length) is not int or not contains(scope["windows"], start, start + length)):
        raise ValueError("passage read is outside the controller's assigned windows")
    return {"start": start, "end": start + length}


def evidence_is_inspected(spans, scope, inspected):
    return all(isinstance(span, dict) and span.get("doc_id") == scope["doc_id"]
               and contains(inspected, span.get("start"), span.get("end")) for span in spans)
