import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.eval.research_pilot_bundle import load_bundle, prepare, validate_questions
from src.eval.research_pilot_tools import FrozenPilotTools, ReaderAdmission
from tests.arxiv_fakes import FakeInvestigator, article_page, make_library

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2609.12345v2"


def sample(tmp_path):
    library, _ = make_library(tmp_path, {VERSION: {"html": article_page()}})
    library.open(VERSION)
    question = {"id": "P01", "question": "What saves memory?", "arxiv_ids": [VERSION]}
    data = {"schema_version": "research-workflow-pilot-v1", "papers": [
        {"arxiv_id": VERSION, "text_sha256": hashlib.sha256(
            library.document(VERSION)["text"].encode()).hexdigest()}],
        "questions": [question], "review_strata": {"P01": "answerable"}}
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data))
    return library, question, path


def test_prepare_freezes_cache_and_refuses_modified_inputs(tmp_path):
    library, _, questions = sample(tmp_path)
    destination = tmp_path / "frozen"
    result = prepare(questions, library.root, destination, ROOT)
    assert result["actual_model_requests"] == 0
    assert result["plan"] == [{"question_id": "P01", "arm": "direct"},
                              {"question_id": "P01", "arm": "delegated"}]
    _, data, frozen = load_bundle(destination)
    assert frozen.document(VERSION) == library.document(VERSION)
    assert "review_strata" in data
    with pytest.raises(ValueError, match="already exists"):
        prepare(questions, library.root, destination, ROOT)
    (destination / "questions.json").write_text("{}")
    with pytest.raises(ValueError, match="questions changed"):
        load_bundle(destination)


def test_hash_mismatch_does_not_leave_half_prepared_bundle(tmp_path):
    library, _, path = sample(tmp_path)
    data = json.loads(path.read_text())
    data["papers"][0]["text_sha256"] = "wrong"
    path.write_text(json.dumps(data))
    destination = tmp_path / "frozen"
    with pytest.raises(ValueError, match="text hash"):
        prepare(path, library.root, destination, ROOT)
    assert not destination.exists()


def test_version_and_stratum_validation(tmp_path):
    _, _, path = sample(tmp_path)
    data = json.loads(path.read_text())
    data["review_strata"] = {}
    with pytest.raises(ValueError, match="stratum"):
        validate_questions(data)
    data["review_strata"] = {"P01": "answerable"}
    data["papers"][0]["arxiv_id"] = VERSION.split("v")[0]
    with pytest.raises(ValueError, match="version-pinned"):
        validate_questions(data)


def test_frozen_tools_scope_and_seen_span_validation(tmp_path):
    library, question, _ = sample(tmp_path)
    tools = FrozenPilotTools(library, question)
    claim = {"answer": "a", "claims": [{"text": "c", "evidence": [
        {"arxiv_id": VERSION, "start": 0, "end": 20}]}], "limitations": []}
    with pytest.raises(ValueError, match="inspected"):
        tools.validate_finish(claim)
    span = tools.get_passage(VERSION, 0, 20)
    checks = tools.validate_finish(claim)
    assert checks["claims"][0]["evidence"][0]["quote"] == span["text"]
    assert checks["semantic_support"] == "not_reviewed"
    with pytest.raises(ValueError, match="frozen"):
        tools.get_passage("2609.54321v1")
    with pytest.raises(ValueError, match="3000"):
        tools.get_passage(VERSION, 0, 3001)
    assert "investigate" not in tools.callbacks("direct")


def test_search_and_reader_spans_are_inspectable_without_source_fetch(tmp_path):
    library, question, _ = sample(tmp_path)
    reader = FakeInvestigator({VERSION: "Eviction saves 40 percent of memory"})
    tools = FrozenPilotTools(library, question, ReaderAdmission(reader))
    found = tools.find_evidence("memory", [VERSION], top_k=2)
    assert found["evidence"]
    packet = tools.investigate(question["question"], [VERSION])
    span = packet["evidence"][0]
    tools.validate_finish({"claims": [{"text": "c", "evidence": [
        {key: span[key] for key in ("arxiv_id", "start", "end")}]}]})
    with pytest.raises(ValueError, match="unchanged"):
        tools.investigate("a different question", [VERSION])
    with pytest.raises(ValueError, match="top_k"):
        tools.find_evidence("memory", [VERSION], top_k=True)


def test_admission_is_shared_across_overlapping_calls_and_releases_on_failure():
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}
    class Reader:
        system_prompt = "prompt"
        def investigate(self, question, snapshot):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                time.sleep(.02)
                if question == "bad":
                    raise RuntimeError("bad")
                return {"status": "ok"}
            finally:
                with lock:
                    state["active"] -= 1
    admission = ReaderAdmission(Reader())
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(admission.investigate, q, None)
                   for q in ["a", "bad", "b", "c", "d", "e"]]
        for future in futures:
            try:
                future.result()
            except RuntimeError:
                pass
    assert state == {"active": 0, "peak": 2}
