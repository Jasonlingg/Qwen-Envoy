"""Data gates, exclusion semantics and offline-only demonstration export."""
import copy
from pathlib import Path

import pytest

from src.eval import passage_pilot_data as data
from src.eval import passage_pilot_replay as runner
from src.eval.reader_improvement import digest, file_hash, load, write
from src.product.passage_scope import make_scope


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Small synthetic sources, exercising actual 18-paper/36-case gate shape."""
    monkeypatch.setattr(data, "check_registry", lambda *args: None)
    root = tmp_path / "prepared"
    root.mkdir()
    cases = []
    for i in range(18):
        split = "train" if i < 12 else "validation"
        qid = f"qasper_{split}_{i:040x}"
        doc_id = f"qasper_2000_{i:05d}"
        doc = {"doc_id": doc_id, "title": "Fixture", "text": "source " * 1200,
               "metadata": {"source_revision": "fixture"}}
        snap = root / "snapshots" / qid
        (snap / "corpus").mkdir(parents=True)
        write(snap / "corpus" / (doc_id + ".json"), doc)
        from src.eval.artifacts import content_hash
        write(snap / "manifest.json", {"corpus_hash": content_hash(snap / "corpus")})
        for state, first in (("present", 0), ("missing", 1600)):
            scope = make_scope({doc_id: doc}, [{"start": first, "end": first + 1600},
                {"start": 4800, "end": 6400}], version="passages-v2")
            cases.append({"case_id": qid + "--" + state, "question_id": qid,
                "role": "train" if i < 12 else "development", "source_split": split,
                "question": "What source?", "doc_id": doc_id,
                "snapshot": str(snap.relative_to(root)), "scope_answerability": state,
                "scope": scope, "annotation_for_review_only": {}, "training_ready": False})
    write(root / "cases.json", cases)
    write(root / "selection.json", {"data_review_rejections": None})
    write(root / "exclusions.json", {"excluded_doc_ids": [], "excluded_question_ids": []})
    def bind():
        write(root / "manifest.json", {"cases_sha256": file_hash(root / "cases.json"),
            "selection_sha256": file_hash(root / "selection.json"),
            "exclusions_sha256": file_hash(root / "exclusions.json"),
            "source_sha256": {str(p.relative_to(root)): file_hash(p)
                              for p in sorted((root / "snapshots").rglob("*.json"))}})
    bind()
    return root, cases, bind


def review_fixture(root, cases):
    review = {"cases_sha256": file_hash(root / "cases.json"), "reviewer_kind": "model_assisted",
              "reviewer_identity": "fixture source reviewer", "reviewed_at": "2026-10-22",
              "rows": []}
    for c in cases:
        start = c["scope"]["windows"][0]["start"]
        doc = load(root / c["snapshot"] / "corpus" / (c["doc_id"] + ".json"))
        span = {"doc_id": c["doc_id"], "start": start, "end": start + 100,
                "quote": doc["text"][start:start + 100]}
        import json
        action = ('SUBMIT: source CITATIONS: ' + json.dumps([c["doc_id"]]) + ' EVIDENCE: ' +
                  json.dumps([{k: span[k] for k in ("doc_id", "start", "end")}]))
        if c["scope_answerability"] == "missing":
            action = ('submit_gap(missing="Specific requested fact missing", '
                      'follow_up_query="source fact")')
        review["rows"].append({"case_id": c["case_id"], "case_sha256": digest(c),
            "decision": "accept", "source_support_reviewed": True, "notes": "source reviewed",
            "requested_fact_review": "requested fact", "supporting_spans": [span],
            "final_action": action})
    return review


def test_metadata_does_not_treat_full_corpus_reservation_as_evaluation():
    used, unused = "qasper_2000_00001", "qasper_2000_00002"
    qid = "qasper_train_" + "1" * 40
    stale = "qasper_train_" + "2" * 40
    docs, qs = data.metadata_ids({"reserved_doc_ids": [used, unused], "papers": [unused],
        "question_ids": [qid, stale], "questions": [{"id": qid, "target_doc_ids": [used],
            "gold_evidence": [{"doc_id": unused}], "text": unused}]})
    assert docs == {used} and qs == {qid}
    locked = {"schema_version": "holdout-lock-v1", "reserved_doc_ids": [unused]}
    assert data.metadata_ids(locked) == ({unused}, set())


def test_inventory_requires_history_and_recomputes_identifiers(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "REQUIRED", ["data/sft/pool/train.jsonl"])
    with pytest.raises(ValueError, match="missing required"):
        data.inventory(tmp_path)
    path = tmp_path / data.REQUIRED[0]
    path.parent.mkdir(parents=True)
    path.write_text('{"doc_id": "qasper_2000_00001"}\n')
    registry = data.inventory(tmp_path)
    data.check_registry(registry, tmp_path)
    assert registry["excluded_doc_ids"] == ["qasper_2000_00001"]
    wrong = copy.deepcopy(registry)
    wrong["files"][0]["paper_ids"] = ["qasper_2000_00002"]
    wrong["excluded_doc_ids"] = ["qasper_2000_00002"]
    with pytest.raises(ValueError, match="identifiers differ"):
        data.check_registry(wrong, tmp_path)
    path.write_text('{}\n')
    with pytest.raises(ValueError, match="source changed"):
        data.check_registry(registry, tmp_path)


def test_paired_windows_propose_scopes_not_semantic_absence():
    text = "answerable by paraphrase " * 400
    text = text[:3400] + "EXACT" + text[3405:]
    windows = data.paired_windows(text, {"evidence": [{"start": 3350, "end": 3450}],
                                        "extractive_spans": ["EXACT"]})
    assert windows and len(windows["missing"]) == 2
    assert all("EXACT" not in text[w["start"]:w["end"]] for w in windows["missing"])
    assert all("paraphrase" in text[w["start"]:w["end"]] for w in windows["missing"])


def test_prepared_integrity_pairing_roles_and_scope(prepared):
    root, cases, bind = prepared
    assert len(data.check_prepared(root)[1]) == 36
    cases[0]["scope"]["read_calls"][0] = "unsafe replacement"
    write(root / "cases.json", cases)
    with pytest.raises(ValueError, match="changed"):
        data.check_prepared(root)
    bind()
    with pytest.raises(ValueError, match="scope differs"):
        data.check_prepared(root)


def test_train_development_overlap_rejected(prepared):
    root, cases, bind = prepared
    cases[24]["doc_id"] = cases[0]["doc_id"]
    write(root / "cases.json", cases)
    bind()
    with pytest.raises(ValueError, match="overlap"):
        data.check_prepared(root)


def test_rejection_rounds_keep_prior_exclusions(prepared, monkeypatch):
    root, cases, bind = prepared
    monkeypatch.setattr(data, "ROOT", root.parent)
    def rejection(case):
        return {"prepared_path": root.name, "cases_sha256": file_hash(root / "cases.json"),
                "reviewer_kind": "model_assisted", "reviewer_identity": "fixture",
                "rows": [{"question_id": case["question_id"], "doc_id": case["doc_id"],
                          "reason": "missing scope answers question"}]}
    first = rejection(cases[0])
    # A separate parent retains the first round, rather than self-referencing.
    import shutil
    parent = root.parent / "parent"
    shutil.copytree(root, parent)
    first["prepared_path"] = "parent"
    write(root / "selection.json", {"data_review_rejections": first})
    bind()
    second = rejection(cases[2])
    assert len(data.reviewed_rejections(second)) == 2
    second["rows"][0]["doc_id"] = "wrong"
    with pytest.raises(ValueError, match="reviewed parent"):
        data.reviewed_rejections(second)


def test_review_requires_attribution_full_coverage_and_exact_source(prepared):
    root, cases, _ = prepared
    review = review_fixture(root, cases)
    assert len(runner.check_review(root, review)[0]) == 36
    review["reviewer_identity"] = None
    with pytest.raises(ValueError, match="attribution"):
        runner.check_review(root, review)
    review = review_fixture(root, cases)
    review["rows"][0]["supporting_spans"][0]["quote"] = "fabricated"
    with pytest.raises(ValueError, match="exact bounded"):
        runner.check_review(root, review)


@pytest.mark.parametrize("action", ["print('gap')",
                                    "submit_gap(missing=str(1), follow_up_query='x')",
                                    "import os\nsubmit_gap(missing='x', follow_up_query='y')"])
def test_missing_targets_cannot_execute_arbitrary_host_code(prepared, action):
    root, cases, _ = prepared
    review = review_fixture(root, cases)
    review["rows"][1]["final_action"] = action
    with pytest.raises(ValueError):
        runner.check_review(root, review)


def test_scripted_transport_is_bounded_and_has_no_usage_or_network_fallback():
    transport = runner.DemonstrationTransport(["fixed action"])
    payload = {"model": "envoy", "messages": [{"role": "system", "content": runner.PROMPT}]}
    assert "usage" not in transport("unused", payload, {}, 1)
    assert transport.payloads == [payload]
    with pytest.raises(ValueError, match="exceeds"):
        transport("unused", payload, {}, 1)


def test_no_docker_blocks_before_replay_output(prepared, tmp_path, monkeypatch):
    root, cases, _ = prepared
    review = tmp_path / "review.json"
    write(review, review_fixture(root, cases))
    monkeypatch.setattr(runner, "frozen_runtime", lambda: {"sandbox_image": "sha256:fixture"})
    monkeypatch.setattr(runner.PersistentREPL, "_docker_available", lambda _: False)
    with pytest.raises(ValueError, match="no host fallback"):
        runner.replay(root, review, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_frozen_runtime_matches_current_baseline():
    assert runner.frozen_runtime()["sandbox_image"].startswith("sha256:")


def test_output_cannot_mutate_frozen_input_or_ancestor(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        runner.outside_inputs(tmp_path / "data" / "export", tmp_path / "data")
    with pytest.raises(ValueError, match="separate"):
        runner.outside_inputs(tmp_path, tmp_path / "data")


def test_partial_or_wrong_terminal_review_blocks_replay(prepared):
    root, cases, _ = prepared
    review = review_fixture(root, cases)
    review["rows"].pop()
    with pytest.raises(ValueError, match="all cases"):
        runner.check_review(root, review)
    review = review_fixture(root, cases)
    review["rows"][0]["final_action"] = 'submit_gap(missing="x", follow_up_query="y")'
    with pytest.raises(ValueError, match="present case"):
        runner.check_review(root, review)


def test_failed_demonstration_retains_failure_and_cleans_owned_containers(
        prepared, tmp_path, monkeypatch):
    root, cases, _ = prepared
    review = tmp_path / "review.json"
    write(review, review_fixture(root, cases))
    plan = {"sandbox_image": "sha256:fixture", "implementation_sha256": {},
            "additional_implementation_sha256": {}}
    monkeypatch.setattr(runner, "frozen_runtime", lambda: plan)
    monkeypatch.setattr(runner.PersistentREPL, "_docker_available", lambda _: True)
    def fail(*args):
        args[-1].append("owned-container")
        raise ValueError("read failed")
    monkeypatch.setattr(runner, "replay_case", fail)
    cleaned = []
    def cleanup(containers):
        cleaned.extend(containers)
        return [{"id": c, "absent": True} for c in containers]
    monkeypatch.setattr(runner, "cleanup", cleanup)
    out = tmp_path / "replay"
    with pytest.raises(ValueError, match="read failed"):
        runner.replay(root, review, out)
    assert cleaned == ["owned-container"]
    assert load(out / "summary.json")["status"] == "failed"
    assert not (out / "train.jsonl").exists()


def test_export_never_includes_development_or_accepts_partial_replay(
        prepared, tmp_path, monkeypatch):
    """Isolate the train/dev boundary; actual replay exercises Docker separately."""
    import json

    import scripts.train_sft as trainer
    import src.product.experiment_tokenizer as tokenizers
    root, cases, _ = prepared
    review = tmp_path / "review.json"
    write(review, review_fixture(root, cases))
    replay = tmp_path / "replay"
    replay.mkdir()
    for c in cases:
        base = replay / "cases" / c["case_id"]
        base.mkdir(parents=True)
        # Mark all messages by case so any development leakage is detectable.
        messages = [{"role": "user", "content": c["case_id"]}]
        for _ in range(2):
            messages += [{"role": "assistant", "content": "read"},
                         {"role": "user", "content": c["case_id"]}]
        write(base / "payloads.json", [{"messages": messages}])
        for name in ("packet", "verified", "receipt"):
            write(base / (name + ".json"), {})
    write(replay / "summary.json", {"status": "passed_demonstration_replay",
        "new_model_calls": 0, "cases": [{"case_id": c["case_id"]} for c in cases]})
    write(replay / "cleanup.json", {"all_absent": True,
        "containers": [{"id": str(i), "absent": True} for i in range(36)]})
    manifest = {"schema_version": "passage-pilot-replay-v1",
        "prepared_manifest_sha256": file_hash(root / "manifest.json"),
        "review_sha256": file_hash(review), "prompt_sha256": runner.PROMPT_SHA,
        "implementation_sha256": {}, "artifact_sha256": {
            str(p.relative_to(replay)): file_hash(p) for p in replay.rglob("*.json")}}
    write(replay / "manifest.json", manifest)
    monkeypatch.setattr(runner, "verify_case", lambda *args: {})
    monkeypatch.setattr(tokenizers, "load_tokenizer", lambda *args: object())
    def tokenize(tokenizer, examples, *args):
        return [{"labels": [-100, 1]} for _ in examples], {"actions": len(examples)}
    monkeypatch.setattr(trainer, "_tokenize_per_action", tokenize)
    result = runner.export(root, review, replay, tmp_path / "export")
    assert result["training_ready_cases"] == 24 and result["token_checks"]["actions"] == 72
    rows = [json.loads(line) for line in (tmp_path / "export/train.jsonl").read_text().splitlines()]
    assert len(rows) == 24
    raw = json.dumps(rows)
    assert all(c["case_id"] not in raw for c in cases if c["role"] == "development")
    assert len(load(tmp_path / "export/development-scoring-only.json")) == 12
    (replay / "summary.json").write_text('{}')
    with pytest.raises(ValueError, match="changed replay artifacts"):
        runner.export(root, review, replay, tmp_path / "tampered-export")
    assert not (tmp_path / "tampered-export").exists()


def test_unavailable_packet_reports_docker_failure_not_missing_counter(prepared, monkeypatch):
    root, cases, _ = prepared
    monkeypatch.setattr(runner, "verify_investigation_packet", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="Docker demonstration unavailable/failed"):
        runner.verify_case(root, cases[0], {}, {"status": "unavailable",
                           "error": "Docker image unavailable"}, [])


def test_saved_bundle_binds_real_observations_review_and_train_only_export():
    """Portable artifact check, not a re-audit of unredistributed historical jobs."""
    import json
    root = Path(__file__).resolve().parents[1]
    bundle = root / "examples/passage-reader-pilot-data"
    validation = load(bundle / "validation.json")
    for name, expected in validation["artifact_sha256"].items():
        assert file_hash(bundle / name) == expected
    prepared = bundle / "prepared"
    cases = load(prepared / "cases.json")
    rows = {r["case_id"]: r for r in load(
        prepared / "implementing-assistant-review.json")["rows"]}
    registry = load(prepared / "exclusions.json")
    assert not {c["doc_id"] for c in cases} & set(registry["excluded_doc_ids"])
    assert not {c["question_id"] for c in cases} & set(registry["excluded_question_ids"])
    train = root / "data/sft/passage-reader-pilot-v1"
    manifest = load(train / "manifest.json")
    assert file_hash(train / "manifest.json") == validation["export_manifest_sha256"]
    for name, expected in manifest["artifact_sha256"].items():
        assert file_hash(train / name) == expected
    actual = [json.loads(line) for line in (train / "train.jsonl").read_text().splitlines()]
    expected = []
    for case in cases:
        saved = bundle / "replay/cases" / case["case_id"]
        packet, payloads = load(saved / "packet.json"), load(saved / "payloads.json")
        assert runner.verify_case(prepared, case, rows[case["case_id"]], packet, payloads) == load(
            saved / "verified.json")
        if case["role"] == "train":
            expected.append({"messages": payloads[-1]["messages"] + [{"role": "assistant",
                "content": rows[case["case_id"]]["final_action"]}]})
    assert actual == expected and len(actual) == 24
    dev = load(train / "development-scoring-only.json")
    assert len(dev) == 12 and all(c["usage"] == "scoring_only_never_training" for c in dev)
    assert manifest["token_checks"] == {"actions": 72, "input_tokens": 225496,
                                       "supervised_tokens": 3018, "max_tokens": 3729}
    assert manifest["training_performed"] is False and manifest["human_review"] is None


def test_structured_training_pools_are_inventoried_without_corpus_reservations(
        tmp_path, monkeypatch):
    monkeypatch.setattr(data, "REQUIRED", [])
    path = tmp_path / "out/research/training-pool/benchmark.json"
    path.parent.mkdir(parents=True)
    qid = "qasper_train_" + "1" * 40
    write(path, {"questions": [{"id": qid, "target_doc_ids": ["qasper_2000_00001"]}],
                 "reserved_doc_ids": ["qasper_2000_99999"]})
    registry = data.inventory(tmp_path)
    assert registry["schema_version"] == "passage-pilot-exclusions-v2"
    assert registry["excluded_doc_ids"] == ["qasper_2000_00001"]
    assert registry["excluded_question_ids"] == [qid]
    data.check_registry(registry, tmp_path)
    registry["schema_version"] = "passage-pilot-exclusions-v1"
    with pytest.raises(ValueError, match="structured benchmark"):
        data.check_registry(registry, tmp_path)


def test_old_saved_pilot_is_blocked_by_incomplete_inventory():
    old = load(Path(__file__).resolve().parents[1] /
               "examples/passage-reader-pilot-data/prepared/exclusions.json")
    with pytest.raises(ValueError, match="structured benchmark"):
        data.check_registry(old)
