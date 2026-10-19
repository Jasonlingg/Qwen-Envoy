import hashlib
import json
import os

import pytest

from src.product.ml_experiment_evidence import EvidenceResolver
from src.product.ml_experiment_io import digest
from src.product.ml_experiment_models import ExperimentBrief
from src.product.ml_experiment_sources import MAX_SOURCE_BYTES, select_pointer
from src.product.ml_experiment_store import ExperimentStore
from tests.arxiv_fakes import article_page, make_library
from tests.ml_experiment_fixtures import complete_brief


def sources(tmp_path):
    library, _ = make_library(tmp_path / "library", {"2609.12345v2": {"html": article_page()}})
    library.open("2609.12345v2")
    exports = tmp_path / "exports"
    exports.mkdir()
    raw = (b'{"metrics":{"passed":3,"total":20},'
           b'"reviewer_kind":"model_assisted","unknown_cost":null}')
    (exports / "run.json").write_bytes(raw)
    resolver = EvidenceResolver(library_root=library.root, artifact_root=exports)
    refs = [{"kind": "paper_span", "reference_id": "paper", "arxiv_id": "2609.12345v2",
             "start": 0, "end": 40},
            {"kind": "local_artifact", "reference_id": "run", "relative_path": "run.json",
             "sha256": digest(raw), "json_pointer": "/metrics"},
            {"kind": "external_reference", "reference_id": "docs",
             "url": "https://example.org/method", "description": "Unverified host reference"}]
    return library, exports, resolver, refs


def test_verifies_sources_and_saves_receipts_that_survive_source_removal(tmp_path):
    library, exports, resolver, refs = sources(tmp_path)
    store = ExperimentStore(tmp_path / "records", evidence_resolver=resolver)
    saved = store.save_brief({**complete_brief(), "references": refs})
    receipts = saved["evidence_receipts"]
    assert [r["verification"] for r in receipts] == [
        "paper_span_verified", "artifact_value_verified", "reference_only"]
    assert receipts[0]["value"] == library.document("2609.12345v2")["text"][:40]
    assert receipts[1]["value"] == {"passed": 3, "total": 20}
    assert all(r["semantic_support"] == "not_reviewed" for r in receipts)
    frozen = store.freeze_plan(saved["experiment_id"], 1)
    assert frozen["frozen_plan"]["evidence_receipts"] == receipts
    (exports / "run.json").unlink()
    (library.papers_dir / "2609.12345v2" / "document.json").unlink()
    assert ExperimentStore(tmp_path / "records").get(saved["experiment_id"]) == frozen


@pytest.mark.parametrize("change", ["quote", "paper_hash", "artifact_hash", "pointer"])
def test_mismatched_evidence_never_creates_a_record(tmp_path, change):
    _, _, resolver, refs = sources(tmp_path)
    if change == "quote":
        refs[0]["quote"] = "x" * 40
    elif change == "paper_hash":
        refs[0]["document_sha256"] = "a" * 64
    elif change == "artifact_hash":
        refs[1]["sha256"] = "a" * 64
    else:
        refs[1]["json_pointer"] = "/absent"
    store = ExperimentStore(tmp_path / "records", evidence_resolver=resolver)
    with pytest.raises(ValueError):
        store.save_brief({"question": "Compare methods", "references": refs})
    assert store.list()["experiments"] == []


@pytest.mark.parametrize("location", ["file", "directory", "cache_parent", "fifo"])
def test_selected_sources_reject_links_and_special_files(tmp_path, location):
    library, exports, resolver, refs = sources(tmp_path)
    if location == "cache_parent":
        library.papers_dir.rename(library.root / "elsewhere")
        library.papers_dir.symlink_to(library.root / "elsewhere", target_is_directory=True)
    else:
        path = exports / "run.json"
        path.unlink()
        if location == "file":
            path.symlink_to(tmp_path / "outside.json")
        elif location == "directory":
            (exports / "linked").symlink_to(exports, target_is_directory=True)
            refs[1]["relative_path"] = "linked/run.json"
        else:
            os.mkfifo(path)
    with pytest.raises(ValueError):
        resolver(ExperimentBrief(question="Question", references=refs).references)


@pytest.mark.parametrize("raw", [b' ' * (MAX_SOURCE_BYTES + 1), b'{"a":1,"a":2}',
                                b'{"a":Infinity}', b'{"huge":"' + b'x' * 5_000 + b'"}'])
def test_exports_are_bounded_strict_json_and_require_small_selections(tmp_path, raw):
    _, exports, resolver, refs = sources(tmp_path)
    (exports / "run.json").write_bytes(raw)
    refs = [{**refs[1], "sha256": digest(raw), "json_pointer": ""}]
    with pytest.raises(ValueError):
        resolver(ExperimentBrief(question="Question", references=refs).references)


def test_cached_document_identity_is_checked_even_with_a_matching_cache_hash(tmp_path):
    library, _, resolver, refs = sources(tmp_path)
    folder = library.papers_dir / "2609.12345v2"
    document = json.loads((folder / "document.json").read_bytes())
    document["metadata"]["arxiv_id"] = "2609.99999v1"
    raw = json.dumps(document).encode()
    (folder / "document.json").write_bytes(raw)
    notes = json.loads((folder / "notes.json").read_bytes())
    notes["sha256"] = hashlib.sha256(b"document.json\0" + raw + b"\0").hexdigest()
    (folder / "notes.json").write_text(json.dumps(notes))
    with pytest.raises(ValueError, match="identity"):
        resolver(ExperimentBrief(question="Question", references=refs).references)


def test_legacy_declared_references_require_resave_before_new_freeze(tmp_path):
    _, _, resolver, refs = sources(tmp_path)
    root = tmp_path / "records"
    old = ExperimentStore(root).save_brief({**complete_brief(), "references": refs})
    new_store = ExperimentStore(root, evidence_resolver=resolver)
    with pytest.raises(ValueError, match="Re-save"):
        new_store.freeze_plan(old["experiment_id"], 1)
    new_store.save_brief(old["brief"], old["experiment_id"], 1)
    assert new_store.freeze_plan(old["experiment_id"], 2)["evidence_receipts"]


@pytest.mark.parametrize("pointer,expected", [("/a~1b/~0key/0", False),
                                             ("/a~1b/~0key/1", 0), ("/empty", None)])
def test_json_pointer_preserves_false_zero_null_and_escaped_keys(pointer, expected):
    assert select_pointer({"a/b": {"~key": [False, 0]}, "empty": None}, pointer) is expected


@pytest.mark.parametrize("pointer", ["x", "/a/01", "/a/-1", "/a/+1", "/a/2", "/a/~2"])
def test_json_pointer_rejects_non_rfc_array_access(pointer):
    with pytest.raises(ValueError):
        select_pointer({"a": [1, 2]}, pointer)
