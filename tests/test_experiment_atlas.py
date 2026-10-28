"""Curated old runs are searchable and source-readable without model inference."""

import hashlib
from pathlib import Path

import pytest

from src.product.experiment_atlas import ExperimentAtlas

ROOT = Path(__file__).resolve().parents[1]


def test_atlas_finds_prior_runs_and_reads_exact_listed_source():
    atlas = ExperimentAtlas(ROOT)
    all_rows = atlas.find()
    assert all_rows["total_entries"] == 11
    assert len(all_rows["matches"]) == 11
    matches = atlas.find("hybrid retrieval")
    assert "hybrid-retrieval-offline" in {m["atlas_id"] for m in matches["matches"]}
    entry = atlas.get("reader-rl-pilot")
    assert "nine verified updates" in entry["observation"]
    assert "1/10 complete, source-supported answers in each arm" in entry["observation"]
    assert "not established" in entry["decision"]
    source = atlas.read_source("reader-rl-pilot", 0, length=1_000)
    raw = (ROOT / source["path"]).read_bytes()
    assert source["sha256"] == hashlib.sha256(raw).hexdigest()
    assert source["text"] == raw[:1_000].decode()
    assert source["truncated"]
    later = atlas.read_source("reader-rl-pilot", 0, start=1_000, length=500)
    assert later["text"] == raw[1_000:1_500].decode()


@pytest.mark.parametrize("args", [
    ("../escape", 0, 0, 4_000), ("reader-rl-pilot", -1, 0, 4_000),
    ("reader-rl-pilot", 9, 0, 4_000), ("reader-rl-pilot", 0, -1, 4_000),
    ("reader-rl-pilot", 0, 0, 8_001), ("reader-rl-pilot", True, 0, 4_000),
])
def test_atlas_rejects_unlisted_or_unbounded_source(args):
    with pytest.raises(ValueError):
        ExperimentAtlas(ROOT).read_source(*args)


def test_atlas_rejects_source_outside_repo(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "EXPERIMENT_ATLAS.md").write_text(
        "| ID | Question and actual attempt | Saved observation | "
        "Decision and next question | Primary evidence |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `bad` | Question | Observation | Decision | [outside](../../outside.txt) |\n")
    with pytest.raises(ValueError, match="escapes repository"):
        ExperimentAtlas(tmp_path).find()


def test_atlas_rejects_symlinked_and_oversized_sources(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("saved")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(source)
    atlas_file = docs / "EXPERIMENT_ATLAS.md"
    header = ("| ID | Question and actual attempt | Saved observation | "
              "Decision and next question | Primary evidence |\n"
              "| --- | --- | --- | --- | --- |\n")
    atlas_file.write_text(header +
        "| `one` | Question | Observation | Decision | [source](../alias.txt) |\n")
    with pytest.raises(ValueError, match="symlinks"):
        ExperimentAtlas(tmp_path).find()

    atlas_file.write_text(header +
        "| `one` | Question | Observation | Decision | [source](../source.txt) |\n")
    with source.open("wb") as stream:
        stream.truncate(4_000_001)
    with pytest.raises(ValueError, match="bounded regular file"):
        ExperimentAtlas(tmp_path).read_source("one", 0)
