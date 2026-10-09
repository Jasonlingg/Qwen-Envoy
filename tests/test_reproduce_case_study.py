"""The public recount works without installed packages and fails on changed inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.reproduce_case_study import (
    DEFAULT_RELEASE,
    main,
    recorded_error_episode,
)


def test_packaged_counts_reproduce_without_site_packages(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/reproduce_case_study.py"
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(script), "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    assert report["status"] == "verified"
    assert report["models"] == {
        "qwen_base": {
            "questions": 40,
            "provisional_passes": 5,
            "answerable_questions": 20,
            "answerable_passes": 3,
            "unanswerable_questions": 20,
            "correct_abstentions": 2,
            "recorded_error_episodes": 22,
            "submissions": 29,
        },
        "qwen_v5": {
            "questions": 40,
            "provisional_passes": 15,
            "answerable_questions": 20,
            "answerable_passes": 3,
            "unanswerable_questions": 20,
            "correct_abstentions": 12,
            "recorded_error_episodes": 1,
            "submissions": 40,
        },
    }
    assert report["checks"]["source_corpus_and_spans_revalidated"] is False
    assert report["provenance"]["review_kind"] == "model_assisted"
    assert report["provenance"]["source_reference_status"] == "unreviewed_qasper"
    assert len(report["provenance"]["question_ids"]) == 40
    assert "no new independent validation" in report["limitations"][0]


@pytest.mark.parametrize("rewrite_checksum", [False, True])
def test_modified_results_fail_even_with_rewritten_checksum(
    tmp_path: Path,
    capsys,
    rewrite_checksum: bool,
) -> None:
    release = tmp_path / "release"
    shutil.copytree(DEFAULT_RELEASE, release)
    path = release / "run/results.json"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    results = json.loads(path.read_text())
    results[0]["predicted_answer"] = "tampered answer"
    path.write_text(json.dumps(results))
    if rewrite_checksum:
        sums = release / "SHA256SUMS"
        sums.write_text(
            sums.read_text().replace(before, hashlib.sha256(path.read_bytes()).hexdigest())
        )
    assert main(["--release-dir", str(release), "--json"]) == 1
    failure = json.loads(capsys.readouterr().err)
    assert failure["status"] == "failed"
    assert "hash" in failure["error"].lower() or "sha256" in failure["error"].lower()


@pytest.mark.parametrize("change", ["omit", "missing", "escape"])
def test_incomplete_or_unsafe_manifest_fails(tmp_path: Path, capsys, change: str) -> None:
    release = tmp_path / "release"
    shutil.copytree(DEFAULT_RELEASE, release)
    sums = release / "SHA256SUMS"
    if change == "omit":
        sums.write_text(
            "\n".join(
                line
                for line in sums.read_text().splitlines()
                if not line.endswith("run/results.json")
            )
            + "\n"
        )
    elif change == "missing":
        (release / "run/results.json").unlink()
    else:
        sums.write_text(sums.read_text() + "0" * 64 + "  ../outside.json\n")
    assert main(["--release-dir", str(release)]) == 1
    assert json.loads(capsys.readouterr().err)["status"] == "failed"


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        ('{"text": "TypeError: syntax errors and timeouts occurred in the experiment."}', False),
        ('{"text": "passage", "metadata": {"error": "a word in nested metadata"}}', False),
        ('[{"error": "unknown document"}]\n\n[Step 1/15]', True),
        ("{'error': 'unknown document'}", True),
        ("\nSTDERR:\nTraceback (most recent call last):\nTypeError: wrong argument", True),
        ("ERROR: Execution timed out after 30s", True),
        ('{"error": "truncated', False),
    ],
)
def test_error_union_does_not_classify_retrieved_prose(observation: str, expected: bool) -> None:
    assert (
        recorded_error_episode(
            {
                "status": "submitted",
                "trajectory": [{"observation": observation}],
            }
        )
        is expected
    )


def test_endpoint_failure_counts_without_recorded_turns() -> None:
    assert recorded_error_episode({"status": "error", "trajectory": []}) is True
