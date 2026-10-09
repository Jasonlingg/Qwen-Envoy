"""Run pinned base Qwen3-8B and existing v5 LoRA on the synthetic memory fixture.

GPU and model downloads are required. This trusted eval executes model-authored Python
and must not be exposed to arbitrary public users without a fail-closed sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.personal_memory_eval import QUESTIONS, baseline, diagnose, freeze  # noqa: E402
from src.env.reward import REWARD_VERSION, REWARD_WEIGHTS  # noqa: E402
from src.eval.artifacts import configuration_hash, content_hash  # noqa: E402
from src.eval.research_review import prepare_review  # noqa: E402
from src.policies.code_execution import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    SYSTEM_PROMPT,
)

BASE_ID = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
V5_ID = "jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5"
V5_REVISION = "27b912a863ff914ad45baa03624d8911dc1e17fb"
V5_SUBFOLDER = "artifacts/full/checkpoint-50"
PROMPT = ROOT / "data/product_memory/prompt_v1.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_weights(base_path: Path | None, adapter_path: Path | None) -> tuple[Path, Path, dict]:
    pinned_base_download = base_path is None
    pinned_adapter_download = adapter_path is None
    if base_path is None or adapter_path is None:
        from huggingface_hub import hf_hub_download, snapshot_download

    if base_path is None:
        base_path = Path(snapshot_download(repo_id=BASE_ID, revision=BASE_REVISION))
    if adapter_path is None:
        config = Path(
            hf_hub_download(
                V5_ID, "adapter_config.json", subfolder=V5_SUBFOLDER, revision=V5_REVISION
            )
        )
        weight = Path(
            hf_hub_download(
                V5_ID, "adapter_model.safetensors", subfolder=V5_SUBFOLDER, revision=V5_REVISION
            )
        )
        if config.parent != weight.parent:
            raise ValueError("v5 adapter config and weights landed in different folders")
        adapter_path = weight.parent
    base_path = base_path.expanduser().resolve(strict=True)
    adapter_path = adapter_path.expanduser().resolve(strict=True)
    weight = adapter_path / "adapter_model.safetensors"
    if not (base_path / "config.json").is_file() or not weight.is_file():
        raise ValueError("base config or v5 adapter weights are missing")
    identity = {
        "base_model": BASE_ID,
        "base_revision": BASE_REVISION,
        "base_path": str(base_path),
        "base_path_is_pinned_download": pinned_base_download,
        "adapter_model": V5_ID,
        "adapter_revision": V5_REVISION,
        "adapter_subfolder": V5_SUBFOLDER,
        "adapter_path": str(adapter_path),
        "adapter_path_is_pinned_download": pinned_adapter_download,
        "adapter_sha256": _sha256(weight),
        "local_override_note": (
            "Provided local paths must be checked against the pinned IDs manually."
        ),
    }
    return base_path, adapter_path, identity


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid JSON artifact: {path}") from exc


def _write_new(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def _prompt_hash() -> str:
    prompt = SYSTEM_PROMPT.rstrip() + "\n\n" + PROMPT.read_text().strip() + "\n"
    return hashlib.sha256(prompt.encode()).hexdigest()


def _recipe(snapshot: Path, identity: dict) -> dict:
    questions = _load_json(QUESTIONS)["questions"]
    return {
        "snapshot": str(snapshot),
        "questions_sha256": content_hash(QUESTIONS),
        "question_ids": [item["id"] for item in questions],
        "corpus_sha256": content_hash(snapshot / "corpus"),
        "system_prompt_sha256": _prompt_hash(),
        "max_steps": 15,
        "seed": 42,
        "require_evidence": True,
        "observation_preamble": True,
        "vector_index": False,
        "system_prompt_suffix": str(PROMPT),
        "search_within_top_k": 3,
        "search_within_mode": "raw",
        "force_known_paper_read": False,
        "workers": 1,
        "evidence_verifier": True,
        "verifier_feedback_budget": 4,
        "escalate_after_verifier_failure": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "reward_version": REWARD_VERSION,
        "split": str(QUESTIONS),
        "model_identity": identity,
    }


def _require_equal(actual: object, expected: object, description: str) -> None:
    if actual != expected:
        raise ValueError(f"resume artifact {description} does not match the pinned run")


def _protocol(recipe: dict) -> dict:
    fields = (
        "question_ids",
        "questions_sha256",
        "corpus_sha256",
        "max_steps",
        "seed",
        "reward_version",
        "workers",
        "require_evidence",
        "observation_preamble",
        "vector_index",
        "system_prompt_sha256",
        "system_prompt_suffix",
        "search_within_top_k",
        "search_within_mode",
        "force_known_paper_read",
        "evidence_verifier",
        "verifier_feedback_budget",
        "escalate_after_verifier_failure",
    )
    decoding = json.dumps(
        {
            "max_tokens": recipe["max_tokens"],
            "temperature": recipe["temperature"],
            "top_p": recipe["top_p"],
        },
        sort_keys=True,
    )
    return {**{key: recipe[key] for key in fields}, "decoding": [decoding]}


def _validate_model_run(
    path: Path, *, label: str, recipe: dict, base_path: Path, adapter_path: Path
) -> dict:
    """A nonzero child exit is acceptable only after all named episodes were saved."""
    manifest = _load_json(path.with_suffix(".manifest.json"))
    rows = _load_json(path)
    if not isinstance(manifest, dict) or not isinstance(rows, list):
        raise ValueError(f"malformed model artifacts: {path}")
    for key in (
        "question_ids",
        "questions_sha256",
        "corpus_sha256",
        "system_prompt_sha256",
        "max_steps",
        "seed",
        "require_evidence",
        "observation_preamble",
        "vector_index",
        "system_prompt_suffix",
        "search_within_top_k",
        "search_within_mode",
        "force_known_paper_read",
        "workers",
        "evidence_verifier",
        "verifier_feedback_budget",
        "escalate_after_verifier_failure",
        "reward_version",
        "split",
    ):
        _require_equal(manifest.get(key), recipe[key], f"{label} {key}")
    _require_equal(manifest.get("run_label"), label, f"{label} run label")
    _require_equal(manifest.get("reward_weights"), REWARD_WEIGHTS, f"{label} reward weights")
    _require_equal(
        manifest.get("comparison_id"),
        configuration_hash(_protocol(recipe)),
        f"{label} comparison ID",
    )
    _require_equal(manifest.get("base_model"), str(base_path), f"{label} base model")
    expected_checkpoint = str(adapter_path) if label == "v5" else None
    _require_equal(manifest.get("checkpoint_id"), expected_checkpoint, f"{label} checkpoint")
    policy = "qwen_sft_policy" if label == "v5" else "qwen_base_policy"
    policies = manifest.get("policy_settings")
    settings = policies.get(policy) if isinstance(policies, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f"missing {label} policy settings: {path}")
    for key, expected in (
        ("model", str(base_path)),
        ("checkpoint", expected_checkpoint),
        ("max_tokens", recipe["max_tokens"]),
        ("temperature", recipe["temperature"]),
        ("top_p", recipe["top_p"]),
    ):
        if key == "checkpoint" and label == "base" and key not in settings:
            continue
        _require_equal(settings.get(key), expected, f"{label} policy {key}")
    _require_equal(manifest.get("decoding"), _protocol(recipe)["decoding"], f"{label} decoding")
    expected_questions = _load_json(QUESTIONS)["questions"]
    if len(rows) != len(expected_questions):
        raise ValueError(f"incomplete {label} transcript: {len(rows)} rows")
    expected_by_id = {item["id"]: item["question"] for item in expected_questions}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"malformed {label} transcript row")
        question_id = row.get("question_id")
        if question_id not in expected_by_id or question_id in seen:
            raise ValueError(f"missing, unknown, or duplicate {label} question ID")
        seen.add(question_id)
        for key, expected in (
            ("question", expected_by_id[question_id]),
            ("policy", policy),
            ("run_label", label),
            ("run_id", manifest.get("run_id")),
            ("comparison_id", manifest.get("comparison_id")),
            ("checkpoint_id", expected_checkpoint),
        ):
            _require_equal(row.get(key), expected, f"{label} row {key}")
        if not isinstance(row.get("status"), str) or not row["status"]:
            raise ValueError(f"missing {label} episode status")
    if not manifest.get("run_id") or not manifest.get("comparison_id"):
        raise ValueError(f"missing {label} run identity")
    return manifest


def _run_model(
    command: list[str],
    path: Path,
    *,
    label: str,
    recipe: dict,
    base_path: Path,
    adapter_path: Path,
    env: dict[str, str],
) -> dict:
    try:
        _run(command, env=env)
    except subprocess.CalledProcessError:
        # run_eval.py exits 1 when any episode has status=error, after saving
        # every completed row. Keep that failure visible and run the paired arm.
        manifest = _validate_model_run(
            path,
            label=label,
            recipe=recipe,
            base_path=base_path,
            adapter_path=adapter_path,
        )
        print(
            f"{label}: nonzero child exit, but all episodes and manifest were saved; "
            "continuing with the paired arm",
            flush=True,
        )
        return manifest
    return _validate_model_run(
        path,
        label=label,
        recipe=recipe,
        base_path=base_path,
        adapter_path=adapter_path,
    )


def _validate_baseline(path: Path, recipe: dict, snapshot: Path) -> None:
    manifest = _load_json(path.with_suffix(".manifest.json"))
    rows = _load_json(path)
    if not isinstance(manifest, dict) or not isinstance(rows, list):
        raise ValueError("malformed BM25 artifacts")
    for key in ("question_ids", "questions_sha256", "corpus_sha256"):
        _require_equal(manifest.get(key), recipe[key], f"BM25 {key}")
    _require_equal(manifest.get("run_label"), "bm25_passage_baseline", "BM25 label")
    with tempfile.TemporaryDirectory() as folder:
        expected_path = Path(folder) / "bm25.json"
        baseline(snapshot, expected_path)
        expected_manifest = _load_json(expected_path.with_suffix(".manifest.json"))
        expected_rows = _load_json(expected_path)
    _require_equal(manifest, expected_manifest, "BM25 manifest")
    if len(rows) != len(expected_rows) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("BM25 transcript is incomplete or malformed")
    for row, expected in zip(rows, expected_rows):
        _require_equal(
            {key: value for key, value in row.items() if key != "duration_seconds"},
            {key: value for key, value in expected.items() if key != "duration_seconds"},
            "BM25 transcript",
        )
        if not isinstance(row.get("duration_seconds"), (int, float)):
            raise ValueError("BM25 duration is missing or malformed")


def _validate_diagnostics(path: Path, snapshot: Path, runs: list[Path]) -> None:
    actual = _load_json(path)
    with tempfile.TemporaryDirectory() as folder:
        expected_path = Path(folder) / "diagnostics.json"
        diagnose(QUESTIONS, snapshot / "corpus", runs, expected_path)
        expected = _load_json(expected_path)
    _require_equal(actual, expected, "diagnostics")


def _validate_review(path: Path, snapshot: Path, runs: list[Path]) -> None:
    expected_review, expected_key, expected_auto = prepare_review(
        QUESTIONS, snapshot / "corpus", runs, 42
    )
    key = _load_json(path / "blind-key.json")
    auto = _load_json(path / "automatic.json")
    review = _load_json(path / "review.json")
    _require_equal(key, expected_key, "blind key")
    _require_equal(auto, expected_auto, "automatic review metrics")
    if not isinstance(review, dict) or not isinstance(review.get("rows"), list):
        raise ValueError("malformed blind review")
    _require_equal(len(review["rows"]), len(expected_review["rows"]), "blind review count")
    for actual_row, expected_row in zip(review["rows"], expected_review["rows"]):
        if not isinstance(actual_row, dict):
            raise ValueError("malformed blind review row")
        for key, value in expected_row.items():
            if key not in {"verdict", "notes"}:
                _require_equal(actual_row.get(key), value, f"blind review {key}")
    if not (path / "review.md").is_file():
        raise ValueError("blind review markdown is missing")


def run(
    snapshot: Path,
    output: Path,
    base_path: Path | None,
    adapter_path: Path | None,
    *,
    resume: bool = False,
) -> dict:
    snapshot = snapshot.expanduser().resolve()
    output = output.expanduser().resolve()
    freeze(snapshot)
    if output.exists() and not resume:
        raise ValueError(f"output already exists: {output}")
    base_path, adapter_path, identity = _resolve_weights(base_path, adapter_path)
    recipe = _recipe(snapshot, identity)
    if resume:
        if not output.is_dir():
            raise ValueError(f"no output directory to resume: {output}")
        _require_equal(_load_json(output / "model-identity.json"), identity, "model identity")
        _require_equal(_load_json(output / "run-config.json"), recipe, "run configuration")
    else:
        output.mkdir(parents=True)
        _write_new(output / "model-identity.json", identity)
        _write_new(output / "run-config.json", recipe)

    shared = [
        "--questions",
        str(QUESTIONS),
        "--corpus",
        str(snapshot / "corpus"),
        "--max-steps",
        "15",
        "--seed",
        "42",
        "--require-evidence",
        "--no-vector-index",
        "--evidence-verifier",
        "--verifier-feedback-budget",
        "4",
        "--escalate-after-verifier-failure",
        "--system-prompt-suffix",
        str(PROMPT),
    ]
    env = {
        **os.environ,
        "BASE_MODEL_PATH": str(base_path),
        "ENVOY_QWEN_TEMPERATURE": "0",
        "ENVOY_QWEN_TOP_P": "1",
    }
    base_run = output / "base.json"
    v5_run = output / "v5.json"
    if not base_run.exists() and not base_run.with_suffix(".manifest.json").exists():
        _run_model(
            [
                sys.executable,
                "scripts/run_eval.py",
                "--policy",
                "qwen_base_policy",
                *shared,
                "--run-label",
                "base",
                "--output",
                str(base_run),
            ],
            path=base_run,
            label="base",
            recipe=recipe,
            base_path=base_path,
            adapter_path=adapter_path,
            env=env,
        )
    else:
        if not resume:
            raise ValueError(f"unexpected existing base artifacts: {base_run}")
        _validate_model_run(
            base_run, label="base", recipe=recipe, base_path=base_path, adapter_path=adapter_path
        )
        print("base: validated existing complete transcript; skipping model run", flush=True)
    env["CHECKPOINT_PATH"] = str(adapter_path)
    if not v5_run.exists() and not v5_run.with_suffix(".manifest.json").exists():
        _run_model(
            [
                sys.executable,
                "scripts/run_eval.py",
                "--policy",
                "qwen_sft_policy",
                *shared,
                "--run-label",
                "v5",
                "--output",
                str(v5_run),
            ],
            path=v5_run,
            label="v5",
            recipe=recipe,
            base_path=base_path,
            adapter_path=adapter_path,
            env=env,
        )
    else:
        if not resume:
            raise ValueError(f"unexpected existing v5 artifacts: {v5_run}")
        _validate_model_run(
            v5_run, label="v5", recipe=recipe, base_path=base_path, adapter_path=adapter_path
        )
        print("v5: validated existing complete transcript; skipping model run", flush=True)

    search_run = output / "bm25.json"
    if search_run.exists() or search_run.with_suffix(".manifest.json").exists():
        if not resume:
            raise ValueError(f"unexpected existing BM25 artifacts: {search_run}")
        _validate_baseline(search_run, recipe, snapshot)
    else:
        baseline(snapshot, search_run)
        _validate_baseline(search_run, recipe, snapshot)
    runs = [search_run, base_run, v5_run]
    diagnostics_path = output / "diagnostics.json"
    if diagnostics_path.exists():
        if not resume:
            raise ValueError(f"unexpected existing diagnostics: {diagnostics_path}")
        _validate_diagnostics(diagnostics_path, snapshot, runs)
    else:
        diagnose(QUESTIONS, snapshot / "corpus", runs, diagnostics_path)
    review_path = output / "blind-review"
    if review_path.exists():
        if not resume:
            raise ValueError(f"unexpected existing blind review: {review_path}")
        _validate_review(review_path, snapshot, [base_run, v5_run])
    else:
        _run(
            [
                sys.executable,
                "scripts/review_code_exec_pilot.py",
                "prepare",
                "--questions",
                str(QUESTIONS),
                "--corpus",
                str(snapshot / "corpus"),
                "--run",
                str(base_run),
                "--run",
                str(v5_run),
                "--output",
                str(output / "blind-review"),
                "--seed",
                "42",
            ],
            env=env,
        )
        _validate_review(review_path, snapshot, [base_run, v5_run])
    return {
        "output": str(output),
        "questions": 10,
        "human_answer_support": "pending blind review",
        "public_execution_safe": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument(
        "--resume", action="store_true", help="Validate and reuse complete artifacts"
    )
    args = parser.parse_args()
    try:
        print(
            "Trusted GPU evaluation only: model-authored Python may use a local process.",
            flush=True,
        )
        print(
            json.dumps(
                run(
                    args.snapshot,
                    args.output,
                    args.base_path,
                    args.adapter_path,
                    resume=args.resume,
                ),
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
