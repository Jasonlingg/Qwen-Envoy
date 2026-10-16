"""Replay saved Nemotron actions in Docker; require exact V1 before comparing V2.

No model calls. Fixed-action replay cannot measure how Nemotron would respond to
new feedback. It measures protocol acceptance and tool execution, not correctness.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.envoybench.frozen_dataset import load_split  # noqa: E402
from src.env.repl import DockerREPL  # noqa: E402
from src.env.tools import TOOL_PREAMBLE  # noqa: E402
from src.eval.harness_replay import (  # noqa: E402
    ENV_SHA256,
    HISTORICAL_COMMIT,
    STATE_SHA256,
    compact_report,
    historical_environment,
    inspect_image,
    replay_baseline,
    sha256,
)
from src.eval.harness_replay_checkpoint import checkpointed_baselines  # noqa: E402
from src.policies.code_execution import QASPER_SYSTEM_PROMPT  # noqa: E402

RUN = ROOT / "release/qasper-agent-study/nebius-amended-run"
LIMITATION = (
    "Fixed-action replay cannot measure how Nemotron would respond to new feedback. "
    "It tests the same saved actions, not a model policy, answer quality, or fine-tuned "
    "Qwen improvement. Originally rejected verifier actions stay skipped; submission "
    "syntax is checked without regrading answers or recomputing verifier decisions."
)


def classify_observation(text: str) -> str | None:
    if "Usage: " in text and "TypeError" in text:
        return "usage_message"
    match = re.search(r"\b(\w+Error)\b", text)
    if match and ("Traceback" in text or "SyntaxError" in text):
        return match.group(1)
    if "ERROR:" in text or "'error':" in text or '"error":' in text:
        return "tool_error"
    return None


def compare_v2(questions: list[dict], corpus: Path, image: str, *, workers: int = 1) -> dict:
    """Apply the shim only after every saved V1 action matches, with no host REPL."""
    if not questions or not all(q["v1_exact_match"] for q in questions):
        raise ValueError("V2 comparison requires an exact V1 match for every question/action")
    from src.env.tools_v2 import HARNESS_V2_PREAMBLE, HARNESS_V2_SHA256
    from src.product.submit_call import submit_call_to_line

    def run_question(question):
        counts = {"v1": Counter(), "v2": Counter()}
        repl = DockerREPL(image=image, corpus_path=str(corpus),
                          extra_preamble=HARNESS_V2_PREAMBLE)
        try:
            repl.start_session()
            for item in question["actions"]:
                action = item["action"]
                kind = classify_observation(item["v1_observation"])
                item["v1_error"] = kind
                counts["v1"][kind or "no_execution_error"] += 1
                outcome, observation = "recorded_verifier_skip", item["recorded_observation"]
                if action.strip().upper().startswith("SUBMIT:"):
                    outcome = "recorded_submission"
                    observation = "Submission unchanged; not regraded."
                elif item["v1_executed"]:
                    try:
                        converted = submit_call_to_line(action)
                    except ValueError as exc:
                        outcome, observation = "submit_call_rejected", str(exc)
                    else:
                        if converted is not None:
                            outcome, observation = "submit_call_converted", converted
                        else:
                            try:
                                ast.parse(action)
                            except SyntaxError:
                                outcome, observation = "format_retry", "Would request a new action."
                            else:
                                outcome, observation = "executed", repl.execute(action, timeout=30)
                error = classify_observation(observation) if outcome == "executed" else None
                item.update(v2_outcome=outcome, v2_observation=observation, v2_error=error)
                counts["v2"][error or outcome] += 1
        finally:
            repl.kill_session()
        question["counts"] = {
            name: dict(Counter(action.get(name + "_error") or action.get(
                name + "_outcome", "no_execution_error") for action in question["actions"]))
            for name in ("v1", "v2")}
        print(f"V2 {question['question_id']}: {question['counts']['v2']}", flush=True)
        return counts

    counts = {"v1": Counter(), "v2": Counter()}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(run_question, questions):
            for name in counts:
                counts[name].update(result[name])
    return {"counts": {key: dict(value) for key, value in counts.items()},
            "harness_shim_sha256": HARNESS_V2_SHA256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/envoybench/data")
    parser.add_argument("--output", type=Path, default=ROOT / "out/harness-v2-replay.json")
    parser.add_argument("--limit", type=int, help="Question subset; marked incomplete in report")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse identity-bound V1 questions")
    parser.add_argument("--summary-output", type=Path, help="Report with observation hashes")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    rows = json.loads((args.run / "results.json").read_text())
    if args.limit is not None and not 1 <= args.limit <= len(rows):
        parser.error("--limit must be between 1 and the recorded question count")
    benchmark, _, benchmark_path, corpus, _ = load_split(args.dataset, manifest["split"])
    if benchmark["corpus_hash"] != manifest["corpus_hash"]:
        raise ValueError("Corpus does not match the recorded run")
    if ([row["question_id"] for row in rows] != manifest["question_ids"]
            or sha256(TOOL_PREAMBLE) != manifest["tool_preamble_sha256"]
            or sha256(QASPER_SYSTEM_PROMPT) != manifest["system_prompt_sha256"]):
        raise ValueError("Recorded question/tool/prompt identities differ")
    image = manifest["sandbox_image"]["image_id"]
    sandbox = inspect_image(image)
    selected = rows[:args.limit] if args.limit else rows
    modules = [historical_environment(ROOT, manifest) for _ in selected]
    questions = {question["id"]: question for question in benchmark["questions"]}
    report = {
        "schema_version": "harness-v2-fixed-action-replay-v1", "model_calls": 0,
        "hypothesis": "V2 accepts more recorded protocol mistakes without changing V1.",
        "expected_signal": "Fewer execution errors for the same saved actions.",
        "decision_rule": (
            "Require every V1 action to match; report V2 outcomes without model-quality claims."),
        "limitation": LIMITATION, "sandbox": sandbox, "normalization": "none",
        "run_id": manifest["run_id"], "corpus_hash": manifest["corpus_hash"],
        "system_prompt_sha256": manifest["system_prompt_sha256"],
        "tool_preamble_sha256": manifest["tool_preamble_sha256"],
        "historical_commit": HISTORICAL_COMMIT, "environment_sha256": ENV_SHA256,
        "evidence_state_sha256": STATE_SHA256, "seed": manifest["seed"],
        "historical_fail_open_verifier": not manifest["escalate_after_verifier_failure"],
        "recorded_models": manifest["models"], "runner_hardware": platform.platform(),
        "results_sha256": hashlib.sha256((args.run / "results.json").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((args.run / "manifest.json").read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "full_recorded_run": len(selected) == len(rows), "questions": [], "v2_compared": False,
        "workers": args.workers,
        "replay_implementation_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("scripts/replay_harness_v2.py", "src/eval/harness_replay.py",
                         "src/eval/harness_replay_checkpoint.py", "src/env/corpus.py",
                         "src/policies/code_execution.py", "src/eval/abstention.py")},
    }

    def baseline(pair):
        row, module = pair
        result = replay_baseline(
            module, row, questions[row["question_id"]], corpus, manifest, image)
        print(f"V1 {row['question_id']}: exact={result['v1_exact_match']}", flush=True)
        return result

    report["questions"], report["v1_checkpoint_reused"] = checkpointed_baselines(
        selected, modules, baseline, checkpoint=args.output.with_suffix(".v1-checkpoint.json"),
        identity=report, workers=args.workers, resume=args.resume)
    report["v1_matches_recorded"] = all(q["v1_exact_match"] for q in report["questions"])
    if report["v1_matches_recorded"] and not args.baseline_only:
        report.update(compare_v2(report["questions"], corpus, image, workers=args.workers),
                      v2_compared=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(compact_report(report), indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "v1_matches_recorded", "v2_compared", "full_recorded_run")}), flush=True)
    if not report["v1_matches_recorded"]:
        raise SystemExit("V1 did not reproduce the recorded run; V2 comparison blocked")


if __name__ == "__main__":
    main()
