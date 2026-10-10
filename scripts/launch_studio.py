"""Open the packaged QASPER Agent Studio runs; live inference is opt-in."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--supplementary-run", type=Path, action="append", default=[],
                        help="additional complete saved run; repeat as needed")
    parser.add_argument("--qasper-score", type=Path, action="append", default=[],
                        help="official QASPER metric artifact for a loaded run")
    parser.add_argument(
        "--live-nebius", action="store_true",
        help="enable explicit paper investigations using the local NEBIUS_API_KEY",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")

    import uvicorn
    from dotenv import load_dotenv

    from benchmarks.envoybench.demo import create_app

    models_path = None
    if args.live_nebius:
        load_dotenv(ROOT / ".env", override=False)
        if not os.environ.get("NEBIUS_API_KEY"):
            parser.error("set NEBIUS_API_KEY in the environment or the repository .env")
        models_path = ROOT / "benchmarks/envoybench/paper-models.nebius.example.json"
    release = ROOT / "release/envoybench-v0.1"
    study = ROOT / "release/qasper-agent-study"
    supplementary_runs = list(dict.fromkeys(path.resolve() for path in args.supplementary_run))
    amended_run = study / "nebius-amended-run"
    nebius_run = amended_run if (amended_run / "manifest.json").is_file() else study / "nebius-run"
    nebius_manifest = nebius_run / "manifest.json"
    if nebius_manifest.is_file():
        manifest = json.loads(nebius_manifest.read_text(encoding="utf-8"))
        if (manifest.get("status") in {"complete", "incomplete"}
                and nebius_run not in supplementary_runs):
            supplementary_runs.append(nebius_run)
    qasper_scores = list(dict.fromkeys(path.resolve() for path in args.qasper_score))
    score_names = ["qwen-official-score.json"]
    if amended_run in supplementary_runs:
        score_names.append("nemotron-official-score.json")
    for name in score_names:
        path = study / name
        if path.is_file() and path not in qasper_scores:
            qasper_scores.append(path)
    app = create_app(
        run_dir=release / "run",
        review_dir=release / "review-prepared",
        judged_review_path=release / "review-model-assisted/review.json",
        token_diagnostic_run=release / "token-diagnostic-smoke",
        supplementary_runs=supplementary_runs, qasper_scores=qasper_scores,
        provisional=True,
        paper_models_path=models_path,
    )
    print(f"QASPER Agent Studio: http://127.0.0.1:{args.port}", flush=True)
    print("Opening runs makes no model calls. Live investigations require clicking Run.")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
