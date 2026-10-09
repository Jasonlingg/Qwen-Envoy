"""Open the packaged EnvoyBench runs locally; live inference is opt-in."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
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
    app = create_app(
        run_dir=release / "run",
        review_dir=release / "review-prepared",
        judged_review_path=release / "review-model-assisted/review.json",
        token_diagnostic_run=release / "token-diagnostic-smoke",
        provisional=True,
        paper_models_path=models_path,
    )
    print(f"EnvoyBench Studio: http://127.0.0.1:{args.port}", flush=True)
    print("Opening runs makes no model calls. Live investigations require clicking Run.")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
