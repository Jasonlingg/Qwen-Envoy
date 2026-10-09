"""Launch a disposable local Envoy demo from the real public Qwen report.

The three Markdown notes are generated from the repository's public report. The app runs
without a model client, so opening the demo makes no GPU or API calls. Any
notes approved during the demo live only in the temporary vault.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import tempfile
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_qwen_experiment_vault import DEFAULT_REPORT, build_vault  # noqa: E402
from scripts.personal_memory_web import create_app  # noqa: E402
from src.product.chat import ChatService  # noqa: E402


def prepare_workspace(workspace: Path) -> tuple[Path, Path, dict[str, Path]]:
    """Build a fresh vault and keep mutable app state outside it."""
    vault = workspace / "vault"
    notes = build_vault(vault)
    state = workspace / "state"
    state.mkdir()
    return vault, state, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    def stop_on_term(_signum, _frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, stop_on_term)
    try:
        with tempfile.TemporaryDirectory(prefix="envoy-qwen-report-demo-") as temp:
            vault, state, notes = prepare_workspace(Path(temp))
            app = create_app(vault, state, chat_service=ChatService())
            print(json.dumps({
                "url": f"http://127.0.0.1:{args.port}",
                "mode": "evidence_only",
                "vault": str(vault),
                "state_dir": str(state),
                "report": str(DEFAULT_REPORT.relative_to(ROOT)),
                "report_sha256": hashlib.sha256(DEFAULT_REPORT.read_bytes()).hexdigest(),
                "result_source_path": notes["result"].relative_to(vault).as_posix(),
                "temporary": True,
                "model_calls": 0,
                "message": (
                    "Real public-report-derived Qwen results; no live model calls. "
                    "Ctrl-C removes this copy."
                ),
            }), flush=True)
            uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
