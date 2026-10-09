"""Launch a disposable, local-only Envoy demo with linked synthetic notes.

This launcher accepts no vault path or model configuration. It copies only the
public sample fixture, keeps application state outside that copy, and starts the
evidence-only web app. Changes made during the demo are removed on exit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import tempfile
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.personal_memory_web import create_app  # noqa: E402

SAMPLE_VAULT = ROOT / "data" / "product_memory" / "linked_demo_vault"


def prepare_workspace(workspace: Path) -> tuple[Path, Path]:
    """Copy the bundled fixture and place mutable server state beside it."""
    if not SAMPLE_VAULT.is_dir() or SAMPLE_VAULT.is_symlink():
        raise ValueError("bundled sample vault is missing or unsafe")
    if any(path.is_symlink() or (path.is_file() and path.suffix != ".md")
           for path in SAMPLE_VAULT.rglob("*")):
        raise ValueError("bundled sample vault must contain only Markdown files")
    vault = workspace / "vault"
    state = workspace / "state"
    shutil.copytree(SAMPLE_VAULT, vault)
    state.mkdir()
    return vault, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    def stop_on_term(_signum, _frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, stop_on_term)
    try:
        with tempfile.TemporaryDirectory(prefix="envoy-sample-demo-") as temp:
            vault, state = prepare_workspace(Path(temp))
            app = create_app(vault, state)
            print(json.dumps({
                "url": f"http://127.0.0.1:{args.port}",
                "mode": "evidence_only",
                "vault": str(vault),
                "state_dir": str(state),
                "temporary": True,
                "message": "Synthetic demo data; no remote model calls. Ctrl-C removes this copy.",
            }), flush=True)
            uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
