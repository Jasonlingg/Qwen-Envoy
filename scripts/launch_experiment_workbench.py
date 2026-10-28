"""Open a localhost workbench for saved experiment evidence and decisions.

No model, GPU, paper download, or paid provider is used by this view.
Historical atlas entries are read-only; decisions bind to experiment-store records.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.product.decision_threads import DecisionThreads  # noqa: E402
from src.product.experiment_atlas import ExperimentAtlas  # noqa: E402
from src.product.ml_experiment_store import ExperimentStore  # noqa: E402

PAGE = ROOT / "src" / "product" / "experiment_workbench.html"


class NewThread(BaseModel):
    experiment_id: str
    question: str = Field(min_length=3, max_length=500)


class NewDecision(BaseModel):
    thread_id: str
    expected_revision: int = Field(ge=1)
    experiment_id: str
    run_id: str | None = None
    research_note_id: str | None = None
    outcome: str
    rationale: str = Field(min_length=1, max_length=4_000)
    revisit_when: str | None = Field(default=None, max_length=4_000)
    author_kind: Literal["agent_proposed", "user_reported"] = "agent_proposed"


def create_app(store_root: Path, *, atlas_root: Path = ROOT) -> FastAPI:
    store = ExperimentStore(store_root)
    decisions = DecisionThreads(store.files.root, store)
    atlas = ExperimentAtlas(atlas_root)
    app = FastAPI(title="Envoy Experiment Workbench", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.headers.get("host", "").split(":", 1)[0].lower()
        if host not in {"127.0.0.1", "localhost"}:
            return JSONResponse({"detail": "Localhost only"}, status_code=403)
        if request.method not in {"GET", "HEAD"}:
            origin = request.headers.get("origin")
            expected = f"{request.url.scheme}://{request.headers.get('host')}"
            if origin != expected:
                return JSONResponse({"detail": "Same-origin request required"}, status_code=403)
            length = request.headers.get("content-length", "")
            if not length.isdecimal():
                return JSONResponse({"detail": "Content-Length required"}, status_code=411)
            if int(length) > 16_384:
                return JSONResponse({"detail": "Request too large"}, status_code=413)
        return await call_next(request)

    def checked(action):
        try:
            return action()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/")
    def home():
        return FileResponse(PAGE, media_type="text/html",
                            headers={"Cache-Control": "no-store",
                                     "Content-Security-Policy": "default-src 'self'; "
                                     "script-src 'self'; style-src 'self'; "
                                     "connect-src 'self'; object-src 'none'; base-uri 'none'"})

    @app.get("/experiment_workbench.css")
    def stylesheet():
        return FileResponse(PAGE.with_suffix(".css"), media_type="text/css")

    @app.get("/experiment_workbench.js")
    def javascript():
        return FileResponse(PAGE.with_suffix(".js"), media_type="text/javascript")

    @app.get("/api/atlas")
    def search_atlas(q: str | None = None):
        return checked(lambda: atlas.find(q))

    @app.get("/api/atlas/{atlas_id}")
    def get_atlas(atlas_id: str):
        return checked(lambda: atlas.get(atlas_id))

    @app.get("/api/atlas/{atlas_id}/source/{source_index}")
    def read_atlas_source(atlas_id: str, source_index: int, start: int = 0):
        return checked(lambda: atlas.read_source(atlas_id, source_index, start))

    @app.get("/api/experiments")
    def search_experiments(q: str | None = None):
        if q:
            return checked(lambda: store.find_related(q, limit=10))
        return checked(lambda: store.list(limit=20))

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str, run_id: str | None = None,
                       research_note_id: str | None = None):
        return checked(lambda: store.get(experiment_id, run_id, research_note_id))

    @app.get("/api/decisions")
    def search_decisions(q: str):
        return checked(lambda: decisions.find(q, limit=10))

    @app.get("/api/decisions/{thread_id}")
    def get_decision(thread_id: str):
        return checked(lambda: decisions.get(thread_id))

    @app.post("/api/decisions")
    def create_decision_thread(body: NewThread):
        return checked(lambda: decisions.create(body.question, body.experiment_id))

    @app.post("/api/decisions/record")
    def record_decision(body: NewDecision):
        basis = {"experiment_id": body.experiment_id, "run_id": body.run_id,
                 "research_note_id": body.research_note_id}
        choice = {"decision_id": f"choice_{uuid4().hex}", "outcome": body.outcome,
                  "rationale": body.rationale, "revisit_when": body.revisit_when,
                  "author_kind": body.author_kind, "basis": [basis]}
        return checked(lambda: decisions.record(body.thread_id,
                                                 body.expected_revision, choice))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path,
                        default=Path.home() / ".envoy" / "ml-experiments")
    parser.add_argument("--port", type=int, default=8976)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    import uvicorn

    print(f"Envoy Experiment Workbench: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(create_app(args.store_root), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
