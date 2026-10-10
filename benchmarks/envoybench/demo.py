"""Local QASPER Agent Studio: source review and inspectable saved traces.

By default this shows a *historical development* Qwen comparison, not an
EnvoyBench test result. Pass --run-dir after a complete EnvoyBench evaluation to
inspect its actual saved trajectories without running any model calls.
The paper reader stores uploads and bounded live investigations separately from
benchmark results. Live inference requires an explicit operator model config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from benchmarks.envoybench.answer_review import (
    finalize_answer_review,
    load_answer_review,
    save_answer_verdict,
)
from benchmarks.envoybench.diagnostics import known_source_id
from benchmarks.envoybench.frozen_dataset import DEFAULT_DATASET, load_split
from benchmarks.envoybench.paper_api import create_paper_router
from benchmarks.envoybench.reference_review import (
    finalize_reference_review,
    load_reference_review,
    save_reference_decision,
)
from benchmarks.envoybench.saved_payload import (
    _load_web_evidence_report,
    _read_json,
    build_demo_payload,
)
from benchmarks.envoybench.score import _validated_references
from src.eval.artifacts import configuration_hash
from src.eval.research_review import load_corpus

HERE = Path(__file__).resolve().parent
DEFAULT_REFERENCE_REVIEW = HERE.parents[1] / "out/envoybench/test-reference-review.json"
DEFAULT_PAPER_STATE = HERE.parents[1] / "out/envoybench/paper-workspace"


def create_app(
    *, run_dir: Path | None = None, review_dir: Path | None = None,
    dataset: Path = DEFAULT_DATASET,
    reference_review_path: Path = DEFAULT_REFERENCE_REVIEW,
    blind_review_dir: Path | None = None,
    provisional: bool = False,
    judged_review_path: Path | None = None,
    paper_state_dir: Path = DEFAULT_PAPER_STATE,
    paper_models_path: Path | None = None,
    web_evidence_report_path: Path | None = None,
    token_diagnostic_run: Path | None = None,
    supplementary_runs: list[Path] | None = None,
    qasper_scores: list[Path] | None = None,
) -> FastAPI:
    if provisional and blind_review_dir is not None:
        raise ValueError("provisional named outputs cannot share a blind human review session")
    if token_diagnostic_run is not None and blind_review_dir is not None:
        raise ValueError("named token diagnostics cannot share a blind human review session")
    if (supplementary_runs or qasper_scores) and blind_review_dir is not None:
        raise ValueError("named saved runs and scores cannot share a blind human review session")
    app = FastAPI(title="QASPER Agent Studio")
    verified_trace_store: dict[tuple[str, ...], dict] = {}
    app.state.payload = build_demo_payload(
        run_dir=run_dir, review_dir=review_dir, dataset=dataset,
        reference_review_path=reference_review_path, provisional=provisional,
        judged_review_path=judged_review_path,
        verified_trace_store=(verified_trace_store if blind_review_dir is None else None),
        token_diagnostic_run=token_diagnostic_run,
        supplementary_runs=supplementary_runs, qasper_scores=qasper_scores,
    )
    app.state.verified_trace_store = verified_trace_store
    app.state.web_evidence_report = (
        _load_web_evidence_report(web_evidence_report_path)
        if blind_review_dir is None and web_evidence_report_path is not None else None
    )
    # Named checkpoints and live runs must not leak into a blind review session.
    if blind_review_dir is None:
        app.include_router(create_paper_router(paper_state_dir, paper_models_path))

        @app.get("/papers", response_class=HTMLResponse)
        def paper_reader() -> HTMLResponse:
            return HTMLResponse((HERE / "paper.html").read_text(encoding="utf-8"))

        @app.get("/model", response_class=HTMLResponse)
        def model_page() -> HTMLResponse:
            return HTMLResponse((HERE / "model.html").read_text(encoding="utf-8"))

        @app.get("/api/model-card", response_class=JSONResponse)
        def model_card() -> JSONResponse:
            return JSONResponse(_read_json(HERE / "model_card.json"))

        @app.get("/web-evidence", response_class=HTMLResponse)
        def web_evidence_page() -> HTMLResponse:
            return HTMLResponse((HERE / "web_evidence.html").read_text(encoding="utf-8"))

        @app.get("/api/web-evidence", response_class=JSONResponse)
        def web_evidence_data() -> JSONResponse:
            if app.state.web_evidence_report is None:
                raise HTTPException(
                    status_code=404, detail="No completed web-evidence report loaded"
                )
            return JSONResponse(app.state.web_evidence_report)

        @app.get("/api/demo/trace", response_class=JSONResponse)
        def full_saved_trace(
            question_id: str, model: str, run_id: str | None = None,
        ) -> JSONResponse:
            """One verified saved Python-turn trajectory; no inference or file lookup."""
            active = app.state.payload["active"]
            if run_id is not None:
                active = next((run for run in app.state.payload["supplementary_runs"]
                               if run["provenance"]["run_id"] == run_id), {})
            if active.get("kind") != "envoybench_run":
                raise HTTPException(status_code=404, detail="No saved run loaded")
            case = next(
                (item for item in active["cases"] if item["id"] == question_id), None
            )
            if case is None or model not in case["systems"]:
                raise HTTPException(status_code=404, detail="Unknown question or model")
            trace_key = (run_id, question_id, model) if run_id is not None else (question_id, model)
            result = app.state.verified_trace_store.get(trace_key)
            if result is None:
                raise HTTPException(status_code=404, detail="No verified trace for selection")
            return JSONResponse(result)

    def source_context() -> tuple[dict, Path]:
        try:
            benchmark, _, _, corpus_dir, _ = load_split(dataset, "test_candidate")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return benchmark, corpus_dir

    def review_response(result: dict) -> dict:
        sheet = result["review"]
        app.state.payload["candidate"]["reference_review_status"] = (
            "complete" if sheet["status"] == "complete" else "pending"
        )
        return {
            **sheet,
            "revision": result["sha256"],
            "decided_count": result["decided_count"],
            "total_count": result["total_count"],
        }

    def require_blind_human_session() -> None:
        if provisional and run_dir is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "human review cannot be recorded while provisional named outputs are visible"
                ),
            )

    def blind_context() -> tuple[dict, Path, Path]:
        if blind_review_dir is None:
            raise HTTPException(status_code=404, detail="no prepared blind review bundle selected")
        benchmark, corpus_dir = source_context()
        try:
            source_sheet = load_reference_review(
                reference_review_path, benchmark, corpus_dir
            )["review"]
            _validated_references(benchmark, source_sheet)
            review_path = blind_review_dir / "review.json"
            blind_sheet = load_answer_review(review_path, benchmark, corpus_dir)
            if blind_sheet["reference_review_hash"] != configuration_hash(source_sheet):
                raise ValueError("blind review differs from finalized source review")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return benchmark, corpus_dir, review_path

    def frozen_paper(question_id: str, benchmark: dict, corpus_dir: Path) -> dict:
        question = next(
            (item for item in benchmark["questions"] if item["id"] == question_id), None
        )
        if question is None:
            raise HTTPException(status_code=404, detail="unknown benchmark question_id")
        documents = load_corpus(corpus_dir)
        try:
            document = documents[known_source_id(question, documents)]
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "doc_id": document["doc_id"],
            "title": document.get("title", ""),
            "text": document["text"],
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((HERE / "demo.html").read_text(encoding="utf-8"))

    @app.get("/api/demo", response_class=JSONResponse)
    def demo_data() -> JSONResponse:
        return JSONResponse(app.state.payload)

    @app.get("/api/reference-review", response_class=JSONResponse)
    def reference_review_data() -> JSONResponse:
        benchmark, corpus_dir = source_context()
        try:
            result = load_reference_review(reference_review_path, benchmark, corpus_dir)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(review_response(result))

    @app.get("/api/reference-review/{question_id}/source", response_class=JSONResponse)
    def reference_source(question_id: str) -> JSONResponse:
        benchmark, corpus_dir = source_context()
        return JSONResponse(frozen_paper(question_id, benchmark, corpus_dir))

    @app.post("/api/reference-review/finalize", response_class=JSONResponse)
    def finalize_reference_data(payload: dict = Body(...)) -> JSONResponse:
        require_blind_human_session()
        benchmark, corpus_dir = source_context()
        try:
            result = finalize_reference_review(
                reference_review_path, benchmark, corpus_dir,
                payload.get("reviewer_id"),
                expected_sha256=payload.get("expected_sha256"),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(review_response(result))

    @app.post("/api/reference-review/{question_id}", response_class=JSONResponse)
    def save_reference_data(question_id: str, payload: dict = Body(...)) -> JSONResponse:
        require_blind_human_session()
        benchmark, corpus_dir = source_context()
        try:
            result = save_reference_decision(
                reference_review_path, benchmark, corpus_dir,
                question_id, payload.get("decision"), payload.get("reason"),
                payload.get("reviewer_id"),
                expected_sha256=payload.get("expected_sha256"),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(review_response(result))

    @app.get("/api/blind-review", response_class=JSONResponse)
    def blind_review_data() -> JSONResponse:
        benchmark, corpus_dir, review_path = blind_context()
        try:
            result = load_answer_review(review_path, benchmark, corpus_dir)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.get("/api/blind-review/{blind_id}/source", response_class=JSONResponse)
    def blind_review_source(blind_id: str) -> JSONResponse:
        benchmark, corpus_dir, review_path = blind_context()
        review = load_answer_review(review_path, benchmark, corpus_dir)
        row = next((item for item in review["rows"] if item["blind_id"] == blind_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown blind_id")
        return JSONResponse(frozen_paper(row["question_id"], benchmark, corpus_dir))

    @app.post("/api/blind-review/finalize", response_class=JSONResponse)
    def finalize_blind_review(payload: dict = Body(...)) -> JSONResponse:
        require_blind_human_session()
        benchmark, corpus_dir, review_path = blind_context()
        try:
            result = finalize_answer_review(
                review_path, benchmark, corpus_dir, payload.get("reviewer_id"),
                expected_sha256=payload.get("expected_sha256"),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post("/api/blind-review/{blind_id}", response_class=JSONResponse)
    def save_blind_review(blind_id: str, payload: dict = Body(...)) -> JSONResponse:
        require_blind_human_session()
        benchmark, corpus_dir, review_path = blind_context()
        try:
            result = save_answer_verdict(
                review_path, benchmark, corpus_dir, blind_id,
                payload.get("verdict"), payload.get("notes"),
                payload.get("relevant_passage"), payload.get("reviewer_id"),
                expected_sha256=payload.get("expected_sha256"),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(result)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run-dir", type=Path, help="complete local EnvoyBench run directory")
    parser.add_argument("--review-dir", type=Path, help="completed blind review bundle")
    parser.add_argument(
        "--judged-review", type=Path,
        help="completed model-graded review JSON; key remains in --review-dir",
    )
    parser.add_argument(
        "--provisional", action="store_true",
        help="show a test_candidate run without human review; never a promotion result",
    )
    parser.add_argument(
        "--blind-review-dir", type=Path,
        help="prepared anonymous answer-review bundle to edit before scoring",
    )
    parser.add_argument(
        "--reference-review", type=Path, default=DEFAULT_REFERENCE_REVIEW,
        help="local source-review sheet; created on the first saved decision",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--paper-state-dir", type=Path, default=DEFAULT_PAPER_STATE,
        help="local paper snapshots and saved investigations (separate from the benchmark)",
    )
    parser.add_argument(
        "--paper-models", type=Path,
        help="operator model JSON for live paper runs; omitted means reader and saved runs only",
    )
    parser.add_argument(
        "--web-evidence-report", type=Path,
        help="completed local web-to-evidence pilot report JSON (read-only)",
    )
    parser.add_argument(
        "--token-diagnostic-run", type=Path,
        help="complete paired dev smoke directory; separate from benchmark quality scores",
    )
    parser.add_argument("--supplementary-run", type=Path, action="append", default=[],
                        help="additional saved run, kept separate from the paired comparison")
    parser.add_argument("--qasper-score", type=Path, action="append", default=[],
                        help="saved official QASPER Answer F1 report bound to a loaded run")
    args = parser.parse_args()
    app = create_app(
        run_dir=args.run_dir, review_dir=args.review_dir, dataset=args.dataset,
        reference_review_path=args.reference_review,
        blind_review_dir=args.blind_review_dir,
        provisional=args.provisional,
        judged_review_path=args.judged_review,
        paper_state_dir=args.paper_state_dir,
        paper_models_path=args.paper_models,
        web_evidence_report_path=args.web_evidence_report,
        token_diagnostic_run=args.token_diagnostic_run,
        supplementary_runs=args.supplementary_run,
        qasper_scores=args.qasper_score,
    )
    print(f"QASPER Agent Studio: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
