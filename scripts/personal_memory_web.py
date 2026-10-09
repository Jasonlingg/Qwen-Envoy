"""Loopback-only vault chat, source review, and approval-gated memory writes.

The default mode is evidence-only lexical retrieval. Remote model calls require
an explicit CLI flag and credentials; Qwen code execution requires Docker.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
from pathlib import Path
from uuid import uuid4

import uvicorn
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policies.code_execution import DEFAULT_MAX_TOKENS  # noqa: E402
from src.policies.openai_compatible import OpenAICompatiblePolicy  # noqa: E402
from src.product.chat import ChatService  # noqa: E402
from src.product.impact import (  # noqa: E402
    IMPACT_VERSION,
    build_impact_review,
    visible_title,
)
from src.product.learning_lab_api import create_learning_lab_router  # noqa: E402
from src.product.memory import (  # noqa: E402
    _record_context,
    approve_note,
    capture_note,
    freeze_vault,
    inspect_evidence,
)
from src.product.nemotron_chat import NemotronChatClient  # noqa: E402
from src.product.qwen_investigator import (  # noqa: E402
    QwenInvestigator,
    prompt_for_source_domain,
)
from src.research.agent import load_snapshot  # noqa: E402

TEMPLATE = Path(__file__).parent / "templates" / "personal_memory.html"


def _configured_key(name: str, env_file: Path | None = None) -> str | None:
    """Read a model key from this process or the ignored repo .env, without printing it."""
    value = os.environ.get(name)
    if not value:
        value = dotenv_values(env_file or ROOT / ".env").get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


class CaptureInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=20_000)
    kind: str = "attempt"
    effective_date: str | None = None


class ApprovalInput(BaseModel):
    review_id: str
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=20_000)
    kind: str
    evidence_ids: list[str]
    effective_date: str | None = None
    supersedes: str | None = None
    confirm: bool = False


class ChatInput(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    session_id: str | None = Field(default=None, max_length=64)


class ImpactInput(BaseModel):
    source_path: str = Field(min_length=1, max_length=1_000)
    draft_with_model: bool = False


def create_app(vault: Path, state_dir: Path,
               *, chat_service: ChatService | None = None) -> FastAPI:
    """Make one local user's app; keep review artifacts bound to frozen snapshots."""
    vault = vault.expanduser().resolve(strict=True)
    if not vault.is_dir():
        raise ValueError("vault must be a directory")
    state_dir = state_dir.expanduser().resolve()
    if state_dir.is_relative_to(vault):
        raise ValueError("state directory must be outside the vault")
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.RLock()
    reviews: dict[str, tuple[Path, dict]] = {}
    sessions: dict[str, tuple[str, list[dict]]] = {}
    session_locks: dict[str, threading.Lock] = {}
    chat_service = chat_service or ChatService()
    model_calls = 0
    latest: Path | None = None

    def snapshot_now() -> Path | None:
        nonlocal latest
        target = state_dir / f"snapshot-{uuid4().hex}"
        try:
            manifest = freeze_vault(vault, target)
        except ValueError as exc:
            if str(exc) != "No Markdown or PDF files found in the selected collection":
                raise
            latest = None
            return None
        if not manifest["papers"]:
            shutil.rmtree(target)
            latest = None
            return None
        latest = target
        return target

    snapshot_now()
    app = FastAPI(title="Envoy Learning Memory · Local Demo")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )

    @app.middleware("http")
    async def reject_cross_origin_writes(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            same_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
            fetch_site = request.headers.get("sec-fetch-site")
            if (origin is not None and origin != same_origin) or fetch_site in {
                "cross-site", "same-site"
            }:
                return JSONResponse(
                    {"detail": "cross-origin writes are not allowed"}, status_code=403,
                )
        return await call_next(request)

    def refresh_for_lab() -> dict:
        with lock:
            snapshot = snapshot_now()
            if snapshot is None:
                return {"documents": 0, "corpus_hash": None, "snapshot_updated": True}
            manifest, docs = load_snapshot(snapshot)
            return {"documents": len(docs), "corpus_hash": manifest["corpus_hash"],
                    "snapshot_updated": True}

    def snapshot_for_lab() -> Path:
        with lock:
            if latest is None:
                raise ValueError("No reviewed notes yet; promote a source draft and refresh")
            return latest

    def record_lab_model_calls(count: int) -> None:
        nonlocal model_calls
        with lock:
            model_calls += count

    app.include_router(create_learning_lab_router(
        vault, state_dir,
        template=TEMPLATE.with_name("learning_lab.html"),
        refresh_vault=refresh_for_lab,
        get_vault_snapshot=snapshot_for_lab,
        coordinator=chat_service.coordinator,
        investigator=chat_service.investigator,
        record_model_call=record_lab_model_calls,
    ))

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/lab", status_code=307)

    @app.get("/chat", response_class=HTMLResponse)
    def chat_page() -> str:
        return TEMPLATE.read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict:
        with lock:
            if latest is None:
                manifest, docs = {"corpus_hash": None}, {}
            else:
                manifest, docs = load_snapshot(latest)
            results = []
            for doc in docs.values():
                record = _record_context(doc)
                source_path = doc["metadata"].get("source_path")
                if record.get("kind") != "result" or not source_path:
                    continue
                results.append({
                    "source_path": source_path,
                    "title": visible_title(doc),
                    "effective_date": record.get("effective_date"),
                    "captured_at": record.get("captured_at"),
                    "review_status": record.get("review_status"),
                })
            results.sort(key=lambda item: (
                item["effective_date"] or item["captured_at"] or "",
                item["source_path"],
            ), reverse=True)
            return {"vault": str(vault), "documents": len(docs),
                    "corpus_hash": manifest["corpus_hash"],
                    "recent_results": results[:5],
                    "retriever": ("qwen_code_execution_with_lexical_fallback"
                                  if chat_service.investigator else "lexical_paragraph_baseline"),
                    "nemotron_configured": chat_service.coordinator is not None,
                    "qwen_configured": chat_service.investigator is not None,
                    "model_requests_attempted": model_calls,
                    "model_calls": model_calls}

    @app.post("/api/chat")
    def chat(body: ChatInput) -> dict:
        nonlocal model_calls
        with lock:
            if latest is None:
                raise HTTPException(
                    status_code=409,
                    detail="No reviewed notes yet; promote a source draft and refresh",
                )
            session_id = body.session_id or uuid4().hex
            if body.session_id is not None and body.session_id not in sessions:
                raise HTTPException(status_code=404, detail="session not found")
            session_lock = session_locks.setdefault(session_id, threading.Lock())
        # A slow model request must not block capture, source review, or status.
        # Only turns in the same conversation are serialized.
        with session_lock:
            with lock:
                if latest is None:
                    raise HTTPException(
                        status_code=409,
                        detail="No reviewed notes yet; promote a source draft and refresh",
                    )
                snapshot = latest
                manifest, _ = load_snapshot(snapshot)
                prior = sessions.get(session_id)
                history_reset = prior is not None and prior[0] != manifest["corpus_hash"]
                history = prior[1][-6:] if prior is not None and not history_reset else []
            try:
                turn = chat_service.reply(body.question, snapshot, history=history)
            except (ValueError, OSError) as exc:
                if body.session_id is None:
                    with lock:
                        session_locks.pop(session_id, None)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            with lock:
                review_id = uuid4().hex
                reviews[review_id] = (snapshot, turn)
                if len(reviews) > 100:
                    reviews.pop(next(iter(reviews)))
                snapshot_stale = snapshot != latest
                if snapshot_stale:
                    if latest is None:
                        sessions[session_id] = ("", [])
                    else:
                        current_manifest, _ = load_snapshot(latest)
                        sessions[session_id] = (current_manifest["corpus_hash"], [])
                else:
                    updated = [*history, {"role": "user", "content": body.question}]
                    if turn["answer"] is not None:
                        updated.append({"role": "assistant", "content": turn["answer"]})
                    sessions[session_id] = (manifest["corpus_hash"], updated[-6:])
                if len(sessions) > 100:
                    for old_id in list(sessions):
                        if old_id != session_id and not session_locks[old_id].locked():
                            sessions.pop(old_id)
                            session_locks.pop(old_id)
                            break
                model_calls += turn.get("model_requests_attempted", 0)
                return {
                    "review_id": review_id, "session_id": session_id,
                    "snapshot_stale": snapshot_stale,
                    "history_reset": history_reset or snapshot_stale,
                    **turn,
                }

    @app.post("/api/capture")
    def capture(body: CaptureInput) -> dict:
        with lock:
            try:
                note = capture_note(vault, title=body.title, text=body.text,
                                    kind=body.kind, effective_date=body.effective_date)
                snapshot_now()
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"saved": str(note.relative_to(vault)), "snapshot_updated": True}

    @app.post("/api/impact")
    def impact(body: ImpactInput) -> dict:
        """Suggest an earlier conclusion to review; do not assert a conflict."""
        nonlocal model_calls
        with lock:
            if latest is None:
                raise HTTPException(status_code=409, detail="No reviewed notes yet")
            snapshot = latest
        try:
            review = build_impact_review(snapshot, body.source_path)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        attempted = 0
        if body.draft_with_model:
            if chat_service.coordinator is None:
                raise HTTPException(status_code=400, detail="Nemotron is not configured")
            proposal = review.get("proposal")
            if proposal is not None:
                # The local candidate and exact snapshot passages are supplied
                # as data. The model may draft wording, never approve a memory.
                packet = {
                    "schema_version": review["schema_version"],
                    "corpus_hash": review["corpus_hash"],
                    "retriever_protocol": review["retriever"],
                    "warning": review["warning"],
                    "evidence": review["evidence"],
                }
                question = (
                    "Draft a possible correction to my earlier decision after "
                    "the newly captured source. Cite both the earlier view and "
                    "the new source. Distinguish "
                    "a measured outcome from a plan, state the narrow scope of "
                    "the evidence, and say what remains uncertain. This is a "
                    "proposal for my review, not a saved conclusion."
                )
                attempted = 1
                try:
                    answer = chat_service.coordinator.answer(question, packet, history=[])
                    if not isinstance(answer, dict) or not isinstance(answer.get("answer"), str):
                        raise ValueError("draft answer must be text")
                    used = set(re.findall(r"\[(E[1-5])\]", answer["answer"]))
                    if not set(proposal["evidence_ids"]).issubset(used):
                        raise ValueError("draft must cite both the new and earlier sources")
                    proposal.update({
                        "text": answer["answer"], "requires_edit": False,
                        "origin": "nemotron_draft_for_user_review",
                        "model": answer["model"],
                        "limitations": answer["limitations"],
                    })
                except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
                    review["draft_error"] = (
                        f"Nemotron draft unavailable ({type(exc).__name__}); "
                        "the local review scaffold remains."
                    )
        with lock:
            model_calls += attempted
            snapshot_stale = snapshot != latest
            if snapshot_stale:
                review["proposal"] = None
                review["reason"] = (
                    "The vault changed during this review. Recheck the new snapshot "
                    "before approving a revision."
                )
            review_id = uuid4().hex
            reviews[review_id] = (snapshot, review)
            if len(reviews) > 100:
                reviews.pop(next(iter(reviews)))
            return {"review_id": review_id, "snapshot_stale": snapshot_stale,
                    "model_requests_attempted": attempted, **review}

    @app.get("/api/search")
    def search(query: str, top_k: int = 5) -> dict:
        with lock:
            if latest is None:
                raise HTTPException(status_code=409, detail="No reviewed notes yet")
            try:
                review = inspect_evidence(latest, query, top_k)
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            review_id = uuid4().hex
            reviews[review_id] = (latest, review)
            if len(reviews) > 100:
                reviews.pop(next(iter(reviews)))
            return {"review_id": review_id, **review}

    @app.get("/api/source/{review_id}/{doc_id}")
    def source(review_id: str, doc_id: str) -> dict:
        with lock:
            stored = reviews.get(review_id)
            if stored is None:
                raise HTTPException(status_code=404, detail="review not found")
            snapshot, review = stored
            allowed_docs = {item["doc_id"] for item in review["evidence"]}
            allowed_docs.update(item["doc_id"] for item in review.get("related_notes", []))
            if doc_id not in allowed_docs:
                raise HTTPException(status_code=404, detail="source not in this review")
            _, docs = load_snapshot(snapshot)
            doc = docs[doc_id]
            return {"doc_id": doc_id, "title": doc["title"], "text": doc["text"],
                    "source_path": doc["metadata"].get("source_path")}

    @app.post("/api/approve")
    def approve(body: ApprovalInput) -> dict:
        if not body.confirm:
            raise HTTPException(status_code=400, detail="explicit approval is required")
        with lock:
            stored = reviews.get(body.review_id)
            if stored is None:
                raise HTTPException(status_code=404, detail="review not found")
            snapshot, review = stored
            if snapshot != latest:
                raise HTTPException(
                    status_code=409,
                    detail="the vault changed since this review; search again before approving",
                )
            proposal = review.get("proposal")
            if (isinstance(proposal, dict) and proposal.get("requires_edit")
                    and body.text.strip() == proposal.get("text", "").strip()):
                raise HTTPException(
                    status_code=400,
                    detail="edit the review scaffold into your conclusion before approving",
                )
            if review.get("schema_version") == IMPACT_VERSION and review.get("prior"):
                expected = {item["evidence_id"] for item in review["evidence"]}
                if (body.kind != "correction"
                        or body.supersedes != review["prior"]["doc_id"]
                        or not expected.issubset(set(body.evidence_ids))):
                    raise HTTPException(
                        status_code=400,
                        detail=("an impact correction must cite both notes and "
                                "supersede the reviewed decision"),
                    )
            try:
                note = approve_note(
                    vault, snapshot, review, title=body.title, text=body.text,
                    kind=body.kind, evidence_ids=body.evidence_ids,
                    effective_date=body.effective_date, supersedes=body.supersedes,
                )
                snapshot_now()
            except (ValueError, OSError, KeyError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"saved": str(note.relative_to(vault)), "snapshot_updated": True}

    return app


def make_qwen_investigator(endpoint: str, model: str, checkpoint: str,
                           *, source_domain: str = "personal_vault") -> QwenInvestigator:
    """Configure a served Qwen worker for an explicit frozen source domain."""
    system_prompt = prompt_for_source_domain(source_domain)
    return QwenInvestigator(
        policy_factory=lambda: OpenAICompatiblePolicy(
            endpoint=endpoint,
            model=model,
            api_key=_configured_key("ENVOY_MODEL_API_KEY"),
            system_prompt=system_prompt,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=0.0,
            # Qwen3 can spend the whole action budget in its thinking block.
            # Keep this model-specific; Nemotron uses a separate client.
            extra_body={"chat_template_kwargs": {"enable_thinking": False},
                        "top_p": 1.0, "seed": 42},
        ),
        model_identity={"checkpoint": checkpoint, "served_model": model,
                        "endpoint": endpoint, "temperature": 0.0,
                        "top_p": 1.0, "seed": 42,
                        "max_tokens": DEFAULT_MAX_TOKENS,
                        "thinking": False},
        source_domain=source_domain,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, required=True,
                        help="Copy a sample vault before writing to it")
    parser.add_argument("--state-dir", type=Path, default=Path("out/personal-memory-web"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--enable-nebius", action="store_true",
                        help="Allow selected-vault excerpts to be sent to Nebius Token Factory")
    parser.add_argument("--nemotron-model", default=None)
    parser.add_argument("--qwen-endpoint", default=None,
                        help="OpenAI-compatible endpoint serving the pinned Qwen checkpoint")
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--qwen-checkpoint", default=None,
                        help="Exact checkpoint identifier to report in chat traces")
    args = parser.parse_args()
    coordinator = None
    investigator = None
    if args.enable_nebius:
        key = _configured_key("NEBIUS_API_KEY")
        if not key:
            parser.error("--enable-nebius requires NEBIUS_API_KEY")
        options = {"api_key": key}
        if args.nemotron_model:
            options["model"] = args.nemotron_model
        coordinator = NemotronChatClient(**options)
    elif args.nemotron_model:
        parser.error("--nemotron-model requires --enable-nebius")
    if args.qwen_endpoint or args.qwen_model or args.qwen_checkpoint:
        if not (args.qwen_endpoint and args.qwen_model and args.qwen_checkpoint):
            parser.error("Qwen requires --qwen-endpoint, --qwen-model, and --qwen-checkpoint")
        investigator = make_qwen_investigator(
            args.qwen_endpoint, args.qwen_model, args.qwen_checkpoint,
        )
    app = create_app(args.vault, args.state_dir, chat_service=ChatService(
        coordinator=coordinator, investigator=investigator,
    ))
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}",
                      "mode": "nebius" if coordinator else "evidence_only",
                      "qwen": bool(investigator), "model_calls": 0}))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
