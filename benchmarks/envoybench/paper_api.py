"""Local, trusted-operator paper uploads and bounded, saved model investigations.

There is deliberately no endpoint, credential, or execution-backend web input.
The operator supplies a pinned model config; uploaded papers are unreviewed and
are never given benchmark scores. Only the labeled Docker sandbox executes code.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from benchmarks.envoybench.runtime_support import (
    _redact,
    _write_json,
    load_models,
    preflight_sandbox,
)
from src.policies.code_execution import clean_action
from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.product.qwen_investigator import (
    MAX_ACTION_CHARS,
    NEMOTRON_PAPER_SYSTEM_PROMPT,
    PUBLIC_PAPER_SYSTEM_PROMPT,
    QwenInvestigator,
)
from src.research.code_exec_packet import MAX_EVIDENCE_SPANS, MAX_QUESTION_CHARS
from src.research.file_sources import MAX_FILE_BYTES, build_file_snapshot, load_file_snapshot
from src.research.web_sources import canonicalize_web_url, fetch_public_pdf

MAX_STEPS = 12
MODEL_TIMEOUT_SECONDS = 60
CODE_TIMEOUT_SECONDS = 20
RUN_TIMEOUT_SECONDS = 300
MAX_SAVED_REASONING_CHARS = 12_000
SEED = 42
_ID = re.compile(r"(?:paper|run)_[0-9a-f]{32}\Z")
_ACTIVE = {"queued", "running"}
_EXPOSURE = {"upstream": "unknown", "fine_tuning": "not_audited"}
_UNCONFIGURED = "No operator model config supplied; upload, reading, and history remain available."
_ARXIV_PAPER_PATH = re.compile(r"/(?:abs|pdf|html)/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?/?\Z")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "Open Studio directly to upload or start a run")


def _safe_path(directory: Path, identifier: str, prefix: str, suffix: str = "") -> Path:
    if not _ID.fullmatch(identifier) or not identifier.startswith(prefix + "_"):
        raise HTTPException(404, "Unknown paper or run ID")
    path = directory / (identifier + suffix)
    if path.is_symlink() or not path.exists():
        raise HTTPException(404, "Unknown paper or run ID")
    return path


def _checked_evidence(items: list, docs: dict) -> list[dict]:
    """Only exact, nonempty spans become clickable source evidence."""
    if not isinstance(items, list) or len(items) > MAX_EVIDENCE_SPANS:
        raise ValueError("Invalid evidence list")
    result, seen = [], set()
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError("Invalid evidence span")
        doc_id, start, end = item.get("doc_id"), item.get("start"), item.get("end")
        if not isinstance(doc_id, str) or doc_id not in docs:
            raise ValueError("Evidence document is absent")
        text = docs[doc_id]["text"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
            or item.get("quote") != text[start:end]
            or (doc_id, start, end) in seen
        ):
            raise ValueError("Evidence quote or offsets are invalid")
        seen.add((doc_id, start, end))
        pages = sorted(
            {
                section["page"]
                for section in docs[doc_id].get("sections", [])
                if type(section.get("page")) is int
                and section["start"] < end
                and section["end"] > start
            }
        )
        result.append(
            {
                "id": f"E{index}",
                "doc_id": doc_id,
                "start": start,
                "end": end,
                "quote": text[start:end],
                "pages": pages,
            }
        )
    return result


def _paper_protocol(model: dict) -> tuple[str, str]:
    """Choose a live-reader prompt without changing the frozen benchmark."""
    model_id = model["safe"]["model_id"].casefold()
    if "nemotron" in model_id:
        return "nemotron", NEMOTRON_PAPER_SYSTEM_PROMPT
    return "default", PUBLIC_PAPER_SYSTEM_PROMPT


def _policy_failure_category(exc: Exception) -> str:
    """Expose only known local parser states, never provider error bodies."""
    if isinstance(exc, ValueError):
        return {
            "chat-completions response did not contain text content": "missing_visible_action",
            "chat-completions response did not contain a final action": "missing_visible_action",
            "thinking response did not contain a final action": "reasoning_without_action",
            "unfinished thinking block in model response": "unfinished_thinking",
            "ambiguous thinking block in model response": "ambiguous_thinking",
            "conflicting structured and inline reasoning": "reasoning_field_conflict",
            "conflicting chat-completions reasoning fields": "reasoning_field_conflict",
            "invalid chat-completions response shape": "invalid_response_shape",
        }.get(str(exc), "model_response_invalid")
    if isinstance(exc, TimeoutError):
        return "model_timeout"
    return "model_request_failed"


class _RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    paper_id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    model_key: str = Field(min_length=1, max_length=100)


class _WebPaperRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    url: str = Field(min_length=1, max_length=2048)


def _pdf_url(value: str) -> tuple[str, str, str]:
    """Resolve arXiv paper pages to PDF URLs; accept direct HTTPS PDF links."""
    requested = canonicalize_web_url(value)
    parts = urlsplit(requested)
    if parts.scheme != "https":
        raise ValueError("Use an HTTPS paper link")
    if parts.hostname in {"arxiv.org", "www.arxiv.org"}:
        match = _ARXIV_PAPER_PATH.fullmatch(parts.path)
        if match:
            identifier = match.group(1)
            return requested, f"https://arxiv.org/pdf/{identifier}", f"arxiv-{identifier}.pdf"
    label = hashlib.sha256(requested.encode()).hexdigest()[:12]
    return requested, requested, f"paper-{label}.pdf"


class _ProgressPolicy:
    """Observe the existing policy protocol without modifying the investigator."""

    def __init__(self, policy, record: dict, save, started: float):
        self.policy, self.record, self.save, self.started = policy, record, save, started
        self.trace: list[dict] = []

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def export(self) -> dict:
        export = getattr(self.policy, "eval_metadata", None)
        return export() if callable(export) else {}

    def act(self, observation: str) -> str:
        if self.trace:
            self.trace[-1]["observation"] = observation
        self.record["trajectory"] = self.trace
        remaining = RUN_TIMEOUT_SECONDS - (time.monotonic() - self.started)
        if remaining <= 0:
            raise TimeoutError("Investigation time budget exhausted")
        # Each HTTP request is bounded too; a slow request cannot use the next
        # turn's budget. Docker execution has its separate 20-second timeout.
        self.policy.timeout = min(MODEL_TIMEOUT_SECONDS, remaining)
        self.save(self.record)
        began = time.monotonic()
        try:
            try:
                action = self.policy.act(observation)
                if not isinstance(action, str):
                    raise ValueError("Policy action must be text")
            except Exception as exc:
                category = _policy_failure_category(exc)
                failed_step = {
                    "step": len(self.trace) + 1,
                    "action": None,
                    "observation": None,
                    "model_seconds": time.monotonic() - began,
                    "model_error_category": category,
                }
                response_metadata = getattr(self.policy, "last_response_metadata", None)
                if isinstance(response_metadata, dict):
                    failed_step["response_metadata"] = response_metadata
                rejected_content = getattr(self.policy, "last_raw_response_content", None)
                if isinstance(exc, ValueError) and isinstance(rejected_content, str):
                    # This is the provider's visible text, not a parsed action.
                    # _save() redacts configured secrets before it leaves memory.
                    failed_step["rejected_model_content_unexecuted"] = rejected_content
                    failed_step["rejected_content_truncated"] = bool(
                        getattr(self.policy, "last_raw_response_truncated", False)
                    )
                self.trace.append(failed_step)
                raise
            step = {
                "step": len(self.trace) + 1,
                "action": clean_action(action)[:MAX_ACTION_CHARS],
                "observation": None,
                "model_seconds": time.monotonic() - began,
            }
            response_metadata = getattr(self.policy, "last_response_metadata", None)
            if isinstance(response_metadata, dict):
                step["response_metadata"] = response_metadata
            reasoning = getattr(self.policy, "last_reasoning", None)
            if isinstance(reasoning, str) and reasoning:
                step["provider_reasoning"] = reasoning[:MAX_SAVED_REASONING_CHARS]
                step["provider_reasoning_truncated"] = len(reasoning) > MAX_SAVED_REASONING_CHARS
            logprobs = getattr(self.policy, "last_logprob_diagnostics", None)
            if isinstance(logprobs, dict):
                step["sampled_logprobs"] = logprobs
            self.trace.append(step)
            if time.monotonic() - self.started > RUN_TIMEOUT_SECONDS:
                self.trace[-1]["observation"] = "Not executed: investigation time budget exhausted."
                raise TimeoutError("Investigation time budget exhausted")
            return action
        finally:
            self.record["usage"] = self.export().get("token_usage", {})
            self.record["duration_seconds"] = time.monotonic() - self.started
            self.save(self.record)


class _PaperService:
    def __init__(self, state_dir: Path, models_path: Path | None):
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.papers = self.state_dir / "papers"
        self.runs = self.state_dir / "runs"
        self.papers.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper-investigation")
        self.config_lock = threading.Lock()
        self.preflight_checked = 0.0
        self.preflight_error = None
        self.models, self.secrets = {}, []
        self.config_error = _UNCONFIGURED
        if models_path is not None:
            try:
                models = load_models(Path(models_path))
                self.models = {model["safe"]["key"]: model for model in models}
                for model in models:
                    endpoint = model["endpoint"]
                    self.secrets.extend(
                        value
                        for value in (
                            model.get("api_key"),
                            endpoint,
                            endpoint.rstrip("/"),
                            urlsplit(endpoint).netloc,
                        )
                        if value
                    )
                self.config_error = None
            except (ValueError, OSError):
                self.config_error = (
                    "Operator model config is invalid or its required environment is unset."
                )
        self._recover_interrupted()

    def _lock(self):
        stream = (self.state_dir / ".active-run.lock").open("a+")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            raise HTTPException(409, "A paper investigation is already active") from None
        return stream

    def _recover_interrupted(self) -> None:
        try:
            lock = self._lock()
        except HTTPException:
            return
        try:
            for path in self.runs.glob("run_*.json"):
                if path.is_symlink():
                    continue
                try:
                    record = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if isinstance(record, dict) and record.get("status") in _ACTIVE:
                    record.update(
                        status="interrupted",
                        error="Server stopped before this run finished.",
                        finished_at=_now(),
                        answer=None,
                        evidence=[],
                    )
                    self._save(record)
        finally:
            lock.close()

    def _save(self, record: dict) -> None:
        identifier = record.get("id", "")
        if not _ID.fullmatch(identifier) or not identifier.startswith("run_"):
            raise ValueError("Invalid saved run ID")
        _write_json(self.runs / f"{identifier}.json", _redact(record, self.secrets))

    def _identity(self, model: dict) -> dict:
        # The public identity retains reproducibility fields, but no endpoint,
        # environment variable names, headers, or arbitrary provider options.
        allowed = (
            "key",
            "model_id",
            "revision",
            "identity_source",
            "base_model_id",
            "base_revision",
            "adapter_id",
            "adapter_revision",
            "adapter_sha256",
            "decoding",
            "send_seed",
            "serving_hardware",
            "serving_runtime",
            "serving_context_window_tokens",
        )
        return _redact(
            {key: model["safe"][key] for key in allowed if key in model["safe"]}, self.secrets
        )

    def config(self) -> dict:
        reason = self.config_error
        if reason is None:
            with self.config_lock:
                if time.monotonic() - self.preflight_checked > 30:
                    try:
                        preflight_sandbox()
                        self.preflight_error = None
                    except Exception:
                        self.preflight_error = (
                            "Labeled Docker sandbox unavailable. Build the EnvoyBench rlm-sandbox "
                            "image and start Docker before running a question."
                        )
                    self.preflight_checked = time.monotonic()
                reason = self.preflight_error
        return {
            "ready": reason is None,
            "reason": reason,
            "models": [self._identity(model) for model in self.models.values()],
            "max_file_bytes": MAX_FILE_BYTES,
            "max_question_chars": MAX_QUESTION_CHARS,
            "max_steps": MAX_STEPS,
            "run_timeout_seconds": RUN_TIMEOUT_SECONDS,
        }

    def _snapshot(self, identifier: str) -> tuple[Path, dict, dict]:
        path = _safe_path(self.papers, identifier, "paper")
        try:
            manifest, docs = load_file_snapshot(path)
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(
                409, "Paper snapshot failed its raw or parsed source integrity check"
            ) from None
        return path, manifest, docs

    def paper(self, identifier: str, *, full: bool = True) -> dict:
        _, manifest, docs = self._snapshot(identifier)
        doc = next(iter(docs.values()))
        value = {
            "id": identifier,
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "filename": doc["metadata"]["filename"],
            "created_at": doc["metadata"]["uploaded_at"],
            "corpus_hash": manifest["corpus_hash"],
            "raw_sha256": manifest["raw_sources_hash"],
            "metadata": doc["metadata"],
            "source_url": doc["metadata"].get("source_url"),
            "exposure": dict(_EXPOSURE),
            "review_status": "unreviewed",
        }
        if full:
            value.update(text=doc["text"], sections=doc.get("sections", []))
        return value

    def upload(self, filename: str, body: bytes) -> dict:
        identifier = "paper_" + uuid4().hex
        staging = self.papers / (".upload_" + uuid4().hex)
        try:
            build_file_snapshot(filename, body, staging)
            staging.rename(self.papers / identifier)
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return self.paper(identifier)

    def fetch_pdf(self, url: str) -> dict:
        try:
            requested, pdf_url, filename = _pdf_url(url)
            fetched = fetch_public_pdf(pdf_url)
            raw_hash = hashlib.sha256(fetched.body).hexdigest()
            for existing in self.papers.glob("paper_*"):
                if not _ID.fullmatch(existing.name) or existing.is_symlink():
                    continue
                try:
                    manifest = json.loads((existing / "manifest.json").read_text())
                except (OSError, ValueError):
                    continue
                if not isinstance(manifest, dict):
                    continue
                rows = manifest.get("papers", [])
                if (
                    manifest.get("raw_sources_hash") == raw_hash
                    and isinstance(rows, list)
                    and len(rows) == 1
                    and isinstance(rows[0], dict)
                    and rows[0].get("requested_url") == requested
                ):
                    return self.paper(existing.name)
            identifier = "paper_" + uuid4().hex
            staging = self.papers / (".fetch_" + uuid4().hex)
            try:
                build_file_snapshot(
                    filename,
                    fetched.body,
                    staging,
                    source_url=fetched.final_url,
                    requested_url=requested,
                )
                staging.rename(self.papers / identifier)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        except ValueError as exc:
            raise HTTPException(422, _redact(str(exc), [url])) from None
        except OSError:
            raise HTTPException(502, "Could not fetch this PDF link") from None
        return self.paper(identifier)

    def _make_policy(self, model: dict):
        spec = model["safe"]
        _, system_prompt = _paper_protocol(model)
        extra_body = {**spec["extra_body"], "top_p": spec["decoding"]["top_p"]}
        if spec.get("send_seed"):
            extra_body["seed"] = SEED
        policy = OpenAICompatiblePolicy(
            endpoint=model["endpoint"],
            model=spec["model_id"],
            api_key=model["api_key"],
            max_tokens=spec["decoding"]["max_tokens"],
            temperature=spec["decoding"]["temperature"],
            timeout=MODEL_TIMEOUT_SECONDS,
            extra_body=extra_body,
            system_prompt=system_prompt,
        )
        # Do not inherit the adapter's legacy credential environment fallback.
        policy.api_key = model["api_key"]
        return policy

    def start(self, request: _RunRequest) -> dict:
        if self.config_error:
            raise HTTPException(503, self.config_error)
        if request.model_key not in self.models:
            raise HTTPException(422, "Unknown operator-configured model key")
        question = request.question.strip()
        if not question:
            raise HTTPException(422, "Question must contain text")
        _, manifest, docs = self._snapshot(request.paper_id)
        doc = next(iter(docs.values()))
        lock = self._lock()
        try:
            protocol_variant, system_prompt = _paper_protocol(self.models[request.model_key])
            record = {
                "schema_version": "envoybench-paper-run-v1",
                "id": "run_" + uuid4().hex,
                "status": "queued",
                "paper_id": request.paper_id,
                "question": question,
                "model_key": request.model_key,
                "model_identity": self._identity(self.models[request.model_key]),
                "created_at": _now(),
                "duration_seconds": 0,
                "trajectory": [],
                "answer": None,
                "evidence": [],
                "outcome": None,
                "usage": {},
                "error": None,
                "review_status": "unreviewed",
                "exposure": dict(_EXPOSURE),
                "corpus_hash": manifest["corpus_hash"],
                "raw_sha256": manifest["raw_sources_hash"],
                "source_url": doc["metadata"].get("source_url"),
                "execution": "docker_only",
                "source_domain": "public_papers",
                "protocol_variant": protocol_variant,
                "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
                "seed": SEED,
                "limits": {
                    "max_steps": MAX_STEPS,
                    "model_timeout_seconds": MODEL_TIMEOUT_SECONDS,
                    "code_timeout_seconds": CODE_TIMEOUT_SECONDS,
                    "run_timeout_seconds": RUN_TIMEOUT_SECONDS,
                },
            }
            self._save(record)
            # Pass a separate object so the HTTP 202 response is a stable queued record.
            self.executor.submit(self._work, dict(record), lock)
            return _redact(record, self.secrets)
        except Exception:
            lock.close()
            raise

    def _work(self, record: dict, lock) -> None:
        started = time.monotonic()
        progress = None
        try:
            record.update(status="running", started_at=_now())
            self._save(record)
            snapshot, manifest, _ = self._snapshot(record["paper_id"])
            if (
                manifest["corpus_hash"] != record["corpus_hash"]
                or manifest["raw_sources_hash"] != record["raw_sha256"]
            ):
                raise ValueError("Paper changed before investigation")
            # Never trust the config endpoint's cached readiness for execution.
            # This must complete before policy creation or any model request.
            record["sandbox"] = preflight_sandbox()
            progress = _ProgressPolicy(
                self._make_policy(self.models[record["model_key"]]), record, self._save, started
            )
            investigator = QwenInvestigator(
                policy_factory=lambda: progress,
                model_identity=record["model_identity"],
                max_steps=MAX_STEPS,
                code_timeout_seconds=CODE_TIMEOUT_SECONDS,
                source_domain="public_papers",
                protocol_variant=record["protocol_variant"],
            )
            packet = investigator.investigate(record["question"], snapshot)
            packet_trace = packet.get("trajectory", [])
            record["trajectory"] = [
                {
                    **(progress.trace[index] if index < len(progress.trace) else {}),
                    **(packet_trace[index] if index < len(packet_trace) else {}),
                }
                for index in range(max(len(progress.trace), len(packet_trace)))
            ]
            _, after, docs = self._snapshot(record["paper_id"])
            if (
                after["corpus_hash"] != record["corpus_hash"]
                or after["raw_sources_hash"] != record["raw_sha256"]
            ):
                raise ValueError("Paper changed during investigation")
            evidence = _checked_evidence(packet.get("evidence", []), docs)
            outcome = packet.get("status", "error")
            record.update(
                outcome=outcome,
                evidence=evidence,
                status="completed" if outcome in {"evidence_found", "no_evidence"} else "failed",
            )
            if outcome == "evidence_found":
                if not evidence:
                    raise ValueError("Answer has no checked evidence")
                record["answer"] = "\n\n".join(
                    claim["text"] for claim in packet["candidate_claims"]
                )
            elif outcome == "no_evidence":
                record["answer"] = "Unanswerable"
            if record["status"] == "failed":
                record["error"] = (
                    "Step limit reached without an accepted answer."
                    if outcome == "incomplete"
                    else "Investigation failed; inspect the saved trajectory and retry explicitly."
                )
            for key in (
                "warning",
                "uncertainty",
                "investigator_version",
                "system_prompt_sha256",
                "tool_search_version",
                "tool_preamble_sha256",
                "model_requests_attempted",
            ):
                if key in packet:
                    record[key] = packet[key]
        except Exception as exc:
            # Provider exceptions can echo credentials or endpoint URLs. Keep
            # their type, never their arbitrary body, in persisted/public errors.
            record.update(
                status="failed",
                outcome="error",
                answer=None,
                evidence=[],
                error=(
                    f"{type(exc).__name__}: investigation failed; "
                    "source, model, or sandbox unavailable."
                ),
            )
        finally:
            if progress is not None:
                record["usage"] = progress.export().get("token_usage", {})
            record.update(duration_seconds=time.monotonic() - started, finished_at=_now())
            try:
                self._save(record)
            finally:
                lock.close()

    def run(self, identifier: str) -> dict:
        path = _safe_path(self.runs, identifier, "run", ".json")
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict) or record.get("id") != identifier:
                raise ValueError("Run identity mismatch")
            if record.get("evidence"):
                _, manifest, docs = self._snapshot(record["paper_id"])
                if manifest["corpus_hash"] != record.get("corpus_hash") or manifest[
                    "raw_sources_hash"
                ] != record.get("raw_sha256"):
                    raise ValueError("Run source changed")
                record["evidence"] = _checked_evidence(record["evidence"], docs)
        except (OSError, ValueError, TypeError, KeyError):
            raise HTTPException(
                409, "Saved run or its evidence failed integrity validation"
            ) from None
        return _redact(record, self.secrets)


def create_paper_router(state_dir: Path, models_path: Path | None = None) -> APIRouter:
    """Attach to the local Studio; configuration is fixed at process startup."""
    service = _PaperService(state_dir, models_path)
    router = APIRouter()
    # An internal seam for isolated tests, never a web configuration surface.
    router.paper_service = service

    @router.get("/api/papers/config")
    def config():
        return service.config()

    @router.post("/api/papers", status_code=201)
    async def upload(request: Request, filename: str):
        _same_origin(request)
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_FILE_BYTES:
                raise HTTPException(413, "File exceeds the 5 MB size limit")
            body.extend(chunk)
        # PDF extraction is CPU work; do not block status polling during import.
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(service.upload, filename, bytes(body))

    @router.post("/api/papers/from-url", status_code=201)
    async def fetch_pdf(request: Request, body: _WebPaperRequest):
        _same_origin(request)
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(service.fetch_pdf, body.url)

    @router.get("/api/papers")
    def papers():
        rows = []
        for path in sorted(service.papers.glob("paper_*"), reverse=True):
            if not _ID.fullmatch(path.name) or path.is_symlink():
                continue
            try:
                rows.append(service.paper(path.name, full=False))
            except HTTPException:
                rows.append(
                    {
                        "id": path.name,
                        "title": "Unavailable source",
                        "filename": "",
                        "integrity_error": True,
                    }
                )
        return {"papers": sorted(rows, key=lambda row: row.get("created_at", ""), reverse=True)}

    @router.get("/api/papers/{paper_id}")
    def paper(paper_id: str):
        return service.paper(paper_id)

    @router.get("/api/papers/{paper_id}/raw")
    def paper_pdf(paper_id: str):
        snapshot, _, docs = service._snapshot(paper_id)
        doc = next(iter(docs.values()))
        if doc["metadata"]["content_type"] != "application/pdf":
            raise HTTPException(404, "This source has no original PDF")
        raw_path = snapshot / "raw" / f"{doc['doc_id']}.pdf"
        return FileResponse(
            raw_path,
            media_type="application/pdf",
            filename=doc["metadata"]["filename"],
            content_disposition_type="inline",
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @router.post("/api/paper-runs", status_code=202)
    def start(request: Request, body: _RunRequest):
        _same_origin(request)
        return service.start(body)

    @router.get("/api/paper-runs")
    def runs():
        rows = []
        for path in service.runs.glob("run_*.json"):
            if path.is_symlink() or not _ID.fullmatch(path.stem):
                continue
            try:
                record = json.loads(path.read_text())
                rows.append(
                    {
                        key: record.get(key)
                        for key in (
                            "id",
                            "paper_id",
                            "question",
                            "model_key",
                            "model_identity",
                            "created_at",
                            "status",
                            "outcome",
                            "duration_seconds",
                            "review_status",
                            "error",
                        )
                    }
                )
            except (OSError, ValueError, AttributeError):
                continue
        return {
            "runs": _redact(
                sorted(rows, key=lambda row: row.get("created_at") or "", reverse=True),
                service.secrets,
            )
        }

    @router.get("/api/paper-runs/{run_id}")
    def run(run_id: str):
        return service.run(run_id)

    @router.get("/api/paper-runs/{run_id}/export")
    def export(run_id: str):
        return JSONResponse(
            service.run(run_id),
            headers={
                "Content-Disposition": f'attachment; filename="{run_id}.json"',
            },
        )

    return router
