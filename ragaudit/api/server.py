"""HTTP API.

Deliberately small: a chat endpoint, a structured query endpoint, figure
retrieval, and the two health probes an orchestrator needs. ``/healthz``
answers whether the process is alive; ``/readyz`` answers whether it can
actually serve a question, which is a different question and was previously
unanswerable at all (D03).
"""

from __future__ import annotations

import logging
import secrets
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _sse(event: str, data: dict) -> str:
    """One Server-Sent Event frame."""
    import json as _json

    return f"event: {event}\ndata: {_json.dumps(data, default=str)}\n\n"

from ragaudit.app import Application
from ragaudit.observability import new_trace_id

logger = logging.getLogger("ragaudit.api")

_app_singleton: Application | None = None

#: One question at a time. The catalogue shares a single SQLite connection
#: across FastAPI's worker threads, which is not safe under concurrent use,
#: and a local Ollama processes requests one by one anyway.
_ask_lock = threading.Lock()


def get_application() -> Application:
    global _app_singleton
    if _app_singleton is None:
        _app_singleton = Application()
    return _app_singleton


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    report_ids: list[str] | None = Field(
        default=None,
        description="Restrict the answer to these reports. Omit to search everything readable.",
    )


class CitationOut(BaseModel):
    chunk_id: str
    report_id: str
    field_path: str
    quote: str
    finding_id: str | None = None


class FigureOut(BaseModel):
    token: str
    report_id: str
    page: int
    available: bool
    url: str | None = None


class AskResponse(BaseModel):
    answer: str
    session_id: str
    route: str
    abstained: bool
    abstention_reason: str = ""
    citations: list[CitationOut] = []
    figures: list[FigureOut] = []
    review_flags: list[str] = []
    blocks: list[dict] = []      # tables, charts, metrics the UI draws
    steps: list[dict] = []       # the tool-call trace, for an auditable answer
    trace_id: str
    latency_s: float
    model: str = ""


def _answer_to_response(answer, session_id: str, fallback_trace: str) -> AskResponse:
    return AskResponse(
        answer=answer.text,
        session_id=session_id,
        route=answer.route,
        abstained=answer.abstained,
        abstention_reason=answer.abstention_reason,
        citations=[
            CitationOut(chunk_id=c.chunk_id, report_id=c.report_id, field_path=c.field_path,
                        quote=c.quote, finding_id=c.finding_id)
            for c in answer.citations
        ],
        figures=[
            FigureOut(token=f.token, report_id=f.report_id, page=f.page, available=f.available,
                      url=f"/figures/{f.report_id}/{f.token.strip('[]')}" if f.available else None)
            for f in answer.figures
        ],
        review_flags=answer.review_flags,
        blocks=answer.blocks,
        steps=answer.steps,
        trace_id=answer.trace_id or fallback_trace,
        latency_s=answer.latency_s,
        model=answer.model,
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def create_app(application: Application | None = None) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_api):
        """Load models in the background while the console opens, so the
        first real question is not the slow one."""
        def run() -> None:
            app_ = get_application()
            with _ask_lock:
                app_.warm_up()

        threading.Thread(target=run, name="warm-up", daemon=True).start()
        yield

    api = FastAPI(
        lifespan=lifespan,
        title="Audit Report Assistant",
        version="0.1.0",
        description="Conversational question answering over cybersecurity audit reports.",
    )
    if application is not None:
        global _app_singleton
        _app_singleton = application

    #: Reachable without a key: the console page itself (it then asks for the
    #: key), liveness, and the API docs.
    open_paths = {"/", "/healthz", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}

    @api.middleware("http")
    async def require_api_key(request: Request, call_next):
        """Optional shared-secret auth (settings.api.api_key / RAGAUDIT_API__API_KEY).

        Off when no key is configured. The key may come as an ``X-API-Key``
        header, or as ``?key=`` for figure images (an <img> cannot send
        headers). Compared in constant time.
        """
        expected = get_application().settings.api.api_key
        if expected and request.url.path not in open_paths:
            supplied = request.headers.get("x-api-key") or request.query_params.get("key") or ""
            if not secrets.compare_digest(supplied.encode(), expected.encode()):
                return JSONResponse({"detail": "missing or wrong API key"}, status_code=401)
        return await call_next(request)

    @api.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """The demo console.

        Served from this package rather than a separate front-end build so the
        whole system starts with one command in an air-gapped environment.
        """
        page = STATIC_DIR / "index.html"
        if not page.exists():
            return HTMLResponse(
                "<p>UI assets are missing. The API still works; see /docs.</p>", status_code=200
            )
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @api.get("/healthz")
    def healthz() -> dict:
        """Liveness: the process is up. Does not touch dependencies."""
        return {"status": "ok"}

    @api.get("/readyz")
    def readyz(app: Application = Depends(get_application)) -> JSONResponse:
        """Readiness: every dependency needed to answer a question is working."""
        report = app.readiness()
        return JSONResponse(report, status_code=200 if report["ready"] else 503)

    @api.get("/stats")
    def stats(app: Application = Depends(get_application)) -> dict:
        return app.catalog.stats()

    @api.get("/reports")
    def reports(app: Application = Depends(get_application)) -> dict:
        with _ask_lock:
            rows = app.catalog.reports_overview()
        return {"reports": [
            {
                "report_id": r["report_id"],
                "title": r["title"],
                "classification": r["classification"],
                "release_date": r["release_date"],
                "report_type": r["report_type"],
                "status": r["ingest_status"],
                "findings": r["findings"],
            }
            for r in rows
        ]}

    @api.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest, app: Application = Depends(get_application)) -> AskResponse:
        trace = new_trace_id()
        try:
            with _ask_lock:
                chat = app.build_chat()
                answer, session = chat.ask(
                    request.question,
                    session_id=request.session_id,
                    report_ids=request.report_ids,
                )
        except Exception as exc:
            logger.exception("ask failed", extra={"trace_id": trace})
            raise HTTPException(status_code=500, detail=f"internal error (trace {trace})") from exc

        return _answer_to_response(answer, session.session_id, trace)

    @api.post("/ask/stream")
    def ask_stream(request: AskRequest, app: Application = Depends(get_application)):
        """Server-Sent Events: progress, then the final answer.

        An agent run on CPU is tens of seconds; a blank blocking POST reads as
        a hang. This streams a 'started' event, then the final 'answer' event.
        (Per-step streaming is a later refinement; the transport is in place so
        the UI shows progress rather than a frozen spinner.)
        """
        import json as _json

        from fastapi.responses import StreamingResponse

        trace = new_trace_id()

        def events():
            yield _sse("status", {"state": "working", "trace_id": trace})
            try:
                with _ask_lock:
                    chat = app.build_chat()
                    answer, session = chat.ask(
                        request.question, session_id=request.session_id, report_ids=request.report_ids,
                    )
            except Exception as exc:  # never leak a stack trace to the stream
                logger.exception("ask_stream failed", extra={"trace_id": trace})
                yield _sse("error", {"detail": f"internal error (trace {trace})"})
                return
            payload = _answer_to_response(answer, session.session_id, trace).model_dump()
            # Steps are emitted as their own events so the UI can show the trace.
            for step in answer.steps:
                yield _sse("step", step)
            yield _sse("answer", payload)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @api.get("/findings")
    def findings(
        app: Application = Depends(get_application),
        report_id: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        cwe: str | None = None,
        limit: int = Query(default=100, le=1000),
    ) -> dict:
        """Exact structured query. No model involved, so the result is
        reproducible and can be re-run by hand against the catalogue."""
        filters = dict(
            report_ids=[report_id] if report_id else None,
            severities=[severity] if severity else None,
            statuses=[status] if status else None,
            cwe=cwe,
        )
        with _ask_lock:
            rows = app.catalog.findings(**filters, limit=limit)
            total = app.catalog.count_findings(**filters)
        return {
            # The true total, not the length of the (limited) page.
            "count": total,
            "returned": len(rows),
            "findings": [
                {
                    "report_id": f.report_id,
                    "finding_id": f.finding_id,
                    "title": f.title,
                    "severity": f.severity.value,
                    "status": f.status.value,
                    "cwe_ids": f.cwe_ids,
                    "cve_ids": f.cve_ids,
                }
                for f in rows
            ],
        }

    @api.get("/figures/{report_id}/{token}")
    def figure(report_id: str, token: str, app: Application = Depends(get_application)):
        rows = app.catalog.figures_for_tokens(report_id, [f"[{token}]"])
        if not rows:
            raise HTTPException(status_code=404, detail="figure not referenced by this report")
        asset = rows[0]["asset_path"]
        if not asset:
            # Referenced but never staged. Reported honestly rather than as a
            # generic 404, so the caller can tell the two cases apart (D07).
            raise HTTPException(
                status_code=409,
                detail="figure is referenced by the report but no image asset has been staged",
            )
        path = Path(asset)
        if not path.exists():
            raise HTTPException(status_code=410, detail="staged asset is missing from disk")
        return FileResponse(path)

    return api


app = create_app()
