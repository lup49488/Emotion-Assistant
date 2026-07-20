"""FastAPI adapter for the chatbot domain services.

Run locally with: ``python api_server.py``.
"""
from __future__ import annotations

import json
import base64
import hashlib
import hmac
import logging
import os
import secrets
import tempfile
import time
import uuid
from collections.abc import Generator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Cookie, Depends, FastAPI, File, Header, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from api_contracts import (
    ApiError,
    ChatRequest,
    ChatResponse,
    ContractInfoResponse,
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationRenameRequest,
    ConversationResponse,
    ExportResponse,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    LongTermMemoryUpdateRequest,
    MemoryQualityResponse,
    MemorySnapshotResponse,
    MoodCheckinListResponse,
    MoodCheckinRequest,
    MoodCheckinResponse,
    PrivacyDeletionResponse,
    PrivacyDeleteRequest,
    PrivacyResponse,
    RagEvaluationRequest,
    RagEvaluationResponse,
    RagDocumentMutationResponse,
    RagQualityResponse,
    RagSearchRequest,
    RagSearchResponse,
    RagStatusResponse,
    SessionResponse,
    StatusResponse,
    UsageEventsResponse,
    UsageSummaryResponse,
    WeeklyMoodResponse,
)
from chatbot import handle_user_message_stream, latest_memory_receipt, make_model_config, session_store
from auth_store import access_key_version
from config import BASE_DIR, DEFAULT_LLM_PROVIDER
from api_usage_store import list_usage_events, usage_summary
from conversation_store import create_conversation, delete_conversation, get_conversation, list_conversations, remove_last_exchange, rename_conversation
from export_store import build_user_export_payload
from gui_auth import authorize
from knowledge_store import (
    SUPPORTED_EXTENSIONS,
    assess_knowledge_quality,
    copy_document_to_store,
    delete_document,
    diagnose_knowledge_search,
    knowledge_status,
    list_document_details,
    rebuild_knowledge_index,
)
from mood_store import add_mood_checkin, delete_mood_checkin, format_mood_fluctuation_analysis, format_weekly_mood_summary, get_weekly_mood_points, load_mood_checkins
from memory_store import record_memory_event
from model_warmup import start_api_background_warmup, warmup_status
from observability import chat_finished, get_request_id, request_finished, request_started, reset_request_id, runtime_metrics, set_request_id
from privacy_store import delete_all_user_data, privacy_summary
from rag_evaluation_store import latest_evaluation_report, run_evaluation
from sqlite_store import connection, storage_backend


logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"
API_CONTRACT_VERSION = "2026-07-20.1"
API_MAX_KNOWLEDGE_UPLOAD_BYTES = max(1_024, int(os.getenv("API_MAX_KNOWLEDGE_UPLOAD_BYTES", str(20 * 1024 * 1024))))
API_SESSION_TTL_SECONDS = max(60, int(os.getenv("API_SESSION_TTL_SECONDS", "43200")))
API_SESSION_COOKIE_NAME = os.getenv("API_SESSION_COOKIE_NAME", "chatbot_session").strip() or "chatbot_session"
API_CSRF_COOKIE_NAME = os.getenv("API_CSRF_COOKIE_NAME", "chatbot_csrf").strip() or "chatbot_csrf"
API_COOKIE_SECURE = os.getenv("API_COOKIE_SECURE", "false").lower() == "true"
API_COOKIE_SAMESITE = os.getenv("API_COOKIE_SAMESITE", "lax").strip().lower() or "lax"
if API_COOKIE_SAMESITE not in {"lax", "strict", "none"}:
    raise RuntimeError("API_COOKIE_SAMESITE 仅支持 lax、strict 或 none。")
if API_COOKIE_SAMESITE == "none" and not API_COOKIE_SECURE:
    raise RuntimeError("API_COOKIE_SAMESITE=none 时必须设置 API_COOKIE_SECURE=true。")
_configured_session_secret = os.getenv("API_SESSION_SECRET", "").strip()
if _configured_session_secret and len(_configured_session_secret) < 32:
    raise RuntimeError("API_SESSION_SECRET 至少需要 32 个字符。")
if not _configured_session_secret:
    logger.warning("API_SESSION_SECRET 未设置，正在使用进程临时密钥；服务重启后登录 Cookie 会失效。")
SESSION_SECRET = _configured_session_secret.encode("utf-8") or secrets.token_bytes(32)


def _cors_origins() -> list[str]:
    configured = os.getenv("API_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    if "*" in origins:
        raise RuntimeError("使用 Cookie 鉴权时 API_CORS_ORIGINS 不能包含 *。")
    return origins


@asynccontextmanager
async def lifespan(_: FastAPI):
    start_api_background_warmup()
    yield


app = FastAPI(
    title="Emotion-Aware Chat API",
    version=API_VERSION,
    description="Versioned API contract for the Emotion-Aware Chat application.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "X-CSRF-Token", "X-Request-ID"],
)


def _request_id_from_header(request: Request) -> str:
    supplied = request.headers.get("X-Request-ID", "").strip()
    if supplied and len(supplied) <= 64 and supplied.replace("-", "").replace("_", "").isalnum():
        return supplied
    return uuid.uuid4().hex


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = _request_id_from_header(request)
    token = set_request_id(request_id)
    started = time.perf_counter()
    request_started()
    response: Response | None = None
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.exception("event=http_request_failed request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
        raise
    finally:
        duration_ms = int((time.perf_counter() - started) * 1000)
        request_finished(status_code, duration_ms)
        logger.info(
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%s",
            request_id, request.method, request.url.path, status_code, duration_ms,
        )
        reset_request_id(token)


def _error_code(status_code: int) -> str:
    return {
        401: "authentication_required",
        403: "csrf_validation_failed",
        404: "not_found",
        422: "validation_error",
    }.get(status_code, "request_failed")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    details = exc.detail if isinstance(exc.detail, list) else None
    payload = ApiError(code=_error_code(exc.status_code), message=message, details=details)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump(exclude_none=True))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    payload = ApiError(
        code="request_validation_error",
        message="Request validation failed.",
        details=exc.errors(),
    )
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=payload.model_dump())


def _custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes, description=app.description)
    schema["info"]["x-contract-version"] = API_CONTRACT_VERSION
    schema["info"]["x-api-version"] = "v1"
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _sign_session(payload: dict[str, Any]) -> str:
    encoded = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = hmac.new(SESSION_SECRET, encoded.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded}.{_b64encode(signature)}"


def _read_signed_session(cookie_value: str) -> dict[str, Any] | None:
    encoded, separator, encoded_signature = (cookie_value or "").partition(".")
    if not separator or not encoded or not encoded_signature:
        return None
    try:
        expected = hmac.new(SESSION_SECRET, encoded.encode("ascii"), hashlib.sha256).digest()
        supplied = _b64decode(encoded_signature)
        if not hmac.compare_digest(expected, supplied):
            return None
        payload = json.loads(_b64decode(encoded).decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        return payload
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None


def _issue_session(user_id: str) -> tuple[str, int]:
    credential_version = access_key_version(user_id)
    if not credential_version:
        raise RuntimeError("无法读取访问凭据版本。")
    expires_at = int(time.time()) + API_SESSION_TTL_SECONDS
    return _sign_session({"sub": user_id, "exp": expires_at, "ver": credential_version}), API_SESSION_TTL_SECONDS


def _set_session_cookies(response: Response, user_id: str) -> int:
    token, expires_in = _issue_session(user_id)
    response.set_cookie(
        key=API_SESSION_COOKIE_NAME, value=token, max_age=expires_in, path="/",
        secure=API_COOKIE_SECURE, httponly=True, samesite=API_COOKIE_SAMESITE,
    )
    response.set_cookie(
        key=API_CSRF_COOKIE_NAME, value=secrets.token_urlsafe(24), max_age=expires_in, path="/",
        secure=API_COOKIE_SECURE, httponly=False, samesite=API_COOKIE_SAMESITE,
    )
    return expires_in


def _current_user(
    session_cookie: Annotated[str | None, Cookie(alias=API_SESSION_COOKIE_NAME)] = None,
) -> str:
    payload = _read_signed_session(session_cookie or "")
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Signed session cookie is required.")
    try:
        user_id = str(payload["sub"])
        expires_at = int(payload["exp"])
        credential_version = str(payload["ver"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signed session cookie.") from None
    if expires_at <= int(time.time()) or access_key_version(user_id) != credential_version:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session is missing or expired.")
    return user_id


def _csrf_protected_user(
    user_id: CurrentUser,
    csrf_cookie: Annotated[str | None, Cookie(alias=API_CSRF_COOKIE_NAME)] = None,
    csrf_token: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
) -> str:
    if not csrf_cookie or not csrf_token or not hmac.compare_digest(csrf_cookie, csrf_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token is invalid or missing.")
    return user_id


CurrentUser = Annotated[str, Depends(_current_user)]
CsrfCurrentUser = Annotated[str, Depends(_csrf_protected_user)]


def _storage_component() -> dict[str, str]:
    try:
        if storage_backend() == "sqlite":
            with connection() as conn:
                conn.execute("SELECT 1")
        else:
            (BASE_DIR / "users").mkdir(parents=True, exist_ok=True)
        return {"status": "ok"}
    except Exception as exc:
        logger.exception("event=health_storage_failed request_id=%s", get_request_id())
        return {"status": "degraded", "detail": type(exc).__name__}


def _health_payload() -> dict[str, Any]:
    storage = _storage_component()
    preload = warmup_status()
    warmup_state = "degraded" if "degraded" in preload.values() else "disabled" if preload and set(preload.values()) == {"disabled"} else "pending" if any(value in {"pending", "running"} for value in preload.values()) else "ok"
    components = {
        "storage": storage,
        "warmup": {"status": warmup_state},
        "rag": {"status": "ok", "detail": knowledge_status()},
    }
    overall = "degraded" if any(component["status"] == "degraded" for component in components.values()) else "ok"
    return {
        "status": overall,
        "storage_backend": storage_backend(),
        "components": components,
        "metrics": runtime_metrics(),
    }


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    """Public liveness/readiness report without configuration secrets."""
    return _health_payload()


@app.get("/api/v1/status", response_model=StatusResponse)
def api_status() -> dict[str, Any]:
    """Versioned operational status for reverse proxies and monitoring."""
    return {**_health_payload(), "api_version": "v1"}


@app.get("/api/v1/contract", response_model=ContractInfoResponse)
def contract_info() -> dict[str, str]:
    return {"api_version": "v1", "contract_version": API_CONTRACT_VERSION, "openapi_path": "/openapi.json"}


@app.post("/api/v1/auth/login", response_model=LoginResponse)
def login(request: LoginRequest, response: Response) -> dict[str, Any]:
    user_id, auth_error = authorize(request.user_id, request.access_key)
    if auth_error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=auth_error)
    expires_in = _set_session_cookies(response, user_id)
    return {"token_type": "cookie", "expires_in": expires_in, "user_id": user_id}


@app.post("/api/v1/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response, _: CsrfCurrentUser) -> None:
    cookie_settings = {"path": "/", "secure": API_COOKIE_SECURE, "samesite": API_COOKIE_SAMESITE}
    response.delete_cookie(API_SESSION_COOKIE_NAME, httponly=True, **cookie_settings)
    response.delete_cookie(API_CSRF_COOKIE_NAME, **cookie_settings)


@app.get("/api/v1/auth/session", response_model=SessionResponse)
def session_info(user_id: CurrentUser) -> dict[str, str]:
    return {"user_id": user_id, "authentication": "signed_cookie"}


def _rollback_short_term_exchange(user_id: str, user_text: str) -> None:
    """Drop the failed user/assistant pair from working memory before a retry.

    Otherwise the model would see its own failed reply plus a duplicated user
    turn in the short-term context while regenerating.
    """
    with session_store.session(user_id) as state:
        if (
            len(state.history) >= 2
            and state.history[-2].get("role") == "user"
            and str(state.history[-2].get("content", "")) == user_text
            and state.history[-1].get("role") == "assistant"
        ):
            del state.history[-2:]


def _chat_chunks(user_id: str, request: ChatRequest) -> Generator[str, None, None]:
    if request.retry_last_response:
        remove_last_exchange(user_id, request.conversation_id, request.message)
        _rollback_short_term_exchange(user_id, request.message)
    config = make_model_config(
        provider=request.provider or DEFAULT_LLM_PROVIDER,
        model=request.model,
        api_key=request.api_key,
        base_url=request.base_url,
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_new_tokens,
        user_id=user_id,
    )
    yield from handle_user_message_stream(
        user_id,
        request.message,
        model_config=config,
        use_knowledge=request.use_knowledge,
        use_style=request.use_style,
        conversation_id=request.conversation_id,
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest, user_id: CsrfCurrentUser) -> dict[str, str]:
    started = time.perf_counter()
    try:
        reply = "".join(_chat_chunks(user_id, request))
    except Exception:
        duration_ms = int((time.perf_counter() - started) * 1000)
        chat_finished(False, duration_ms, streaming=False)
        logger.exception("event=chat_failed request_id=%s provider=%s streaming=false", get_request_id(), request.provider or DEFAULT_LLM_PROVIDER)
        raise
    duration_ms = int((time.perf_counter() - started) * 1000)
    chat_finished(True, duration_ms, streaming=False)
    logger.info("event=chat_completed request_id=%s provider=%s streaming=false duration_ms=%s reply_chars=%s", get_request_id(), request.provider or DEFAULT_LLM_PROVIDER, duration_ms, len(reply))
    payload = {"reply": reply}
    if request.show_memory_receipt:
        with session_store.session(user_id) as state:
            payload["memory_receipt"] = latest_memory_receipt(state)
    return payload


def _memory_receipt(user_id: str) -> str:
    with session_store.session(user_id) as state:
        return latest_memory_receipt(state)


@app.post(
    "/api/v1/chat/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-sent events: `chunk` ({text}), optional `receipt` ({text}), then `done` ({}). On generation failure emits `error` ({code, message}).",
            "content": {"text/event-stream": {}},
        }
    },
)
async def chat_stream(request: ChatRequest, user_id: CsrfCurrentUser) -> StreamingResponse:
    request_id = get_request_id()

    async def event_source():
        token = set_request_id(request_id)
        started = time.perf_counter()
        try:
            # The chat generator does blocking model inference; iterate it in a
            # worker thread so the async event loop stays responsive.
            async for chunk in iterate_in_threadpool(_chat_chunks(user_id, request)):
                yield f"event: chunk\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            if request.show_memory_receipt:
                receipt = await run_in_threadpool(_memory_receipt, user_id)
                yield f"event: receipt\ndata: {json.dumps({'text': receipt}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
            duration_ms = int((time.perf_counter() - started) * 1000)
            chat_finished(True, duration_ms, streaming=True)
            logger.info("event=chat_completed request_id=%s provider=%s streaming=true duration_ms=%s", request_id, request.provider or DEFAULT_LLM_PROVIDER, duration_ms)
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            chat_finished(False, duration_ms, streaming=True)
            logger.exception("event=chat_failed request_id=%s provider=%s streaming=true", request_id, request.provider or DEFAULT_LLM_PROVIDER)
            error = ApiError(code="generation_failed", message=str(exc))
            yield f"event: error\ndata: {error.model_dump_json()}\n\n"
        finally:
            reset_request_id(token)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/api/v1/memory/quality", response_model=MemoryQualityResponse)
def memory_quality(user_id: CurrentUser) -> dict[str, str]:
    # The signed session cookie has already authenticated the user.
    with session_store.session(user_id) as state:
        from gui_memory import build_memory_quality_report

        return {"report": build_memory_quality_report(user_id, state)}


@app.post("/api/v1/mood/checkins", response_model=MoodCheckinResponse)
def create_mood_checkin(request: MoodCheckinRequest, user_id: CsrfCurrentUser) -> dict[str, Any]:
    try:
        record = add_mood_checkin(
            user_id=user_id,
            mood=request.mood,
            intensity=request.intensity,
            note=request.note,
            checkin_date=request.checkin_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"record": record}


@app.get("/api/v1/mood/checkins", response_model=MoodCheckinListResponse)
def list_mood_records(user_id: CurrentUser) -> dict[str, Any]:
    return {"records": load_mood_checkins(user_id)}


@app.delete("/api/v1/mood/checkins/{checkin_date}", status_code=status.HTTP_204_NO_CONTENT)
def remove_mood_checkin(checkin_date: str, user_id: CsrfCurrentUser) -> None:
    try:
        deleted = delete_mood_checkin(user_id, checkin_date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Mood check-in was not found.")


@app.get("/api/v1/mood/weekly", response_model=WeeklyMoodResponse)
def weekly_mood(
    user_id: CurrentUser,
    end_date: str | None = None,
    days: int = Query(default=7, ge=1, le=31),
) -> dict[str, Any]:
    try:
        points = get_weekly_mood_points(user_id, end_date=end_date, days=days)
        return {
            "points": points,
            "summary": format_weekly_mood_summary(user_id, end_date=end_date, days=days),
            "analysis": format_mood_fluctuation_analysis(points),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/conversations", response_model=ConversationListResponse)
def conversations(user_id: CurrentUser) -> dict[str, Any]:
    return {"conversations": list_conversations(user_id)}


@app.post("/api/v1/conversations", response_model=ConversationResponse)
def new_conversation(request: ConversationCreateRequest, user_id: CsrfCurrentUser) -> dict[str, Any]:
    return {"conversation": create_conversation(user_id, request.title)}


@app.get("/api/v1/conversations/{conversation_id}", response_model=ConversationResponse)
def conversation(conversation_id: str, user_id: CurrentUser) -> dict[str, Any]:
    result = get_conversation(user_id, conversation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation was not found.")
    return {"conversation": result}


@app.put("/api/v1/conversations/{conversation_id}", response_model=ConversationResponse)
def rename_saved_conversation(
    conversation_id: str, request: ConversationRenameRequest, user_id: CsrfCurrentUser,
) -> dict[str, Any]:
    result = rename_conversation(user_id, conversation_id, request.title)
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation was not found.")
    return {"conversation": result}


@app.delete("/api/v1/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_conversation(conversation_id: str, user_id: CsrfCurrentUser) -> None:
    if not delete_conversation(user_id, conversation_id):
        raise HTTPException(status_code=404, detail="Conversation was not found.")


@app.get("/api/v1/memory", response_model=MemorySnapshotResponse)
def memory_snapshot(user_id: CurrentUser) -> dict[str, Any]:
    with session_store.session(user_id) as state:
        return {
            "history": list(state.history),
            "emotion_memory": list(state.emotion_memory),
            "long_memory": list(state.long_memory),
            "stable_profile": list(state.stable_profile),
            "interest_memory": list(state.interest_store.items),
            "memory_events": list(state.memory_events),
        }


@app.put("/api/v1/memory/long-term", response_model=MemorySnapshotResponse)
def update_long_term_memory(request: LongTermMemoryUpdateRequest, user_id: CsrfCurrentUser) -> dict[str, Any]:
    normalized_items: list[dict[str, Any]] = []
    for item in request.items:
        text = str(item.get("text", "")).strip()
        if not text:
            raise HTTPException(status_code=422, detail="Each long-term memory needs non-empty text.")
        if len(text) > 2_000:
            raise HTTPException(status_code=422, detail="A long-term memory cannot exceed 2000 characters.")
        normalized_items.append({
            **item,
            "text": text,
            "time": str(item.get("time") or datetime.now().isoformat(timespec="seconds")),
            "kind": str(item.get("kind") or "manual"),
        })

    with session_store.session(user_id) as state:
        state.long_memory = normalized_items
        record_memory_event(
            state,
            section="long",
            action="updated",
            text=f"Long-term memories updated ({len(normalized_items)})",
            reason="User edited long-term memory from the personal-data panel.",
        )
        return {
            "history": list(state.history),
            "emotion_memory": list(state.emotion_memory),
            "long_memory": list(state.long_memory),
            "stable_profile": list(state.stable_profile),
            "interest_memory": list(state.interest_store.items),
            "memory_events": list(state.memory_events),
        }


@app.get("/api/v1/rag/status", response_model=RagStatusResponse)
def rag_status(_: CurrentUser) -> dict[str, Any]:
    return {"status": knowledge_status(), "documents": list_document_details()}


@app.post("/api/v1/rag/documents", response_model=RagDocumentMutationResponse)
async def upload_rag_document(file: Annotated[UploadFile, File(...)], _: CsrfCurrentUser) -> dict[str, Any]:
    """Store one supported document and rebuild the derived RAG index."""
    filename = (file.filename or "").strip()
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=422, detail=f"Unsupported document type. Allowed: {allowed}.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=422, detail="The uploaded document is empty.")
    if len(contents) > API_MAX_KNOWLEDGE_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail="The uploaded document exceeds the configured size limit.")

    def _store_and_rebuild() -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="rag-upload-") as temporary_directory:
            temporary_path = Path(temporary_directory) / Path(filename).name
            temporary_path.write_bytes(contents)
            # copy_document_to_store normalizes the filename and prevents storage outside the RAG directory.
            stored_path = copy_document_to_store(temporary_path)
        result = rebuild_knowledge_index()
        result["uploaded"] = stored_path.name
        return result

    try:
        # Index rebuilding embeds every chunk and can take seconds; keep the
        # event loop responsive by running the blocking work in a thread.
        result = await run_in_threadpool(_store_and_rebuild)
        return {"result": result}
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/v1/rag/rebuild", response_model=RagDocumentMutationResponse)
def rebuild_rag(_: CsrfCurrentUser) -> dict[str, Any]:
    return {"result": rebuild_knowledge_index()}


@app.delete("/api/v1/rag/documents/{name}", response_model=RagDocumentMutationResponse)
def remove_rag_document(name: str, _: CsrfCurrentUser) -> dict[str, Any]:
    try:
        return {"result": delete_document(name)}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404 if isinstance(exc, FileNotFoundError) else 422, detail=str(exc)) from exc


@app.get("/api/v1/rag/quality", response_model=RagQualityResponse)
def rag_quality(_: CurrentUser) -> dict[str, Any]:
    return assess_knowledge_quality()


@app.post("/api/v1/rag/search", response_model=RagSearchResponse)
def rag_search(request: RagSearchRequest, _: CsrfCurrentUser) -> dict[str, Any]:
    return diagnose_knowledge_search(
        request.query, top_k=request.top_k, threshold=request.threshold,
        candidate_multiplier=request.candidate_multiplier,
    )


@app.get("/api/v1/rag/evaluations/latest", response_model=RagEvaluationResponse)
def rag_latest_evaluation(_: CurrentUser) -> dict[str, Any]:
    return {"report": latest_evaluation_report()}


@app.post("/api/v1/rag/evaluations/run", response_model=RagEvaluationResponse)
def rag_run_evaluation(request: RagEvaluationRequest, _: CsrfCurrentUser) -> dict[str, Any]:
    try:
        return {"report": run_evaluation(
            top_k=request.top_k, threshold=request.threshold,
            candidate_multiplier=request.candidate_multiplier,
        )}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/usage", response_model=UsageSummaryResponse)
def api_usage(user_id: CurrentUser) -> dict[str, Any]:
    return usage_summary(user_id)


@app.get("/api/v1/usage/events", response_model=UsageEventsResponse)
def api_usage_event_list(user_id: CurrentUser, limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
    return {"events": list_usage_events(user_id, limit=limit)}


@app.get("/api/v1/privacy", response_model=PrivacyResponse)
def privacy(user_id: CurrentUser) -> dict[str, Any]:
    return privacy_summary(user_id)


@app.get("/api/v1/export", response_model=ExportResponse)
def export_data(user_id: CurrentUser) -> dict[str, Any]:
    return build_user_export_payload(user_id)


@app.delete("/api/v1/privacy/data", response_model=PrivacyDeletionResponse)
def delete_privacy_data(request: PrivacyDeleteRequest, user_id: CsrfCurrentUser) -> dict[str, Any]:
    if request.confirmation.strip() != "DELETE":
        raise HTTPException(status_code=422, detail="Confirmation must be DELETE.")
    return {"deleted": delete_all_user_data(user_id)}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host=os.getenv("API_SERVER_HOST", "127.0.0.1"),
        port=int(os.getenv("API_SERVER_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
