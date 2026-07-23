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
from starlette.middleware.trustedhost import TrustedHostMiddleware

from api_contracts import (
    ApiError,
    BackgroundJobListResponse,
    BackgroundJobResponse,
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
    ObservabilitySummaryResponse,
    OperationsDashboardResponse,
    PrivacyDeletionResponse,
    PrivacyDeleteRequest,
    PrivacyResponse,
    RagEvaluationRequest,
    RagEvaluationResponse,
    RagFeedbackRequest,
    RagFeedbackResponse,
    RagFeedbackSummaryResponse,
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
from config import API_ENABLE_DOCS, API_MAX_REQUEST_BYTES, API_OPERATIONS_USER_IDS, API_PUBLIC_MODE, API_RAG_ADMIN_USER_IDS, API_TRUSTED_HOSTS, API_TRUST_PROXY_HEADERS, BASE_DIR, DEFAULT_LLM_PROVIDER
from auth_rate_limit import clear_login_failures, login_allowed, record_login_failure
from api_usage_store import list_usage_events, usage_summary
from conversation_store import create_conversation, delete_conversation, get_conversation, list_conversations, remove_last_exchange, rename_conversation
from export_store import build_user_export_payload
from gui_auth import authorize
from knowledge_store import (
    SUPPORTED_EXTENSIONS,
    assess_knowledge_quality,
    copy_document_to_store,
    delete_document_from_store,
    diagnose_knowledge_search,
    knowledge_status,
    list_document_details,
    build_knowledge_bundle,
    rebuild_with_release_gate,
    release_gate_status,
)
from job_store import get_job, job_manager, list_jobs, mark_interrupted_jobs
from mood_store import add_mood_checkin, delete_mood_checkin, format_mood_fluctuation_analysis, format_weekly_mood_summary, get_weekly_mood_points, load_mood_checkins
from memory_store import reconcile_memory_ownership, record_memory_event
from model_warmup import start_api_background_warmup, warmup_status
from observability import chat_finished, get_request_id, request_finished, request_started, reset_request_id, runtime_metrics, set_request_id
from observability_store import observability_summary, record_http_event
from operations_store import operations_dashboard
from privacy_store import delete_all_user_data, privacy_summary
from rag_evaluation_store import latest_evaluation_report, run_evaluation
from rag_feedback_store import create_citation_trace, feedback_summary, submit_feedback
from service_errors import ServiceError
from sqlite_store import connection, storage_backend


logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"
API_CONTRACT_VERSION = "2026-07-21.6"
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


def _trusted_hosts() -> list[str]:
    return [host.strip() for host in API_TRUSTED_HOSTS.split(",") if host.strip()]


if API_PUBLIC_MODE:
    if not API_COOKIE_SECURE:
        raise RuntimeError("公网模式要求 API_COOKIE_SECURE=true。")
    if not _configured_session_secret:
        raise RuntimeError("公网模式要求设置 API_SESSION_SECRET。")
    if not _trusted_hosts() or any(host in {"*", "localhost", "127.0.0.1", "[::1]"} for host in _trusted_hosts()):
        raise RuntimeError("公网模式要求 API_TRUSTED_HOSTS 仅包含明确的公网域名。")


def _cors_origins() -> list[str]:
    configured = os.getenv("API_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    if "*" in origins:
        raise RuntimeError("使用 Cookie 鉴权时 API_CORS_ORIGINS 不能包含 *。")
    return origins


@asynccontextmanager
async def lifespan(_: FastAPI):
    interrupted = mark_interrupted_jobs()
    if interrupted:
        logger.warning("event=background_jobs_interrupted count=%s", interrupted)
    start_api_background_warmup()
    yield


app = FastAPI(
    title="Emotion-Aware Chat API",
    version=API_VERSION,
    description="Versioned API contract for the Emotion-Aware Chat application.",
    lifespan=lifespan,
    docs_url="/docs" if API_ENABLE_DOCS else None,
    redoc_url="/redoc" if API_ENABLE_DOCS else None,
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=_trusted_hosts())
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


def _client_ip(request: Request) -> str:
    if API_TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("CF-Connecting-IP") or request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def _require_operations_user(user_id: str) -> str:
    allowed = {item.strip() for item in API_OPERATIONS_USER_IDS.split(",") if item.strip()}
    if user_id not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Operations dashboard access is not permitted.")
    return user_id


def _rag_admin_users() -> set[str]:
    """Use a dedicated RAG allowlist, falling back to operations administrators."""
    configured = API_RAG_ADMIN_USER_IDS or API_OPERATIONS_USER_IDS
    return {item.strip() for item in configured.split(",") if item.strip()}


def _can_manage_knowledge(user_id: str) -> bool:
    return user_id in _rag_admin_users()


def _require_knowledge_admin(user_id: str) -> str:
    if not _can_manage_knowledge(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Knowledge-base management access is not permitted.")
    return user_id


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = _request_id_from_header(request)
    token = set_request_id(request_id)
    started = time.perf_counter()
    request_started()
    response: Response | None = None
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    try:
        content_length = request.headers.get("content-length")
        try:
            too_large = bool(content_length and int(content_length) > API_MAX_REQUEST_BYTES)
        except ValueError:
            too_large = True
        if too_large:
            return JSONResponse(status_code=413, content={"code": "request_too_large", "message": "Request body exceeds the configured size limit."})
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
        record_http_event(request_id=request_id, method=request.method, path=request.url.path, status_code=status_code, duration_ms=duration_ms)
        logger.info(
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%s",
            request_id, request.method, request.url.path, status_code, duration_ms,
        )
        reset_request_id(token)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
    if API_COOKIE_SECURE:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


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


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
    payload = ApiError(code=exc.code, message=str(exc), retryable=exc.retryable)
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload.model_dump())


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


@app.get("/api/v1/observability/summary", response_model=ObservabilitySummaryResponse)
def observability_summary_endpoint(user_id: CurrentUser, days: int = Query(default=7, ge=1, le=90)) -> dict[str, Any]:
    _require_operations_user(user_id)
    return observability_summary(days=days)


@app.get("/api/v1/operations/dashboard", response_model=OperationsDashboardResponse)
def operations_dashboard_endpoint(user_id: CurrentUser, days: int = Query(default=7, ge=1, le=90)) -> dict[str, Any]:
    _require_operations_user(user_id)
    return operations_dashboard(days=days)


@app.get("/api/v1/contract", response_model=ContractInfoResponse)
def contract_info() -> dict[str, str]:
    return {"api_version": "v1", "contract_version": API_CONTRACT_VERSION, "openapi_path": "/openapi.json"}


@app.post("/api/v1/auth/login", response_model=LoginResponse)
def login(request: LoginRequest, response: Response, raw_request: Request) -> dict[str, Any]:
    client_ip = _client_ip(raw_request)
    allowed, retry_after = login_allowed(client_ip, request.user_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )
    user_id, auth_error = authorize(request.user_id, request.access_key)
    if auth_error:
        record_login_failure(client_ip, request.user_id)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=auth_error)
    clear_login_failures(client_ip, request.user_id)
    expires_in = _set_session_cookies(response, user_id)
    return {"token_type": "cookie", "expires_in": expires_in, "user_id": user_id}


@app.post("/api/v1/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response, _: CsrfCurrentUser) -> None:
    cookie_settings = {"path": "/", "secure": API_COOKIE_SECURE, "samesite": API_COOKIE_SAMESITE}
    response.delete_cookie(API_SESSION_COOKIE_NAME, httponly=True, **cookie_settings)
    response.delete_cookie(API_CSRF_COOKIE_NAME, **cookie_settings)


@app.get("/api/v1/auth/session", response_model=SessionResponse)
def session_info(user_id: CurrentUser) -> dict[str, Any]:
    allowed = {item.strip() for item in API_OPERATIONS_USER_IDS.split(",") if item.strip()}
    return {
        "user_id": user_id,
        "authentication": "signed_cookie",
        "can_access_operations": user_id in allowed,
        "can_manage_knowledge": _can_manage_knowledge(user_id),
    }


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


def _chat_chunks(user_id: str, request: ChatRequest, knowledge_context: str | None = None) -> Generator[str, None, None]:
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
        knowledge_context=knowledge_context,
    )


def _knowledge_bundle(request: ChatRequest) -> dict[str, Any]:
    return build_knowledge_bundle(request.message) if request.use_knowledge else {"context": "", "citations": []}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest, user_id: CsrfCurrentUser) -> dict[str, str]:
    bundle = _knowledge_bundle(request)
    started = time.perf_counter()
    try:
        reply = "".join(_chat_chunks(user_id, request, bundle["context"]))
    except Exception:
        duration_ms = int((time.perf_counter() - started) * 1000)
        chat_finished(False, duration_ms, streaming=False)
        logger.exception("event=chat_failed request_id=%s provider=%s streaming=false", get_request_id(), request.provider or DEFAULT_LLM_PROVIDER)
        raise
    duration_ms = int((time.perf_counter() - started) * 1000)
    chat_finished(True, duration_ms, streaming=False)
    logger.info("event=chat_completed request_id=%s provider=%s streaming=false duration_ms=%s reply_chars=%s", get_request_id(), request.provider or DEFAULT_LLM_PROVIDER, duration_ms, len(reply))
    trace_id = create_citation_trace(user_id, request.conversation_id, bundle["citations"])
    payload = {"reply": reply, "citations": bundle["citations"], "citation_trace_id": trace_id}
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
            bundle = await run_in_threadpool(_knowledge_bundle, request)
            # 始终复用 bundle 已检索出的上下文（与非流式 /chat 一致），
            # 避免 use_knowledge=True 但检索为空时在 chatbot 内部重复检索。
            generator = _chat_chunks(user_id, request, bundle["context"])
            async for chunk in iterate_in_threadpool(generator):
                yield f"event: chunk\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            trace_id = await run_in_threadpool(create_citation_trace, user_id, request.conversation_id, bundle["citations"])
            if bundle["citations"]:
                yield f"event: citations\ndata: {json.dumps({'trace_id': trace_id, 'citations': bundle['citations']}, ensure_ascii=False)}\n\n"
            if request.show_memory_receipt:
                receipt = await run_in_threadpool(_memory_receipt, user_id)
                yield f"event: receipt\ndata: {json.dumps({'text': receipt}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
            duration_ms = int((time.perf_counter() - started) * 1000)
            chat_finished(True, duration_ms, streaming=True)
            logger.info("event=chat_completed request_id=%s provider=%s streaming=true duration_ms=%s", request_id, request.provider or DEFAULT_LLM_PROVIDER, duration_ms)
        except ServiceError as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            chat_finished(False, duration_ms, streaming=True)
            logger.warning("event=chat_failed request_id=%s code=%s streaming=true", request_id, exc.code)
            error = ApiError(code=exc.code, message=str(exc), retryable=exc.retryable)
            yield f"event: error\ndata: {error.model_dump_json()}\n\n"
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            chat_finished(False, duration_ms, streaming=True)
            logger.exception("event=chat_failed request_id=%s provider=%s streaming=true", request_id, request.provider or DEFAULT_LLM_PROVIDER)
            error = ApiError(code="generation_failed", message=str(exc), retryable=True)
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


@app.post("/api/v1/rag/feedback", response_model=RagFeedbackResponse)
def rag_feedback(request: RagFeedbackRequest, user_id: CsrfCurrentUser) -> dict[str, Any]:
    try:
        return submit_feedback(user_id, request.trace_id, request.helpful, request.comment)
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
        reconcile_memory_ownership(state)
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
def rag_status(user_id: CurrentUser) -> dict[str, Any]:
    documents = list_document_details() if _can_manage_knowledge(user_id) else []
    return {"status": knowledge_status(), "documents": documents, "release": release_gate_status()}


@app.get("/api/v1/jobs", response_model=BackgroundJobListResponse)
def background_jobs(user_id: CurrentUser, limit: int = Query(default=30, ge=1, le=100)) -> dict[str, Any]:
    return {"jobs": list_jobs(user_id, limit)}


@app.get("/api/v1/jobs/{job_id}", response_model=BackgroundJobResponse)
def background_job(job_id: str, user_id: CurrentUser) -> dict[str, Any]:
    job = get_job(job_id, user_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Background job was not found.")
    return {"job": job}


def _submit_rag_job(user_id: str, kind: str, worker, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    job = job_manager.submit(user_id, kind, worker, payload)
    logger.info("event=background_job_queued request_id=%s job_id=%s kind=%s", get_request_id(), job["id"], kind)
    return {"job": job}


@app.post("/api/v1/rag/documents", response_model=BackgroundJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_rag_document(file: Annotated[UploadFile, File(...)], user_id: CsrfCurrentUser) -> dict[str, Any]:
    """Stage a document, then copy and rebuild the index in the job queue."""
    _require_knowledge_admin(user_id)
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

    staging_directory = BASE_DIR / "data" / "job_uploads" / uuid.uuid4().hex
    temporary_path = staging_directory / Path(filename).name
    try:
        staging_directory.mkdir(parents=True, exist_ok=False)
        temporary_path.write_bytes(contents)

        def _store_and_rebuild(context) -> dict[str, Any]:
            context.progress(10, "Copying document")
            try:
                context.progress(35, "Rebuilding knowledge index")
                result = rebuild_with_release_gate(mutate=lambda: copy_document_to_store(temporary_path))
                return {**result, "uploaded": result.get("document", Path(filename).name)}
            finally:
                temporary_path.unlink(missing_ok=True)
                staging_directory.rmdir()

        return _submit_rag_job(user_id, "rag_document_upload", _store_and_rebuild, payload={"document": Path(filename).name})
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/v1/rag/rebuild", response_model=BackgroundJobResponse, status_code=status.HTTP_202_ACCEPTED)
def rebuild_rag(user_id: CsrfCurrentUser) -> dict[str, Any]:
    _require_knowledge_admin(user_id)
    def _rebuild(context) -> dict[str, Any]:
        context.progress(10, "Preparing knowledge index")
        context.progress(35, "Embedding knowledge chunks")
        return rebuild_with_release_gate()

    return _submit_rag_job(user_id, "rag_rebuild", _rebuild)


@app.delete("/api/v1/rag/documents/{name}", response_model=BackgroundJobResponse, status_code=status.HTTP_202_ACCEPTED)
def remove_rag_document(name: str, user_id: CsrfCurrentUser) -> dict[str, Any]:
    _require_knowledge_admin(user_id)
    def _delete(context) -> dict[str, Any]:
        context.progress(10, "Removing document")
        return {**rebuild_with_release_gate(mutate=lambda: delete_document_from_store(name)), "deleted": name}

    return _submit_rag_job(user_id, "rag_document_delete", _delete, payload={"document": name})


@app.get("/api/v1/rag/quality", response_model=RagQualityResponse)
def rag_quality(_: CurrentUser) -> dict[str, Any]:
    return assess_knowledge_quality()


@app.get("/api/v1/rag/feedback/summary", response_model=RagFeedbackSummaryResponse)
def rag_feedback_summary(user_id: CurrentUser) -> dict[str, Any]:
    _require_knowledge_admin(user_id)
    return feedback_summary()


@app.post("/api/v1/rag/search", response_model=RagSearchResponse)
def rag_search(request: RagSearchRequest, _: CsrfCurrentUser) -> dict[str, Any]:
    return diagnose_knowledge_search(
        request.query, top_k=request.top_k, threshold=request.threshold,
        candidate_multiplier=request.candidate_multiplier,
    )


@app.get("/api/v1/rag/evaluations/latest", response_model=RagEvaluationResponse)
def rag_latest_evaluation(user_id: CurrentUser) -> dict[str, Any]:
    _require_knowledge_admin(user_id)
    return {"report": latest_evaluation_report()}


@app.post("/api/v1/rag/evaluations/run", response_model=BackgroundJobResponse, status_code=status.HTTP_202_ACCEPTED)
def rag_run_evaluation(request: RagEvaluationRequest, user_id: CsrfCurrentUser) -> dict[str, Any]:
    _require_knowledge_admin(user_id)
    def _evaluate(context) -> dict[str, Any]:
        context.progress(10, "Loading evaluation cases")
        context.progress(30, "Running retrieval evaluation")
        return run_evaluation(
            top_k=request.top_k, threshold=request.threshold,
            candidate_multiplier=request.candidate_multiplier,
        )

    return _submit_rag_job(user_id, "rag_evaluation", _evaluate)


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
