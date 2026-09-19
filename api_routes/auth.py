"""Cookie-session authentication route definitions."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from api_contracts import (
    EmailAuthConfigResponse,
    EmailChallengeStartRequest,
    EmailChallengeStartResponse,
    EmailChallengeVerifyRequest,
    EmailChallengeVerifyResponse,
    EmailPasswordLoginRequest,
    EmailPasswordRegisterRequest,
    LegacyMigrationRequest,
    LoginRequest,
    LoginResponse,
    SessionResponse,
)
from auth_rate_limit import clear_login_failures, login_allowed, record_login_failure
from email_auth_store import EmailAuthError, EmailAuthService, EmailAuthUnavailable
from gui_auth import authorize


def create_auth_router(
    *,
    current_user: Callable[..., str],
    csrf_protected_user: Callable[..., str],
    login_origin_allowed: Callable[[str, Request], bool],
    client_ip: Callable[[Request], str],
    set_session_cookies: Callable[[Response, str], int],
    cookie_settings: Callable[[], dict[str, Any]],
    operations_users: Callable[[], set[str]],
    can_manage_knowledge: Callable[[str], bool],
    email_auth: EmailAuthService,
    request_id: Callable[[Request], str],
) -> APIRouter:
    """Create login, logout, and session routes from application callbacks."""
    router = APIRouter()

    def _require_allowed_origin(raw_request: Request) -> None:
        origin = raw_request.headers.get("origin", "").strip()
        if origin and not login_origin_allowed(origin, raw_request):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-site login is not permitted.")

    def _email_error(error: EmailAuthError) -> HTTPException:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE if isinstance(error, EmailAuthUnavailable) else status.HTTP_400_BAD_REQUEST
        return HTTPException(status_code=status_code, detail=str(error))

    @router.get("/api/v1/auth/config", response_model=EmailAuthConfigResponse)
    def auth_config() -> dict[str, Any]:
        return email_auth.public_config()

    @router.post("/api/v1/auth/email/start", status_code=status.HTTP_202_ACCEPTED, response_model=EmailChallengeStartResponse)
    def start_email_challenge(request: EmailChallengeStartRequest, raw_request: Request) -> dict[str, str]:
        _require_allowed_origin(raw_request)
        try:
            challenge_id = email_auth.start_challenge(
                request.email,
                request.purpose,
                request.turnstile_token,
                client_ip(raw_request),
                request_id(raw_request),
            )
        except EmailAuthError as exc:
            raise _email_error(exc) from None
        return {"status": "verification_started", "challenge_id": challenge_id}

    @router.post("/api/v1/auth/email/verify", response_model=EmailChallengeVerifyResponse)
    def verify_email_challenge(request: EmailChallengeVerifyRequest, raw_request: Request) -> dict[str, str]:
        _require_allowed_origin(raw_request)
        try:
            verified_intent = email_auth.verify_challenge(request.challenge_id, request.code, request.purpose, request_id(raw_request))
        except EmailAuthError as exc:
            raise _email_error(exc) from None
        return {"verified_intent": verified_intent}

    @router.post("/api/v1/auth/register", response_model=LoginResponse)
    def register(request: EmailPasswordRegisterRequest, response: Response, raw_request: Request) -> dict[str, Any]:
        _require_allowed_origin(raw_request)
        try:
            user_id = email_auth.register(request.verified_intent, request.password, request_id(raw_request))
        except EmailAuthError as exc:
            raise _email_error(exc) from None
        expires_in = set_session_cookies(response, user_id)
        return {"token_type": "cookie", "expires_in": expires_in, "user_id": user_id}

    @router.post("/api/v1/auth/migrate-legacy", response_model=LoginResponse)
    def migrate_legacy(request: LegacyMigrationRequest, response: Response, raw_request: Request) -> dict[str, Any]:
        _require_allowed_origin(raw_request)
        allowed, retry_after = login_allowed(client_ip(raw_request), request.legacy_user_id)
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=f"Too many failed login attempts. Try again in {retry_after} seconds.", headers={"Retry-After": str(retry_after)})
        try:
            user_id = email_auth.migrate_legacy(request.verified_intent, request.legacy_user_id, request.legacy_access_key, request.password, request_id(raw_request))
        except EmailAuthError as exc:
            record_login_failure(client_ip(raw_request), request.legacy_user_id)
            raise _email_error(exc) from None
        clear_login_failures(client_ip(raw_request), request.legacy_user_id)
        expires_in = set_session_cookies(response, user_id)
        return {"token_type": "cookie", "expires_in": expires_in, "user_id": user_id}

    @router.post("/api/v1/auth/password/login", response_model=LoginResponse)
    def password_login(request: EmailPasswordLoginRequest, response: Response, raw_request: Request) -> dict[str, Any]:
        _require_allowed_origin(raw_request)
        address = client_ip(raw_request)
        allowed, retry_after = login_allowed(address, request.email)
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=f"Too many failed login attempts. Try again in {retry_after} seconds.", headers={"Retry-After": str(retry_after)})
        try:
            user_id = email_auth.login_password(request.email, request.password, request_id(raw_request))
        except EmailAuthError as exc:
            record_login_failure(address, request.email)
            raise _email_error(exc) from None
        clear_login_failures(address, request.email)
        expires_in = set_session_cookies(response, user_id)
        return {"token_type": "cookie", "expires_in": expires_in, "user_id": user_id}

    @router.post("/api/v1/auth/login", response_model=LoginResponse)
    def login(request: LoginRequest, response: Response, raw_request: Request) -> dict[str, Any]:
        if not email_auth.legacy_login_enabled():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Legacy sign-in is no longer available.")
        _require_allowed_origin(raw_request)
        address = client_ip(raw_request)
        allowed, retry_after = login_allowed(address, request.user_id)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many failed login attempts. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        user_id, auth_error = authorize(request.user_id, request.access_key)
        if auth_error:
            record_login_failure(address, request.user_id)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=auth_error)
        clear_login_failures(address, request.user_id)
        expires_in = set_session_cookies(response, user_id)
        return {"token_type": "cookie", "expires_in": expires_in, "user_id": user_id}

    @router.post("/api/v1/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(response: Response, _: str = Depends(csrf_protected_user)) -> None:
        settings = cookie_settings()
        response.delete_cookie(settings["session_name"], httponly=True, **settings["cookie_options"])
        response.delete_cookie(settings["csrf_name"], **settings["cookie_options"])

    @router.get("/api/v1/auth/session", response_model=SessionResponse)
    def session_info(user_id: str = Depends(current_user)) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "authentication": "signed_cookie",
            "can_access_operations": user_id in operations_users(),
            "can_manage_knowledge": can_manage_knowledge(user_id),
        }

    return router
