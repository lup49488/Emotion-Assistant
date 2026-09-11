"""Cookie-session authentication route definitions."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from api_contracts import LoginRequest, LoginResponse, SessionResponse
from auth_rate_limit import clear_login_failures, login_allowed, record_login_failure
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
) -> APIRouter:
    """Create login, logout, and session routes from application callbacks."""
    router = APIRouter()

    @router.post("/api/v1/auth/login", response_model=LoginResponse)
    def login(request: LoginRequest, response: Response, raw_request: Request) -> dict[str, Any]:
        origin = raw_request.headers.get("origin", "").strip()
        if origin and not login_origin_allowed(origin, raw_request):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-site login is not permitted.")
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
