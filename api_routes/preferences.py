"""User preference route definitions."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends

from api_contracts import (
    MemorySavePreferenceRequest,
    MemorySavePreferenceResponse,
    ReplyBasisPreferenceRequest,
    ReplyBasisPreferenceResponse,
    ReplyBasisTurnCorrectionRequest,
    ReplyBasisTurnCorrectionResponse,
    StylePreferenceRequest,
    StylePreferenceResponse,
)
from memory_preference_store import get_memory_save_mode, set_memory_save_mode
from reply_basis_store import get_reply_basis_preference, set_reply_basis_preference
from style_preference_store import get_style_prefix, set_style_prefix
from style_store import style_prefixes


def create_preferences_router(
    *,
    current_user: Callable[..., str],
    csrf_protected_user: Callable[..., str],
    update_session_memory_save_mode: Callable[[str, str], None],
    correct_turn_memory: Callable[[str, str, str], str],
) -> APIRouter:
    """Create preference routes while leaving session state with the app."""
    router = APIRouter()

    @router.get("/api/v1/style/preference", response_model=StylePreferenceResponse)
    def read_style_preference(user_id: str = Depends(current_user)) -> dict[str, Any]:
        return {"style_prefix": get_style_prefix(user_id), "available": style_prefixes()}

    @router.put("/api/v1/style/preference", response_model=StylePreferenceResponse)
    def update_style_preference(
        request: StylePreferenceRequest, user_id: str = Depends(csrf_protected_user),
    ) -> dict[str, Any]:
        return {"style_prefix": set_style_prefix(user_id, request.style_prefix), "available": style_prefixes()}

    @router.get("/api/v1/memory/preference", response_model=MemorySavePreferenceResponse)
    def read_memory_save_preference(user_id: str = Depends(current_user)) -> dict[str, str]:
        return {"mode": get_memory_save_mode(user_id)}

    @router.put("/api/v1/memory/preference", response_model=MemorySavePreferenceResponse)
    def update_memory_save_preference(
        request: MemorySavePreferenceRequest, user_id: str = Depends(csrf_protected_user),
    ) -> dict[str, str]:
        mode = set_memory_save_mode(user_id, request.mode)
        update_session_memory_save_mode(user_id, mode)
        return {"mode": mode}

    @router.get("/api/v1/reply-basis/preference", response_model=ReplyBasisPreferenceResponse)
    def read_reply_basis_preference(user_id: str = Depends(current_user)) -> dict[str, bool | str | None]:
        return get_reply_basis_preference(user_id)

    @router.put("/api/v1/reply-basis/preference", response_model=ReplyBasisPreferenceResponse)
    def update_reply_basis_preference(
        request: ReplyBasisPreferenceRequest, user_id: str = Depends(csrf_protected_user),
    ) -> dict[str, bool | str | None]:
        return set_reply_basis_preference(user_id, request.enabled, request.correction)

    @router.post("/api/v1/reply-basis/corrections", response_model=ReplyBasisTurnCorrectionResponse)
    def correct_reply_basis_turn(
        request: ReplyBasisTurnCorrectionRequest, user_id: str = Depends(csrf_protected_user),
    ) -> dict[str, Any]:
        memory_action = correct_turn_memory(user_id, request.turn_id, request.action)
        preference = get_reply_basis_preference(user_id)
        if request.action in {"frustrated", "not_this"}:
            preference = set_reply_basis_preference(user_id, True, "steady")
        return {"memory_action": memory_action, "preference": preference}

    return router
