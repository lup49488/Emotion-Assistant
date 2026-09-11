"""Conversation lifecycle route definitions."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from api_contracts import (
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationRenameRequest,
    ConversationResponse,
)
from conversation_store import create_conversation, delete_conversation, get_conversation, list_conversations, rename_conversation


def create_conversations_router(
    *,
    current_user: Callable[..., str],
    csrf_protected_user: Callable[..., str],
) -> APIRouter:
    """Create conversation routes with injected application authentication."""
    router = APIRouter()

    @router.get("/api/v1/conversations", response_model=ConversationListResponse)
    def conversations(user_id: str = Depends(current_user)) -> dict[str, Any]:
        return {"conversations": list_conversations(user_id)}

    @router.post("/api/v1/conversations", response_model=ConversationResponse)
    def new_conversation(
        request: ConversationCreateRequest, user_id: str = Depends(csrf_protected_user),
    ) -> dict[str, Any]:
        return {"conversation": create_conversation(user_id, request.title)}

    @router.get("/api/v1/conversations/{conversation_id}", response_model=ConversationResponse)
    def conversation(conversation_id: str, user_id: str = Depends(current_user)) -> dict[str, Any]:
        result = get_conversation(user_id, conversation_id)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation was not found.")
        return {"conversation": result}

    @router.put("/api/v1/conversations/{conversation_id}", response_model=ConversationResponse)
    def rename_saved_conversation(
        conversation_id: str,
        request: ConversationRenameRequest,
        user_id: str = Depends(csrf_protected_user),
    ) -> dict[str, Any]:
        result = rename_conversation(user_id, conversation_id, request.title)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation was not found.")
        return {"conversation": result}

    @router.delete("/api/v1/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
    def remove_conversation(conversation_id: str, user_id: str = Depends(csrf_protected_user)) -> None:
        if not delete_conversation(user_id, conversation_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation was not found.")

    return router
