"""FastAPI adapter for the chatbot domain services.

Run locally with: ``python api_server.py``.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from chatbot import handle_user_message_stream, latest_memory_receipt, make_model_config, session_store
from gui_auth import authorize
from mood_store import add_mood_checkin
from sqlite_store import storage_backend


API_SESSION_TTL_SECONDS = int(os.getenv("API_SESSION_TTL_SECONDS", "43200"))


@dataclass
class ApiSession:
    user_id: str
    expires_at: float


_sessions: dict[str, ApiSession] = {}


class LoginRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    access_key: str = Field(min_length=1, max_length=512)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    provider: str = "local_hf"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    max_new_tokens: int = Field(default=300, ge=1, le=8_192)
    use_knowledge: bool = False
    use_style: bool = False
    show_memory_receipt: bool = True
    conversation_id: str | None = Field(default=None, max_length=64)


class MoodCheckinRequest(BaseModel):
    mood: str = Field(min_length=1, max_length=100)
    intensity: int = Field(ge=1, le=5)
    note: str = Field(default="", max_length=5_000)
    checkin_date: str | None = None


def _cors_origins() -> list[str]:
    configured = os.getenv("API_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = FastAPI(title="Emotion-Aware Chat API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


def _issue_session(user_id: str) -> tuple[str, int]:
    token = secrets.token_urlsafe(32)
    _sessions[token] = ApiSession(user_id=user_id, expires_at=time.time() + API_SESSION_TTL_SECONDS)
    return token, API_SESSION_TTL_SECONDS


def _current_user(authorization: Annotated[str | None, Header()] = None) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token is required.")
    session = _sessions.get(token)
    if session is None or session.expires_at <= time.time():
        _sessions.pop(token, None)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session is missing or expired.")
    return session.user_id


CurrentUser = Annotated[str, Depends(_current_user)]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "storage_backend": storage_backend()}


@app.post("/api/v1/auth/login")
def login(request: LoginRequest) -> dict[str, Any]:
    user_id, auth_error = authorize(request.user_id, request.access_key)
    if auth_error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=auth_error)
    token, expires_in = _issue_session(user_id)
    return {"access_token": token, "token_type": "bearer", "expires_in": expires_in, "user_id": user_id}


def _chat_chunks(user_id: str, request: ChatRequest) -> Generator[str, None, None]:
    config = make_model_config(
        provider=request.provider,
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


@app.post("/api/v1/chat")
def chat(request: ChatRequest, user_id: CurrentUser) -> dict[str, str]:
    reply = "".join(_chat_chunks(user_id, request))
    payload = {"reply": reply}
    if request.show_memory_receipt:
        with session_store.session(user_id) as state:
            payload["memory_receipt"] = latest_memory_receipt(state)
    return payload


def _memory_receipt(user_id: str) -> str:
    with session_store.session(user_id) as state:
        return latest_memory_receipt(state)


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest, user_id: CurrentUser) -> StreamingResponse:
    async def event_source():
        try:
            # The chat generator does blocking model inference; iterate it in a
            # worker thread so the async event loop stays responsive.
            async for chunk in iterate_in_threadpool(_chat_chunks(user_id, request)):
                yield f"event: chunk\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            if request.show_memory_receipt:
                receipt = await run_in_threadpool(_memory_receipt, user_id)
                yield f"event: receipt\ndata: {json.dumps({'text': receipt}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/api/v1/memory/quality")
def memory_quality(user_id: CurrentUser) -> dict[str, str]:
    # The bearer token has already authenticated the user.
    with session_store.session(user_id) as state:
        from gui_memory import build_memory_quality_report

        return {"report": build_memory_quality_report(user_id, state)}


@app.post("/api/v1/mood/checkins")
def create_mood_checkin(request: MoodCheckinRequest, user_id: CurrentUser) -> dict[str, Any]:
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
