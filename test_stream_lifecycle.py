"""Streaming lifecycle regressions using isolated state and an offline provider."""
from __future__ import annotations

import anyio
import pytest
from starlette.requests import ClientDisconnect

import api_server
import chatbot
import observability
import session_store


@pytest.fixture
def isolated_chat(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setattr(chatbot, "USERS_DIR", tmp_path / "users")
    monkeypatch.setattr(chatbot, "safe_analyze", lambda *_args: ("neutral", 0.0))
    monkeypatch.setattr(chatbot, "check_crisis", lambda *_args: None)
    monkeypatch.setattr(chatbot, "smart_memory_filter", lambda *_args, **_kwargs: "discard")
    return "lifecycle-user"


class OfflineStream:
    """Keep an external reference so cleanup cannot depend on reference counting."""

    def __init__(self, *, fails=False):
        self.chunks = iter(["first", "second"])
        self.fails = fails
        self.started = False
        self.closed = False
        self.request_ids = []

    def __iter__(self):
        return self

    def __next__(self):
        self.request_ids.append(observability.get_request_id())
        if self.started and self.fails:
            raise RuntimeError("offline provider failed")
        self.started = True
        return next(self.chunks)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("outcome", ["completed", "cancelled", "failed"])
def test_chat_closes_provider_stream_for_every_exit(monkeypatch, isolated_chat, outcome):
    provider = OfflineStream(fails=outcome == "failed")
    monkeypatch.setattr(chatbot, "stream_model_response", lambda *_args: provider)
    stream = chatbot.handle_user_message_stream(
        isolated_chat, "hello", use_knowledge=False, use_style=False,
    )
    assert next(stream) == "first"

    if outcome == "cancelled":
        stream.close()
    elif outcome == "failed":
        with pytest.raises(RuntimeError, match="offline provider failed"):
            next(stream)
    else:
        assert list(stream) == ["second"]

    assert provider.closed
    lock = chatbot.session_store._get_user_lock(isolated_chat)
    assert lock.acquire(blocking=False)
    lock.release()
    assert isolated_chat not in chatbot.session_store._active_counts
    with chatbot.session_store.session(isolated_chat) as state:
        assert bool(state.history) is (outcome == "completed")


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_sse_send_disconnect_releases_real_session_and_records_failure(
    monkeypatch, isolated_chat, spec_version,
):
    provider = OfflineStream()
    monkeypatch.setattr(chatbot, "stream_model_response", lambda *_args: provider)
    finished = []
    monkeypatch.setattr(api_server, "chat_finished", lambda success, *_args, **_kwargs: finished.append(success))

    async def run():
        original_request_id = observability.get_request_id()
        disconnected = anyio.Event()
        token = observability.set_request_id("lifecycle-request")
        try:
            response = await api_server.chat_stream(
                api_server.ChatRequest(message="hello", use_knowledge=False, use_style=False),
                isolated_chat,
            )
        finally:
            observability.reset_request_id(token)

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                if spec_version == "2.4":
                    raise OSError("client disconnected")
                disconnected.set()
                await anyio.sleep_forever()

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        with anyio.fail_after(3):
            try:
                await response({"type": "http", "asgi": {"spec_version": spec_version}}, receive, send)
            except ClientDisconnect:
                assert spec_version == "2.4"

        assert observability.get_request_id() == original_request_id
        assert provider.request_ids == ["lifecycle-request"]
        assert finished == [False]
        assert provider.closed
        lock = chatbot.session_store._get_user_lock(isolated_chat)
        assert lock.acquire(blocking=False)
        lock.release()
        # A follow-up memory operation must complete after the aborted stream.
        with anyio.fail_after(3):
            await anyio.to_thread.run_sync(api_server._memory_receipt, isolated_chat)

    anyio.run(run)
