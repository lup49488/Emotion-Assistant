from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import llm_providers


class Stream:
    def __init__(self, events, *, error=None):
        self.events = events
        self.error = error
        self.closed = False

    def __iter__(self):
        yield from self.events
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


def event(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=None)])


@pytest.mark.parametrize("outcome", ["success", "cancel", "failure"])
def test_provider_closes_stream_and_client_on_every_exit(monkeypatch, outcome):
    response = Stream([event("first"), event("second")], error=TimeoutError() if outcome == "failure" else None)
    client = Mock()
    client.chat.completions.create.return_value = response
    monkeypatch.setattr(llm_providers, "require_openai_client", lambda: Mock(return_value=client))
    monkeypatch.setattr(llm_providers, "check_request_allowed", Mock())
    monkeypatch.setattr(llm_providers, "record_usage", Mock())
    config = llm_providers.ModelRuntimeConfig(provider="deepseek", api_key="synthetic-key")
    stream = llm_providers._stream_openai_compatible([], config)
    assert next(stream) == "first"
    if outcome == "cancel":
        stream.close()
    elif outcome == "failure":
        with pytest.raises(llm_providers.ProviderRequestError):
            list(stream)
    else:
        assert list(stream) == ["second"]
    assert response.closed
    client.close.assert_called_once()


def test_failed_attempt_closes_before_retrying(monkeypatch):
    first = Stream([], error=TimeoutError())
    second = Stream([event("recovered")])
    client = Mock()
    def create(**_kwargs):
        if client.chat.completions.create.call_count == 1:
            return first
        assert first.closed, "the failed upstream response remained open during retry"
        return second
    client.chat.completions.create.side_effect = create
    monkeypatch.setattr(llm_providers, "require_openai_client", lambda: Mock(return_value=client))
    monkeypatch.setattr(llm_providers, "check_request_allowed", Mock())
    monkeypatch.setattr(llm_providers, "record_usage", Mock())
    monkeypatch.setattr(llm_providers, "API_MAX_RETRIES", 1)
    monkeypatch.setattr(llm_providers, "API_RETRY_BACKOFF_SECONDS", 0)
    config = llm_providers.ModelRuntimeConfig(provider="deepseek", api_key="synthetic-key")
    assert list(llm_providers._stream_openai_compatible([], config)) == ["recovered"]
    assert second.closed
    client.close.assert_called_once()


class AnthropicStream:
    """Minimal stand-in for the Messages API streaming context manager."""

    def __init__(self, texts):
        self.texts = texts
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.exited = True
        return False

    @property
    def text_stream(self):
        yield from self.texts

    def get_final_message(self):
        return SimpleNamespace(stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1, output_tokens=1))


@pytest.mark.parametrize("outcome", ["success", "cancel"])
def test_anthropic_closes_stream_and_client_on_every_exit(monkeypatch, outcome):
    response = AnthropicStream(["first", "second"])
    client = Mock()
    client.messages.stream.return_value = response
    monkeypatch.setattr(llm_providers, "require_anthropic_client", lambda: Mock(return_value=client))
    monkeypatch.setattr(llm_providers, "check_request_allowed", Mock())
    monkeypatch.setattr(llm_providers, "record_usage", Mock())
    config = llm_providers.ModelRuntimeConfig(
        provider="anthropic", model="claude-opus-5", api_key="synthetic-key",
    )
    stream = llm_providers._stream_anthropic([{"role": "user", "content": "hi"}], config)
    assert next(stream) == "first"
    if outcome == "cancel":
        stream.close()
    else:
        assert list(stream) == ["second"]
    assert response.exited
    client.close.assert_called_once()
