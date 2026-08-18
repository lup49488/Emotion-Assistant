"""Regression tests for the native Anthropic (Messages API) provider."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import config as config_module
import llm_providers


class _FakeAnthropicStream:
    """Mimic the SDK's streaming context manager: text deltas, then a final message."""

    def __init__(self, texts, final):
        self._texts = texts
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    @property
    def text_stream(self):
        return iter(self._texts)

    def get_final_message(self):
        return self._final


def _fake_client(texts, *, stop_reason="end_turn", input_tokens=11, output_tokens=7):
    final = SimpleNamespace(
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )
    client = Mock()
    client.messages.stream.return_value = _FakeAnthropicStream(texts, final)
    return client


def _patched(client, **overrides):
    """Patch the module-level seams every streaming test needs."""
    patches = [
        patch.object(llm_providers, "require_anthropic_client", return_value=Mock(return_value=client)),
        patch.object(llm_providers, "record_usage"),
        patch.object(llm_providers, "check_request_allowed"),
    ]
    patches.extend(patch.object(llm_providers, name, value) for name, value in overrides.items())
    return patches


def _run(client, messages, config, **overrides):
    patches = _patched(client, **overrides)
    for item in patches:
        item.start()
    try:
        return list(llm_providers._stream_anthropic(messages, config))
    finally:
        for item in reversed(patches):
            item.stop()


def test_stream_sends_the_messages_api_shape():
    client = _fake_client(["你好", "，在的"])
    config = llm_providers.ModelRuntimeConfig(
        provider="anthropic", model="claude-opus-5", api_key="test-key", max_new_tokens=256,
    )

    chunks = _run(client, [
        {"role": "system", "content": "你是一个温柔的助手。"},
        {"role": "user", "content": "我今天有点难过"},
    ], config)

    request = client.messages.stream.call_args.kwargs
    assert chunks == ["你好", "，在的"]
    # The system prompt is a top-level field, not a message role.
    assert request["system"] == "你是一个温柔的助手。"
    assert request["messages"] == [{"role": "user", "content": "我今天有点难过"}]
    assert request["max_tokens"] == 256
    assert request["model"] == "claude-opus-5"


def test_request_omits_sampling_parameters():
    """Current Claude models reject temperature/top_p/top_k with a 400."""
    client = _fake_client(["ok"])
    config = llm_providers.ModelRuntimeConfig(
        provider="anthropic", api_key="test-key", temperature=0.8, top_p=0.9,
    )

    _run(client, [{"role": "user", "content": "hi"}], config)

    request = client.messages.stream.call_args.kwargs
    assert "temperature" not in request
    assert "top_p" not in request
    assert "top_k" not in request


def test_client_disables_sdk_retries_so_billing_is_not_doubled():
    client = _fake_client(["ok"])
    factory = Mock(return_value=client)
    config = llm_providers.ModelRuntimeConfig(provider="anthropic", api_key="test-key")

    with patch.object(llm_providers, "require_anthropic_client", return_value=factory), \
         patch.object(llm_providers, "record_usage"), \
         patch.object(llm_providers, "check_request_allowed"):
        list(llm_providers._stream_anthropic([{"role": "user", "content": "hi"}], config))

    assert factory.call_args.kwargs["max_retries"] == 0
    assert factory.call_args.kwargs["timeout"] == llm_providers.API_REQUEST_TIMEOUT_SECONDS


def test_disabled_thinking_caps_effort_at_high():
    """Anthropic rejects disabled thinking above `high`, so the pair is reconciled first."""
    client = _fake_client(["ok"])
    config = llm_providers.ModelRuntimeConfig(provider="anthropic", api_key="test-key")

    _run(client, [{"role": "user", "content": "hi"}], config,
         ANTHROPIC_THINKING="off", ANTHROPIC_EFFORT="max")

    request = client.messages.stream.call_args.kwargs
    assert request["thinking"] == {"type": "disabled"}
    assert request["output_config"] == {"effort": "high"}


def test_adaptive_thinking_keeps_the_configured_effort():
    client = _fake_client(["ok"])
    config = llm_providers.ModelRuntimeConfig(provider="anthropic", api_key="test-key")

    _run(client, [{"role": "user", "content": "hi"}], config,
         ANTHROPIC_THINKING="adaptive", ANTHROPIC_EFFORT="xhigh")

    request = client.messages.stream.call_args.kwargs
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "xhigh"}


def test_refusal_is_reported_as_a_non_retryable_error():
    client = _fake_client([], stop_reason="refusal", output_tokens=0)
    config = llm_providers.ModelRuntimeConfig(provider="anthropic", api_key="test-key")

    with pytest.raises(Exception) as captured:
        _run(client, [{"role": "user", "content": "hi"}], config)

    assert getattr(captured.value, "code", None) == "provider_refusal"
    assert not getattr(captured.value, "retryable", True)


def test_usage_prefers_the_api_token_counts_over_local_estimates():
    client = _fake_client(["ok"], input_tokens=123, output_tokens=45)
    config = llm_providers.ModelRuntimeConfig(provider="anthropic", api_key="test-key")

    with patch.object(llm_providers, "require_anthropic_client", return_value=Mock(return_value=client)), \
         patch.object(llm_providers, "record_usage") as recorded, \
         patch.object(llm_providers, "check_request_allowed"):
        list(llm_providers._stream_anthropic([{"role": "user", "content": "hi"}], config))

    assert recorded.call_args.kwargs["input_tokens"] == 123
    assert recorded.call_args.kwargs["output_tokens"] == 45


def test_stream_retries_a_timeout_that_happened_before_the_first_chunk():
    client = Mock()
    client.messages.stream.side_effect = [
        TimeoutError("request timeout"),
        _FakeAnthropicStream(["hello"], SimpleNamespace(stop_reason="end_turn", usage=None)),
    ]
    config = llm_providers.ModelRuntimeConfig(provider="anthropic", api_key="test-key")

    chunks = _run(client, [{"role": "user", "content": "hi"}], config,
                  API_MAX_RETRIES=1, API_RETRY_BACKOFF_SECONDS=0)

    assert chunks == ["hello"]
    assert client.messages.stream.call_count == 2


def test_split_merges_system_turns_and_drops_a_leading_assistant_turn():
    system, turns = llm_providers.split_anthropic_messages([
        {"role": "system", "content": "第一段"},
        {"role": "system", "content": "第二段"},
        # A trimmed short-term history can start on an assistant turn, which the
        # Messages API rejects.
        {"role": "assistant", "content": "被截断的历史开头"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
    ])

    assert system == "第一段\n\n第二段"
    assert turns == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
    ]


def test_request_without_a_user_turn_is_rejected_before_calling_the_api():
    config = llm_providers.ModelRuntimeConfig(provider="anthropic", api_key="test-key")
    client = _fake_client(["never"])

    with pytest.raises(Exception) as captured:
        _run(client, [{"role": "system", "content": "只有系统提示"}], config)

    assert getattr(captured.value, "code", None) == "provider_empty_request"
    assert client.messages.stream.call_count == 0


def test_dispatch_routes_anthropic_to_the_messages_api_path():
    config = llm_providers.ModelRuntimeConfig(provider="anthropic", api_key="test-key")

    with patch.object(llm_providers, "_stream_anthropic", return_value=iter(["ok"])) as anthropic_stream, \
         patch.object(llm_providers, "_stream_openai_compatible") as openai_stream:
        chunks = list(llm_providers.stream_model_response([{"role": "user", "content": "hi"}], config))

    assert chunks == ["ok"]
    assert anthropic_stream.call_count == 1
    assert openai_stream.call_count == 0


def test_dispatch_still_routes_other_providers_to_the_openai_path():
    config = llm_providers.ModelRuntimeConfig(provider="deepseek", api_key="test-key")

    with patch.object(llm_providers, "_stream_anthropic") as anthropic_stream, \
         patch.object(llm_providers, "_stream_openai_compatible", return_value=iter(["ok"])) as openai_stream:
        chunks = list(llm_providers.stream_model_response([{"role": "user", "content": "hi"}], config))

    assert chunks == ["ok"]
    assert anthropic_stream.call_count == 0
    assert openai_stream.call_count == 1


def test_missing_anthropic_key_uses_the_shared_service_error_code():
    config = llm_providers.ModelRuntimeConfig(provider="anthropic", api_key="")

    with patch.object(llm_providers.ModelRuntimeConfig, "resolved_api_key", return_value=None), \
         pytest.raises(Exception) as captured:
        list(llm_providers._stream_anthropic([{"role": "user", "content": "hi"}], config))

    assert getattr(captured.value, "code", None) == "provider_api_key_missing"


def test_resolved_defaults_come_from_the_anthropic_settings(monkeypatch):
    # Driven through the real environment rather than a module attribute: these
    # settings moved into provider_registry once, and a patched attribute would
    # have kept passing while the shipped path changed underneath it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "")
    config = llm_providers.ModelRuntimeConfig(provider="anthropic")

    assert config.resolved_model() == config_module.ANTHROPIC_MODEL
    assert config.resolved_api_key() == "env-key"
    # A blank override must never leave an empty endpoint behind. Whether it
    # resolves to None or straight to the official host depends on how the
    # process started, and _stream_anthropic normalises both to the same value.
    assert config.resolved_base_url() in (None, llm_providers.ANTHROPIC_DEFAULT_BASE_URL)


def test_an_explicit_base_url_override_is_normalized(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "gateway.internal/v1")
    config = llm_providers.ModelRuntimeConfig(provider="anthropic")

    assert config.resolved_base_url() == "https://gateway.internal/v1"


def test_sdk_exception_types_are_classified_before_message_text():
    """Message-text heuristics mislabel reworded or localized errors."""
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    cases = {
        anthropic.APIConnectionError(request=request): "network",
        anthropic.APITimeoutError(request=request): "timeout",
        anthropic.RateLimitError("", response=httpx.Response(429, request=request), body=None): "rate_limit",
        anthropic.AuthenticationError("", response=httpx.Response(401, request=request), body=None): "authentication",
        anthropic.InternalServerError("", response=httpx.Response(503, request=request), body=None): "server",
    }

    assert {llm_providers._api_error_kind(exc) for exc in cases} == set(cases.values())
    for exc, kind in cases.items():
        assert llm_providers._api_error_kind(exc) == kind
    # Only the credential failure is worth failing over on.
    assert not llm_providers._is_retryable_api_error(
        anthropic.AuthenticationError("", response=httpx.Response(401, request=request), body=None)
    )


def test_failure_log_records_the_endpoint_that_was_actually_used(caplog, monkeypatch):
    """An OS env var can override .env, so the log must show where it connected."""
    client = Mock()
    client.messages.stream.side_effect = TimeoutError("request timeout")
    config = llm_providers.ModelRuntimeConfig(provider="anthropic", api_key="test-key")

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.internal")
    with caplog.at_level("WARNING"), pytest.raises(Exception):
        _run(client, [{"role": "user", "content": "hi"}], config, API_MAX_RETRIES=0)

    assert "endpoint=https://gateway.internal" in caplog.text
    assert "error_type=TimeoutError" in caplog.text
    assert "test-key" not in caplog.text


def test_endpoint_is_always_passed_so_an_empty_env_var_cannot_poison_the_sdk(monkeypatch):
    """`ANTHROPIC_BASE_URL=` in a .env injects "", which the SDK treats as a real
    base URL and turns into a scheme-less request — an opaque "Connection error".
    """
    client = _fake_client(["ok"])
    factory = Mock(return_value=client)
    config = llm_providers.ModelRuntimeConfig(provider="anthropic", api_key="test-key")

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "")
    with patch.object(llm_providers, "require_anthropic_client", return_value=factory), \
         patch.object(llm_providers, "record_usage"), \
         patch.object(llm_providers, "check_request_allowed"):
        list(llm_providers._stream_anthropic([{"role": "user", "content": "hi"}], config))

    assert factory.call_args.kwargs["base_url"] == llm_providers.ANTHROPIC_DEFAULT_BASE_URL


def test_an_explicit_endpoint_still_wins_over_the_default(monkeypatch):
    client = _fake_client(["ok"])
    factory = Mock(return_value=client)
    config = llm_providers.ModelRuntimeConfig(provider="anthropic", api_key="test-key")

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.internal")
    with patch.object(llm_providers, "require_anthropic_client", return_value=factory), \
         patch.object(llm_providers, "record_usage"), \
         patch.object(llm_providers, "check_request_allowed"):
        list(llm_providers._stream_anthropic([{"role": "user", "content": "hi"}], config))

    assert factory.call_args.kwargs["base_url"] == "https://gateway.internal"
