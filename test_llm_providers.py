import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import llm_providers
from provider_registry import provider_catalog, provider_definition


def test_resolved_base_url_adds_https_for_public_host():
    config = llm_providers.ModelRuntimeConfig(
        provider="deepseek", base_url="api.deepseek.com",
    )

    assert config.resolved_base_url() == "https://api.deepseek.com"


def test_resolved_base_url_adds_http_for_local_host():
    config = llm_providers.ModelRuntimeConfig(
        provider="openai_compatible", base_url="127.0.0.1:8001/v1",
    )

    assert config.resolved_base_url() == "http://127.0.0.1:8001/v1"


def test_resolved_base_url_rejects_unsupported_scheme():
    config = llm_providers.ModelRuntimeConfig(
        provider="deepseek", base_url="ftp://api.deepseek.com",
    )

    with pytest.raises(ValueError, match="http"):
        config.resolved_base_url()


def test_provider_registry_exposes_multiple_models_without_secrets(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-key")

    deepseek = provider_definition("deepseek")
    catalog = provider_catalog()
    deepseek_item = next(item for item in catalog["providers"] if item["id"] == "deepseek")

    assert deepseek is not None
    assert deepseek.default_model_value() == "deepseek-v4-pro"
    assert "deepseek-flash" in deepseek.model_options()
    assert "deepseek-v4-pro" in deepseek_item["models"]
    assert "secret-deepseek-key" not in json.dumps(catalog)


def test_openai_catalog_exposes_verified_gpt6_models_but_not_astra():
    catalog = provider_catalog()
    openai = next(item for item in catalog["providers"] if item["id"] == "openai")

    assert "gpt-6-luna" in openai["models"]
    assert "gpt-6-sol" in openai["models"]
    assert "gpt-6-astra" not in openai["models"]


def test_gpt6_chat_request_uses_its_supported_parameters():
    event = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"), finish_reason="stop")]
    )
    fake_client = Mock()
    fake_client.chat.completions.create.return_value = iter([event])
    fake_openai = Mock(return_value=fake_client)
    config = llm_providers.ModelRuntimeConfig(
        provider="openai",
        model="gpt-6-luna",
        api_key="test-key",
        temperature=0.4,
        top_p=0.9,
        max_new_tokens=8,
    )

    with patch.object(llm_providers, "require_openai_client", return_value=fake_openai), \
         patch.object(llm_providers, "record_usage"), \
         patch.object(llm_providers, "check_request_allowed"):
        assert list(llm_providers._stream_openai_compatible([
            {"role": "user", "content": "hello"},
        ], config)) == ["ok"]

    request = fake_client.chat.completions.create.call_args.kwargs
    assert request["max_completion_tokens"] == 8
    assert "max_tokens" not in request
    assert "temperature" not in request
    assert "top_p" not in request


def test_existing_openai_models_keep_sampling_parameters():
    definition = provider_definition("openai")
    assert definition is not None

    parameters = definition.chat_completion_parameters(
        "gpt-4o", temperature=0.4, top_p=0.9, max_new_tokens=8,
    )

    assert parameters == {"max_tokens": 8, "temperature": 0.4, "top_p": 0.9}


def test_default_provider_is_deepseek(monkeypatch):
    # resolved_model reads the environment at call time, so clear an ambient
    # per-provider value before asserting the package defaults.
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    config = llm_providers.ModelRuntimeConfig()

    assert llm_providers.DEFAULT_LLM_PROVIDER == "deepseek"
    assert config.normalized_provider() == "deepseek"
    assert config.resolved_model() == "deepseek-flash"
    assert config.resolved_base_url() == "https://api.deepseek.com"


def test_blank_deepseek_model_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "   ")
    config = llm_providers.ModelRuntimeConfig(provider="deepseek")

    assert config.resolved_base_url() == "https://api.deepseek.com"
    assert config.resolved_model() == "deepseek-flash"


def test_deepseek_uses_its_default_endpoint():
    config = llm_providers.ModelRuntimeConfig(provider="deepseek", api_key="deepseek-key")
    client = Mock()
    client.chat.completions.create.return_value = iter([])
    factory = Mock(return_value=client)

    with patch.object(llm_providers, "require_openai_client", return_value=factory), \
         patch.object(llm_providers, "record_usage"), \
         patch.object(llm_providers, "check_request_allowed"):
        list(llm_providers._stream_openai_compatible([{"role": "user", "content": "hi"}], config))

    assert factory.call_args.kwargs["base_url"] == "https://api.deepseek.com"


def test_deepseek_provider_uses_dedicated_defaults_and_key():
    config = llm_providers.ModelRuntimeConfig(provider="deepseek")

    with patch.dict("os.environ", {
        "DEEPSEEK_API_KEY": "deepseek-key",
        "DEEPSEEK_MODEL": "deepseek-v4-pro",
        "LLM_API_KEY": "fallback-key",
    }, clear=False):
        assert config.resolved_api_key() == "deepseek-key"
        assert config.resolved_model() == "deepseek-v4-pro"
        assert config.resolved_base_url() == "https://api.deepseek.com"


def test_openrouter_provider_can_be_used_as_server_fallback():
    config = llm_providers.ModelRuntimeConfig(
        provider="deepseek", model="primary-model", api_key=None, max_new_tokens=8,
    )
    calls = []

    def stream(_, candidate):
        calls.append(candidate)
        if candidate.normalized_provider() == "deepseek":
            raise llm_providers.ProviderRequestError(
                "primary timed out", kind="timeout", retryable=True,
            )
        assert candidate.resolved_api_key() == "router-key"
        yield "router fallback"

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "router-key"}, clear=False), \
        patch.object(llm_providers, "LLM_FALLBACKS_JSON", json.dumps([
            {"provider": "openrouter", "model": "openrouter/auto"},
        ])), patch.object(llm_providers, "_stream_openai_compatible", side_effect=stream):
        chunks = list(llm_providers.stream_model_response([
            {"role": "user", "content": "hello"},
        ], config))

    assert chunks == ["router fallback"]
    assert [item.normalized_provider() for item in calls] == ["deepseek", "openrouter"]


def test_get_llm_applies_local_model_loading_options():
    fake_tokenizer_cls = Mock()
    fake_tokenizer_cls.from_pretrained.return_value = Mock(name_or_path="tokenizer")

    fake_model = Mock()
    fake_model_cls = Mock()
    fake_model_cls.from_pretrained.return_value = fake_model

    with patch.object(llm_providers, "_tokenizer", None), \
        patch.object(llm_providers, "_llm_model", None), \
        patch.object(llm_providers, "LOCAL_MODEL_LOW_CPU_MEM_USAGE", True), \
        patch.object(llm_providers, "LOCAL_MODEL_ATTN_IMPLEMENTATION", "sdpa"), \
        patch.object(llm_providers, "LOCAL_MODEL_DTYPE", "float32"), \
        patch.object(llm_providers, "LOCAL_MODEL_COMPILE", False), \
        patch.object(llm_providers, "LOCAL_MODEL_CPU_THREADS", 0), \
        patch.object(llm_providers, "_ensure_local_model_memory"), \
        patch.object(llm_providers, "require_transformers", return_value=(fake_model_cls, fake_tokenizer_cls, Mock())):
        tokenizer, model = llm_providers.get_llm()

    assert tokenizer is fake_tokenizer_cls.from_pretrained.return_value
    assert model is fake_model
    kwargs = fake_model_cls.from_pretrained.call_args.kwargs
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["torch_dtype"] is llm_providers.torch.float32
    fake_model.eval.assert_called_once()


def test_memory_precheck_blocks_local_model_when_vram_is_insufficient():
    gib = 1024 ** 3
    with patch.object(llm_providers, "LOCAL_MODEL_MEMORY_CHECK", True), \
        patch.object(llm_providers, "LOCAL_MODEL_ALLOW_CPU_OFFLOAD", False), \
        patch.object(llm_providers, "LOCAL_MODEL_PARAMETER_COUNT_B", 3.0), \
        patch.object(llm_providers.torch.cuda, "is_available", return_value=True), \
        patch.object(llm_providers.torch.cuda, "mem_get_info", return_value=(4 * gib, 8 * gib), create=True), \
        patch.object(llm_providers, "_resolve_torch_dtype", return_value=llm_providers.torch.float16):
        with pytest.raises(RuntimeError, match="needs about"):
            llm_providers._ensure_local_model_memory()


def test_get_llm_configures_cpu_offload_when_enabled(tmp_path):
    fake_tokenizer_cls = Mock()
    fake_tokenizer_cls.from_pretrained.return_value = Mock(name_or_path="tokenizer")
    fake_model = Mock()
    fake_model_cls = Mock()
    fake_model_cls.from_pretrained.return_value = fake_model

    with patch.object(llm_providers, "_tokenizer", None), \
        patch.object(llm_providers, "_llm_model", None), \
        patch.object(llm_providers, "BASE_DIR", tmp_path), \
        patch.object(llm_providers, "LOCAL_MODEL_ALLOW_CPU_OFFLOAD", True), \
        patch.object(llm_providers, "LOCAL_MODEL_GPU_MAX_MEMORY_GB", 5.5), \
        patch.object(llm_providers, "LOCAL_MODEL_CPU_MAX_MEMORY_GB", 12.0), \
        patch.object(llm_providers, "LOCAL_MODEL_LOW_CPU_MEM_USAGE", True), \
        patch.object(llm_providers, "LOCAL_MODEL_ATTN_IMPLEMENTATION", None), \
        patch.object(llm_providers, "LOCAL_MODEL_DTYPE", "float16"), \
        patch.object(llm_providers, "LOCAL_MODEL_COMPILE", False), \
        patch.object(llm_providers, "LOCAL_MODEL_CPU_THREADS", 0), \
        patch.object(llm_providers, "_ensure_local_model_memory"), \
        patch.object(llm_providers, "require_transformers", return_value=(fake_model_cls, fake_tokenizer_cls, Mock())):
        llm_providers.get_llm()

    kwargs = fake_model_cls.from_pretrained.call_args.kwargs
    assert kwargs["max_memory"] == {0: "5.5GiB", "cpu": "12.0GiB"}
    assert kwargs["offload_state_dict"] is True
    assert (tmp_path / "data" / "model_offload").is_dir()


def test_openai_stream_retries_timeout_before_first_chunk():
    event = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))])
    fake_client = Mock()
    fake_client.chat.completions.create.side_effect = [TimeoutError("request timeout"), iter([event])]
    fake_openai = Mock(return_value=fake_client)
    config = llm_providers.ModelRuntimeConfig(
        provider="deepseek", model="test-model", api_key="test-key", max_new_tokens=8,
    )

    with patch.object(llm_providers, "require_openai_client", return_value=fake_openai), \
        patch.object(llm_providers, "API_MAX_RETRIES", 1), \
        patch.object(llm_providers, "API_RETRY_BACKOFF_SECONDS", 0):
        chunks = list(llm_providers._stream_openai_compatible([
            {"role": "user", "content": "hello"},
        ], config))

    assert chunks == ["hello"]
    assert fake_client.chat.completions.create.call_count == 2
    assert fake_openai.call_args.kwargs["timeout"] == llm_providers.API_REQUEST_TIMEOUT_SECONDS
    assert fake_openai.call_args.kwargs["max_retries"] == 0


def test_retryable_provider_failure_fails_over_to_server_configured_provider():
    calls = []
    config = llm_providers.ModelRuntimeConfig(
        provider="deepseek", model="primary-model", api_key=None, max_new_tokens=8,
    )

    def stream(_, candidate):
        calls.append(candidate)
        if candidate.normalized_provider() == "deepseek":
            raise llm_providers.ProviderRequestError(
                "primary timed out", kind="timeout", retryable=True,
            )
        yield "fallback reply"

    with patch.object(llm_providers, "LLM_FALLBACKS_JSON", json.dumps([
        {"provider": "openai", "model": "fallback-model"},
    ])), patch.object(llm_providers, "_stream_openai_compatible", side_effect=stream):
        chunks = list(llm_providers.stream_model_response([
            {"role": "user", "content": "hello"},
        ], config))

    assert chunks == ["fallback reply"]
    assert [item.normalized_provider() for item in calls] == ["deepseek", "openai"]
    assert calls[1].resolved_model() == "fallback-model"
    assert calls[1].max_new_tokens == config.max_new_tokens


def test_provider_failover_does_not_switch_after_streaming_has_started():
    config = llm_providers.ModelRuntimeConfig(provider="deepseek", model="primary-model", max_new_tokens=8)

    def stream(_, __):
        raise llm_providers.ProviderRequestError(
            "connection dropped", kind="network", retryable=True, emitted_content=True,
        )
        yield "unreachable"

    with patch.object(llm_providers, "LLM_FALLBACKS_JSON", '[{"provider":"openai"}]'), \
        patch.object(llm_providers, "_stream_openai_compatible", side_effect=stream) as mocked:
        with pytest.raises(llm_providers.ProviderRequestError, match="connection dropped"):
            list(llm_providers.stream_model_response([], config))

    assert mocked.call_count == 1


def test_request_scoped_api_key_disables_server_failover():
    config = llm_providers.ModelRuntimeConfig(
        provider="deepseek", model="primary-model", api_key="temporary-key", max_new_tokens=8,
    )

    def stream(_, __):
        raise llm_providers.ProviderRequestError("timed out", kind="timeout", retryable=True)
        yield "unreachable"

    with patch.object(llm_providers, "LLM_FALLBACKS_JSON", '[{"provider":"openai"}]'), \
        patch.object(llm_providers, "_stream_openai_compatible", side_effect=stream) as mocked:
        with pytest.raises(llm_providers.ProviderRequestError, match="timed out"):
            list(llm_providers.stream_model_response([], config))

    assert mocked.call_count == 1


def test_missing_provider_key_uses_stable_service_error_code():
    config = llm_providers.ModelRuntimeConfig(provider="deepseek", api_key="")
    with patch.object(llm_providers.ModelRuntimeConfig, "resolved_api_key", return_value=None), \
         pytest.raises(Exception) as captured:
        list(llm_providers._stream_openai_compatible([], config))

    assert getattr(captured.value, "code", None) == "provider_api_key_missing"
    assert not getattr(captured.value, "retryable", True)


def test_frontend_fallback_catalog_stays_in_sync_with_the_registry():
    """App.jsx carries a hardcoded catalog for when the API call fails.

    It is a deliberate second copy, so it must at least never name a provider the
    registry does not have — otherwise the settings panel offers a dead option.
    """
    import re
    from pathlib import Path

    from provider_registry import PROVIDER_CHOICES

    source = (Path(__file__).parent / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
    block = re.search(r"const FALLBACK_PROVIDER_CATALOG = \[(.*?)\n\]", source, re.S)
    assert block, "FALLBACK_PROVIDER_CATALOG was renamed or removed"
    fallback_ids = set(re.findall(r"id: '([a-z_]+)'", block.group(1)))

    assert fallback_ids <= set(PROVIDER_CHOICES), (
        f"fallback lists providers the registry does not offer: {sorted(fallback_ids - set(PROVIDER_CHOICES))}"
    )
