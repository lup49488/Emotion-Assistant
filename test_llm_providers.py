from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import llm_providers


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
