from unittest.mock import Mock, patch

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
        patch.object(llm_providers, "require_transformers", return_value=(fake_model_cls, fake_tokenizer_cls, Mock())):
        tokenizer, model = llm_providers.get_llm()

    assert tokenizer is fake_tokenizer_cls.from_pretrained.return_value
    assert model is fake_model
    kwargs = fake_model_cls.from_pretrained.call_args.kwargs
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["torch_dtype"] is llm_providers.torch.float32
    fake_model.eval.assert_called_once()
