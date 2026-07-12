import os
import threading
import torch

from dataclasses import dataclass
from typing import Any, Generator
import numpy as np

from config import (
    DEFAULT_LLM_PROVIDER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_NEW_TOKENS,
    CHAT_MODEL_NAME,
    DEFAULT_API_MODEL,
    DEFAULT_API_BASE_URL,
    HF_TOKEN,
    EMBEDDING_MODEL_NAME,
)


_embedding_model: Any | None = None
_tokenizer: Any | None = None
_llm_model: Any | None = None
_model_init_lock = threading.Lock()       # 保护模型懒加载本身的并发初始化
_llm_inference_lock = threading.Lock()    # 保护 model.generate() 调用的并发执行

@dataclass
class ModelRuntimeConfig:
    provider: str = DEFAULT_LLM_PROVIDER
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS

    def normalized_provider(self) -> str:
        return (self.provider or "local_hf").strip().lower()

    def resolved_model(self) -> str:
        if self.model and self.model.strip():
            return self.model.strip()
        if self.normalized_provider() == "local_hf":
            return CHAT_MODEL_NAME
        return DEFAULT_API_MODEL

    def resolved_base_url(self) -> str | None:
        if self.base_url and self.base_url.strip():
            return self.base_url.strip()
        provider = self.normalized_provider()
        if provider == "deepseek":
            return "https://api.deepseek.com"
        if provider == "openai":
            return None
        if provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        if provider in {"openai_compatible", "custom"}:
            return DEFAULT_API_BASE_URL
        return self.base_url

    def resolved_api_key(self) -> str | None:
        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        provider = self.normalized_provider()
        if provider == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
        if provider == "openai":
            return os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if provider == "openrouter":
            return os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
        return os.getenv("LLM_API_KEY")


def make_model_config(
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
    )


def require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 transformers，请先安装后再使用大模型回复功能。") from exc
    return AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def require_openai_client():
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 openai，请先安装后再使用 API 模型功能。") from exc
    return OpenAI


def require_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 sentence-transformers，请先安装后再使用向量记忆功能。") from exc
    return SentenceTransformer


def get_embedding_model() -> Any:
    global _embedding_model
    if _embedding_model is None:
        with _model_init_lock:
            if _embedding_model is None:
                SentenceTransformer = require_sentence_transformer()
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_embedding_dimension() -> int:
    model = get_embedding_model()
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def encode_texts(texts: list[str]) -> np.ndarray:
    embeddings = get_embedding_model().encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)

def get_llm() -> tuple[Any, Any]:
    global _tokenizer, _llm_model
    if _tokenizer is None or _llm_model is None:
        with _model_init_lock:
            if _tokenizer is None or _llm_model is None:
                AutoModelForCausalLM, AutoTokenizer, _ = require_transformers()
                if _tokenizer is None:
                    _tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME, token=HF_TOKEN)
                if _llm_model is None:
                    kwargs: dict[str, Any] = {"token": HF_TOKEN, "device_map": "auto"}
                    if torch.cuda.is_available():
                        kwargs["torch_dtype"] = torch.bfloat16
                    _llm_model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL_NAME, **kwargs)
    return _tokenizer, _llm_model

def get_text_iterator_streamer() -> type:
    """
    懒加载 TextIteratorStreamer，独立封装方便测试 mock。
    调用方：chat() 函数。
    """
    _, _, TextIteratorStreamer = require_transformers()
    return TextIteratorStreamer

def _stream_local_hf(
    full_messages: list[dict[str, str]], config: ModelRuntimeConfig
) -> Generator[str, None, None]:
    tokenizer, model = get_llm()
    prompt  = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

    TextIteratorStreamer = get_text_iterator_streamer()
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generate_kwargs = dict(
        **encoded,
        streamer=streamer,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=True,
        use_cache=True,
    )

    generate_thread = threading.Thread(
        target=_run_generate_locked,
        args=(model, generate_kwargs),
        daemon=True,
    )
    generate_thread.start()

    try:
        for chunk in streamer:
            if chunk:
                yield chunk
    finally:
        generate_thread.join(timeout=60)


def _stream_openai_compatible(
    full_messages: list[dict[str, str]], config: ModelRuntimeConfig
) -> Generator[str, None, None]:
    api_key = config.resolved_api_key()
    if not api_key:
        raise RuntimeError(
            "当前 API Provider 缺少 API Key。请在 GUI 中填写 API Key，"
            "或设置 DEEPSEEK_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY / LLM_API_KEY。"
        )

    OpenAI = require_openai_client()
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = config.resolved_base_url()
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    response = client.chat.completions.create(
        model=config.resolved_model(),
        messages=full_messages,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
        stream=True,
    )
    for event in response:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content


def stream_model_response(
    full_messages: list[dict[str, str]], config: ModelRuntimeConfig
) -> Generator[str, None, None]:
    provider = config.normalized_provider()
    if provider == "local_hf":
        yield from _stream_local_hf(full_messages, config)
    elif provider in {"deepseek", "openai", "openrouter", "openai_compatible", "custom"}:
        yield from _stream_openai_compatible(full_messages, config)
    else:
        raise ValueError(f"未知模型 Provider: {config.provider!r}")
    
    
def _run_generate_locked(model: Any, generate_kwargs: dict[str, Any]) -> None:
    """
    在独立线程里持有 _llm_inference_lock 并调用 model.generate()。
    锁保证多用户并发时同一时间只有一个推理在跑。
    """
    with _llm_inference_lock:
        with torch.inference_mode():
            model.generate(**generate_kwargs)
