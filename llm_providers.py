import os
import threading
import logging
import time
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
    LOCAL_MODEL_ATTN_IMPLEMENTATION,
    LOCAL_MODEL_COMPILE,
    LOCAL_MODEL_CPU_THREADS,
    LOCAL_MODEL_DTYPE,
    LOCAL_MODEL_LOW_CPU_MEM_USAGE,
)

logger = logging.getLogger(__name__)

_embedding_model: Any | None = None
_tokenizer: Any | None = None
_llm_model: Any | None = None
_model_init_lock = threading.Lock()       # 保护模型懒加载本身的并发初始化
_llm_inference_lock = threading.Lock()    # 保护 model.generate() 调用的并发执行

def _resolve_torch_dtype() -> Any | None:
    dtype = LOCAL_MODEL_DTYPE
    if dtype in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return None
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    logger.warning("未知 LOCAL_MODEL_DTYPE=%r，将使用默认 dtype。", LOCAL_MODEL_DTYPE)
    return None


def _configure_torch_runtime() -> None:
    if LOCAL_MODEL_CPU_THREADS > 0:
        torch.set_num_threads(LOCAL_MODEL_CPU_THREADS)


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
                start = time.perf_counter()
                logger.info("正在加载向量模型：%s", EMBEDDING_MODEL_NAME)
                SentenceTransformer = require_sentence_transformer()
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                logger.info("向量模型加载完成，用时 %.2fs", time.perf_counter() - start)
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
                start = time.perf_counter()
                _configure_torch_runtime()
                logger.info("正在加载本地聊天模型：%s", CHAT_MODEL_NAME)
                AutoModelForCausalLM, AutoTokenizer, _ = require_transformers()
                if _tokenizer is None:
                    _tokenizer = AutoTokenizer.from_pretrained(
                        CHAT_MODEL_NAME,
                        token=HF_TOKEN,
                        use_fast=True,
                    )
                if _llm_model is None:
                    kwargs: dict[str, Any] = {
                        "token": HF_TOKEN,
                        "device_map": "auto",
                        "low_cpu_mem_usage": LOCAL_MODEL_LOW_CPU_MEM_USAGE,
                    }
                    dtype = _resolve_torch_dtype()
                    if dtype is not None:
                        kwargs["torch_dtype"] = dtype
                    if LOCAL_MODEL_ATTN_IMPLEMENTATION:
                        kwargs["attn_implementation"] = LOCAL_MODEL_ATTN_IMPLEMENTATION
                    _llm_model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL_NAME, **kwargs)
                    _llm_model.eval()
                    if LOCAL_MODEL_COMPILE and hasattr(torch, "compile"):
                        try:
                            _llm_model = torch.compile(_llm_model, mode="reduce-overhead")
                            logger.info("已启用 torch.compile 优化本地模型。")
                        except Exception:
                            logger.exception("torch.compile 失败，继续使用未编译模型。")
                logger.info("本地聊天模型加载完成，用时 %.2fs", time.perf_counter() - start)
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
    start = time.perf_counter()
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
        do_sample=config.temperature > 0,
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
        logger.info("本地模型回复完成，用时 %.2fs", time.perf_counter() - start)


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
