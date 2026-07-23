import os
import json
import threading
import logging
import time
import torch
import ctypes
import re
from urllib.parse import urlparse

from dataclasses import dataclass
from typing import Any, Generator
import numpy as np

from config import (
    BASE_DIR,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_NEW_TOKENS,
    CHAT_MODEL_NAME,
    DEFAULT_API_MODEL,
    DEFAULT_API_BASE_URL,
    LLM_FALLBACKS_JSON,
    HF_TOKEN,
    EMBEDDING_MODEL_NAME,
    LOCAL_MODEL_ATTN_IMPLEMENTATION,
    LOCAL_MODEL_COMPILE,
    LOCAL_MODEL_CPU_THREADS,
    LOCAL_MODEL_DTYPE,
    LOCAL_MODEL_LOW_CPU_MEM_USAGE,
    LOCAL_MODEL_MEMORY_CHECK,
    LOCAL_MODEL_MEMORY_SAFETY_FACTOR,
    LOCAL_MODEL_PARAMETER_COUNT_B,
    LOCAL_MODEL_ALLOW_CPU_OFFLOAD,
    LOCAL_MODEL_GPU_MAX_MEMORY_GB,
    LOCAL_MODEL_CPU_MAX_MEMORY_GB,
    API_REQUEST_TIMEOUT_SECONDS,
    API_MAX_RETRIES,
    API_RETRY_BACKOFF_SECONDS,
)
from api_usage_store import check_request_allowed, estimate_input_tokens, estimate_tokens, record_usage
from observability import get_request_id
from service_errors import ServiceError

logger = logging.getLogger(__name__)

_API_PROVIDERS = {"deepseek", "openai", "openrouter", "openai_compatible", "custom"}


class ProviderRequestError(ServiceError):
    """A provider failure with enough context to decide whether to fail over."""

    def __init__(self, message: str, *, kind: str, retryable: bool, emitted_content: bool = False) -> None:
        super().__init__(f"provider_{kind}", message, retryable=retryable)
        self.kind = kind
        self.emitted_content = emitted_content

_embedding_model: Any | None = None
_tokenizer: Any | None = None
_llm_model: Any | None = None
_model_init_lock = threading.Lock()       # 保护模型懒加载本身的并发初始化
_llm_inference_lock = threading.Lock()    # 保护 model.generate() 调用的并发执行

def _normalize_api_base_url(base_url: str | None) -> str | None:
    """Return an OpenAI client compatible HTTP(S) base URL."""
    value = (base_url or "").strip()
    if not value:
        return None

    if "://" not in value:
        local_prefixes = ("localhost", "127.0.0.1", "[::1]", "::1")
        scheme = "http" if value.lower().startswith(local_prefixes) else "https"
        value = f"{scheme}://{value}"

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API Base URL must be a valid http:// or https:// address.")
    return value.rstrip("/")


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


def _available_system_memory_bytes() -> int | None:
    """Return currently available RAM without requiring psutil."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def _model_parameter_count_billions(model_name: str) -> float:
    if LOCAL_MODEL_PARAMETER_COUNT_B > 0:
        return float(LOCAL_MODEL_PARAMETER_COUNT_B)
    match = re.search(r"(?:^|[-_/])(\d+(?:\.\d+)?)b(?:[-_/]|$)", model_name.lower())
    return float(match.group(1)) if match else 3.0


def _estimated_local_model_bytes(model_name: str, dtype: Any | None) -> int:
    bytes_per_parameter = 2 if dtype in {torch.float16, torch.bfloat16} else 4
    parameters = _model_parameter_count_billions(model_name) * 1_000_000_000
    return int(parameters * bytes_per_parameter * max(1.0, LOCAL_MODEL_MEMORY_SAFETY_FACTOR))


def _ensure_local_model_memory() -> None:
    """Reject a likely OOM before Transformers starts allocating model weights."""
    if not LOCAL_MODEL_MEMORY_CHECK:
        return
    dtype = _resolve_torch_dtype()
    required = _estimated_local_model_bytes(CHAT_MODEL_NAME, dtype)
    required_gib = required / (1024 ** 3)

    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes >= required:
            logger.info("Local model memory check passed: %.1f/%.1f GiB VRAM free.", free_bytes / (1024 ** 3), total_bytes / (1024 ** 3))
            return
        if not LOCAL_MODEL_ALLOW_CPU_OFFLOAD:
            raise RuntimeError(
                f"Local model loading cancelled: {CHAT_MODEL_NAME} needs about {required_gib:.1f} GiB free VRAM, "
                f"but only {free_bytes / (1024 ** 3):.1f} GiB is available. Close GPU-heavy apps, use a smaller model, or enable CPU offload."
            )
        available_ram = _available_system_memory_bytes()
        cpu_budget = int(LOCAL_MODEL_CPU_MAX_MEMORY_GB * (1024 ** 3))
        if available_ram is None or available_ram >= cpu_budget:
            logger.warning("VRAM is insufficient; loading with CPU offload. GPU free: %.1f GiB.", free_bytes / (1024 ** 3))
            return
        raise RuntimeError(
            f"CPU offload was cancelled: it requires at least {LOCAL_MODEL_CPU_MAX_MEMORY_GB:.1f} GiB available RAM, "
            f"but only {available_ram / (1024 ** 3):.1f} GiB is free."
        )

    available_ram = _available_system_memory_bytes()
    if available_ram is not None and available_ram < required:
        raise RuntimeError(
            f"CPU model loading cancelled: {CHAT_MODEL_NAME} needs about {required_gib:.1f} GiB free RAM, "
            f"but only {available_ram / (1024 ** 3):.1f} GiB is available."
        )


@dataclass
class ModelRuntimeConfig:
    provider: str = DEFAULT_LLM_PROVIDER
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    user_id: str | None = None

    def normalized_provider(self) -> str:
        return (self.provider or DEFAULT_LLM_PROVIDER).strip().lower()

    def resolved_model(self) -> str:
        if self.model and self.model.strip():
            return self.model.strip()
        provider = self.normalized_provider()
        if provider == "local_hf":
            return CHAT_MODEL_NAME
        if provider == "deepseek":
            return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        if provider == "openai":
            return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        if provider == "openrouter":
            return os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
        return DEFAULT_API_MODEL

    def resolved_base_url(self) -> str | None:
        if self.base_url and self.base_url.strip():
            return _normalize_api_base_url(self.base_url)
        provider = self.normalized_provider()
        if provider == "deepseek":
            return "https://api.deepseek.com"
        if provider == "openai":
            return None
        if provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        if provider in {"openai_compatible", "custom"}:
            return _normalize_api_base_url(DEFAULT_API_BASE_URL)
        return _normalize_api_base_url(self.base_url)

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
    user_id: str | None = None,
) -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        user_id=user_id,
    )


def _fallback_configs(config: ModelRuntimeConfig) -> list[ModelRuntimeConfig]:
    """Build server-managed fallback targets without exposing their credentials."""
    if config.api_key and config.api_key.strip():
        return []
    if not LLM_FALLBACKS_JSON:
        return []
    try:
        raw_targets = json.loads(LLM_FALLBACKS_JSON)
    except json.JSONDecodeError:
        logger.error("event=provider_failover_config_invalid reason=invalid_json")
        return []
    if not isinstance(raw_targets, list):
        logger.error("event=provider_failover_config_invalid reason=not_a_list")
        return []

    primary_fingerprint = (config.normalized_provider(), config.resolved_model(), config.resolved_base_url())
    candidates: list[ModelRuntimeConfig] = []
    for index, target in enumerate(raw_targets):
        if not isinstance(target, dict):
            logger.warning("event=provider_failover_target_skipped index=%s reason=not_an_object", index)
            continue
        if "api_key" in target:
            logger.warning("event=provider_failover_target_skipped index=%s reason=api_key_not_allowed", index)
            continue
        provider = str(target.get("provider", "")).strip().lower()
        if provider not in _API_PROVIDERS:
            logger.warning("event=provider_failover_target_skipped index=%s reason=unsupported_provider", index)
            continue
        model = str(target.get("model", "")).strip() or None
        base_url = str(target.get("base_url", "")).strip() or None
        candidate = make_model_config(
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
            user_id=config.user_id,
        )
        try:
            fingerprint = (candidate.normalized_provider(), candidate.resolved_model(), candidate.resolved_base_url())
        except ValueError:
            logger.warning("event=provider_failover_target_skipped index=%s reason=invalid_base_url", index)
            continue
        if fingerprint == primary_fingerprint or any(
            fingerprint == (item.normalized_provider(), item.resolved_model(), item.resolved_base_url())
            for item in candidates
        ):
            logger.warning("event=provider_failover_target_skipped index=%s reason=duplicate", index)
            continue
        candidates.append(candidate)
    return candidates


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
                _ensure_local_model_memory()
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
                    if LOCAL_MODEL_ALLOW_CPU_OFFLOAD:
                        offload_folder = BASE_DIR / "data" / "model_offload"
                        offload_folder.mkdir(parents=True, exist_ok=True)
                        kwargs["max_memory"] = {
                            0: f"{LOCAL_MODEL_GPU_MAX_MEMORY_GB:.1f}GiB",
                            "cpu": f"{LOCAL_MODEL_CPU_MAX_MEMORY_GB:.1f}GiB",
                        }
                        kwargs["offload_folder"] = str(offload_folder)
                        kwargs["offload_state_dict"] = True
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

    generation_errors: list[Exception] = []

    def run_generation() -> None:
        try:
            _run_generate_locked(model, generate_kwargs)
        except Exception as exc:
            generation_errors.append(exc)
            logger.exception("本地模型生成线程失败。")
            streamer.end()

    generate_thread = threading.Thread(
        target=run_generation,
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
    if generation_errors:
        raise RuntimeError(f"本地模型生成失败：{generation_errors[0]}") from generation_errors[0]


def _stream_openai_compatible(
    full_messages: list[dict[str, str]], config: ModelRuntimeConfig
) -> Generator[str, None, None]:
    api_key = config.resolved_api_key()
    if not api_key:
        raise ServiceError(
            "provider_api_key_missing",
            "当前 API Provider 缺少 API Key。请在 GUI 中填写 API Key，或设置服务端 API Key。",
        )

    input_tokens = estimate_input_tokens(full_messages)
    check_request_allowed(
        config.user_id,
        projected_input_tokens=input_tokens,
        projected_output_tokens=config.max_new_tokens,
    )

    OpenAI = require_openai_client()
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = config.resolved_base_url()
    if base_url:
        client_kwargs["base_url"] = base_url
    client_kwargs["timeout"] = API_REQUEST_TIMEOUT_SECONDS
    # 应用层仅在尚未输出任何 token 时重试，避免 SDK 自动重试造成重复计费。
    client_kwargs["max_retries"] = 0
    client = OpenAI(**client_kwargs)
    started = time.perf_counter()
    chunks: list[str] = []
    provider = config.normalized_provider()
    model = config.resolved_model()

    for attempt in range(API_MAX_RETRIES + 1):
        emitted_content = False
        finish_reason: str | None = None
        try:
            response = client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_new_tokens,
                stream=True,
            )
            for event in response:
                if not event.choices:
                    continue
                choice = event.choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = str(choice.finish_reason)
                delta = choice.delta
                content = getattr(delta, "content", None)
                if content:
                    emitted_content = True
                    chunks.append(content)
                    yield content
            if finish_reason == "length":
                logger.warning(
                    "event=model_output_truncated request_id=%s provider=%s model=%s max_new_tokens=%s",
                    get_request_id(), provider, model, config.max_new_tokens,
                )
            record_usage(
                config.user_id,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=estimate_tokens("".join(chunks)) if chunks else 0,
                duration_ms=int((time.perf_counter() - started) * 1000),
                success=True,
            )
            return
        except Exception as exc:
            retryable = _is_retryable_api_error(exc)
            if emitted_content or attempt >= API_MAX_RETRIES or not retryable:
                error_kind = _api_error_kind(exc)
                record_usage(
                    config.user_id,
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=estimate_tokens("".join(chunks)) if chunks else 0,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    success=False,
                    error_kind=error_kind,
                )
                raise ProviderRequestError(
                    f"API 请求失败（{error_kind}）：{exc}",
                    kind=error_kind,
                    retryable=retryable,
                    emitted_content=emitted_content,
                ) from exc
            delay = API_RETRY_BACKOFF_SECONDS * (2 ** attempt)
            logger.warning(
                "event=provider_retry request_id=%s provider=%s model=%s attempt=%s/%s delay_seconds=%.1f error=%s",
                get_request_id(), provider, model, attempt + 1, API_MAX_RETRIES + 1, delay, exc,
            )
            if delay:
                time.sleep(delay)


def _api_error_kind(exc: Exception) -> str:
    raw_status_code = getattr(exc, "status_code", None)
    try:
        status_code = int(raw_status_code) if raw_status_code is not None else None
    except (TypeError, ValueError):
        status_code = None
    text = str(exc).lower()
    if status_code in {401, 403} or "unauthorized" in text or "authentication" in text:
        return "authentication"
    if status_code == 429 or "rate limit" in text or "quota" in text:
        return "rate_limit"
    if status_code and int(status_code) >= 500:
        return "server"
    if isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text:
        return "timeout"
    if any(word in text for word in ("connection", "network", "dns", "ssl", "proxy")):
        return "network"
    return "unknown"


def _is_retryable_api_error(exc: Exception) -> bool:
    try:
        status_code = int(getattr(exc, "status_code", None))
    except (TypeError, ValueError):
        status_code = None
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    return _api_error_kind(exc) in {"timeout", "network", "server", "rate_limit"}


def stream_model_response(
    full_messages: list[dict[str, str]], config: ModelRuntimeConfig
) -> Generator[str, None, None]:
    provider = config.normalized_provider()
    if provider == "local_hf":
        try:
            yield from _stream_local_hf(full_messages, config)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("local_model_failed", f"本地模型运行失败：{exc}", retryable=True) from exc
    elif provider in _API_PROVIDERS:
        try:
            yield from _stream_openai_compatible(full_messages, config)
            return
        except ProviderRequestError as primary_error:
            if not primary_error.retryable or primary_error.emitted_content:
                raise
            fallbacks = _fallback_configs(config)
            if not fallbacks:
                raise
            primary_model = config.resolved_model()
            for fallback in fallbacks:
                logger.warning(
                    "event=provider_failover request_id=%s from_provider=%s from_model=%s to_provider=%s to_model=%s error_kind=%s",
                    get_request_id(), provider, primary_model, fallback.normalized_provider(),
                    fallback.resolved_model(), primary_error.kind,
                )
                try:
                    yield from _stream_openai_compatible(full_messages, fallback)
                    return
                except ProviderRequestError as fallback_error:
                    if not fallback_error.retryable or fallback_error.emitted_content:
                        raise
                    primary_error = fallback_error
            raise primary_error
    else:
        raise ServiceError("provider_not_supported", f"未知模型 Provider: {config.provider!r}")
    
    
def _run_generate_locked(model: Any, generate_kwargs: dict[str, Any]) -> None:
    """
    在独立线程里持有 _llm_inference_lock 并调用 model.generate()。
    锁保证多用户并发时同一时间只有一个推理在跑。
    """
    with _llm_inference_lock:
        with torch.inference_mode():
            model.generate(**generate_kwargs)
