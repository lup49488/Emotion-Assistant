from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)

LOCAL_DTYPE_CHOICES = ["auto", "bfloat16", "float16", "float32"]


def env_quote(value: str) -> str:
    if value == "":
        return ""
    if any(char.isspace() or char in '#"\\' for char in value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def read_env_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def upsert_env_values(path: Path, updates: dict[str, str]) -> None:
    lines = read_env_lines(path)
    remaining = dict(updates)
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        prefix = ""
        candidate = stripped
        if candidate.startswith("export "):
            prefix = "export "
            candidate = candidate[7:].lstrip()
        if not candidate or candidate.startswith("#") or "=" not in candidate:
            new_lines.append(line)
            continue

        key = candidate.split("=", 1)[0].strip()
        if key in remaining:
            new_lines.append(f"{prefix}{key}={env_quote(remaining.pop(key))}")
        else:
            new_lines.append(line)

    if remaining and new_lines and new_lines[-1].strip():
        new_lines.append("")
    for key, value in remaining.items():
        new_lines.append(f"{key}={env_quote(value)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def provider_key_name(provider: str) -> str:
    return {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider, "LLM_API_KEY")


def save_model_config_to_env(
    env_path: Path,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    provider = (provider or "local_hf").strip()
    model = (model or "").strip()
    base_url = (base_url or "").strip()
    updates = {
        "LLM_PROVIDER": provider,
        "LLM_TEMPERATURE": str(float(temperature)),
        "LLM_TOP_P": str(float(top_p)),
        "LLM_MAX_NEW_TOKENS": str(int(max_new_tokens)),
    }

    if provider == "local_hf":
        updates["CHAT_MODEL_NAME"] = model
    else:
        updates["LLM_API_MODEL"] = model
        if base_url:
            updates["LLM_API_BASE_URL"] = base_url
        model_key = {"deepseek": "DEEPSEEK_MODEL", "openai": "OPENAI_MODEL", "openrouter": "OPENROUTER_MODEL"}.get(provider)
        if model_key:
            updates[model_key] = model

    key_saved = bool(api_key and api_key.strip())
    if key_saved:
        updates[provider_key_name(provider)] = api_key.strip()

    upsert_env_values(env_path, updates)
    os.environ.update(updates)
    logger.info("模型配置已保存：provider=%s model=%s path=%s", provider, model, env_path.name)
    message = f"已保存模型配置到 {env_path.name}。"
    return message + (
        " API Key 已写入本机 .env.local，请不要提交该文件。"
        if key_saved else " API Key 输入为空，未改动已有密钥。"
    )


def normalize_local_dtype(dtype: str) -> str:
    normalized = (dtype or "auto").strip().lower()
    normalized = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(normalized, normalized)
    if normalized not in LOCAL_DTYPE_CHOICES:
        raise ValueError("模型精度仅支持 auto、bfloat16、float16 或 float32。")
    return normalized


def build_local_runtime_config_text(
    dtype: str, attention_implementation: str, low_cpu_mem_usage: bool,
    compile_model: bool, cpu_threads: int | float,
) -> str:
    attention = (attention_implementation or "").strip() or "自动 / 默认"
    threads = int(cpu_threads or 0)
    return "\n".join([
        "本地 Hugging Face 模型运行配置",
        f"模型精度：{normalize_local_dtype(dtype)}",
        f"注意力实现：{attention}",
        f"低内存加载：{'开启' if low_cpu_mem_usage else '关闭'}",
        f"torch.compile：{'开启' if compile_model else '关闭'}",
        f"CPU 线程：{'使用 PyTorch 默认值' if threads == 0 else threads}",
        "提示：配置保存后需重启应用，下一次加载本地模型时才会生效。",
    ])


def save_local_runtime_config_to_env(
    env_path: Path,
    dtype: str,
    attention_implementation: str,
    low_cpu_mem_usage: bool,
    compile_model: bool,
    cpu_threads: int | float,
) -> str:
    normalized_dtype = normalize_local_dtype(dtype)
    attention = (attention_implementation or "").strip()
    if "\r" in attention or "\n" in attention:
        raise ValueError("注意力实现不能包含换行符。")
    threads = int(cpu_threads or 0)
    if not 0 <= threads <= 512:
        raise ValueError("CPU 线程数应在 0 到 512 之间。")

    updates = {
        "LOCAL_MODEL_DTYPE": normalized_dtype,
        "LOCAL_MODEL_ATTN_IMPLEMENTATION": attention,
        "LOCAL_MODEL_LOW_CPU_MEM_USAGE": str(bool(low_cpu_mem_usage)).lower(),
        "LOCAL_MODEL_COMPILE": str(bool(compile_model)).lower(),
        "LOCAL_MODEL_CPU_THREADS": str(threads),
    }
    upsert_env_values(env_path, updates)
    os.environ.update(updates)
    logger.info("本地模型运行配置已保存：path=%s", env_path.name)
    return f"已保存本地模型运行配置到 {env_path.name}。请重启应用后再加载本地模型。"


def connection_result(level: str, message: str, detail: str = "") -> str:
    parts = [f"[{level}] {message}"]
    if detail:
        parts.append(f"详情：{detail}")
    return "\n".join(parts)


def classify_connection_error(exc: Exception, provider: str) -> str:
    text = str(exc)
    type_name = type(exc).__name__.lower()
    lowered = text.lower()
    provider = (provider or "").lower()
    if "api key" in lowered or "缺少" in text:
        return connection_result("配置错误", "缺少 API Key", text)
    if "module" in lowered or "缺少依赖" in text or "modulenotfounderror" in type_name:
        return connection_result("依赖缺失", "缺少运行依赖", text)
    if "401" in lowered or "unauthorized" in lowered or "authentication" in lowered:
        return connection_result("认证失败", "API Key 无效或无权限", text)
    if "429" in lowered or "rate limit" in lowered or "quota" in lowered or "insufficient" in lowered:
        return connection_result("限流/额度", "请求频率或账户额度受限", text)
    if any(word in lowered for word in ["timeout", "timed out", "connection", "network", "proxy", "dns", "ssl"]):
        return connection_result("网络错误", "无法稳定连接模型服务", text)
    if provider == "local_hf":
        return connection_result("本地模型错误", "本地模型加载或推理失败", text)
    return connection_result("响应异常", "模型服务返回异常响应", text)


def test_model_connection(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    *,
    make_config: Callable[..., Any],
    get_llm_fn: Callable[[], tuple[Any, Any]],
    require_openai_client_fn: Callable[[], Any],
    status_callback: Callable[[str], None],
) -> str:
    config = make_config(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    status_callback(f"正在测试 {config.normalized_provider()} / {config.resolved_model()}")
    try:
        if config.normalized_provider() == "local_hf":
            tokenizer, model_object = get_llm_fn()
            model_name = getattr(getattr(model_object, "config", None), "name_or_path", None)
            tokenizer_name = getattr(tokenizer, "name_or_path", None)
            loaded_name = model_name or tokenizer_name or config.resolved_model()
            result = connection_result("成功", f"本地模型加载成功：{loaded_name}")
        else:
            resolved_api_key = config.resolved_api_key()
            if not resolved_api_key:
                raise RuntimeError("当前 Provider 缺少 API Key，请在 GUI 输入或写入 .env.local")
            client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
            resolved_base_url = config.resolved_base_url()
            if resolved_base_url:
                client_kwargs["base_url"] = resolved_base_url
            client = require_openai_client_fn()(**client_kwargs)
            response = client.chat.completions.create(
                model=config.resolved_model(),
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
                stream=False,
            )
            if not response.choices:
                raise RuntimeError("接口返回成功，但没有 choices 字段")
            result = connection_result(
                "成功", f"API 连接成功：{config.normalized_provider()} / {config.resolved_model()}"
            )
    except Exception as exc:
        result = classify_connection_error(exc, config.normalized_provider())
    status_callback(result)
    return result


def build_connection_test_detail(
    result: str, provider: str, model: str, base_url: str, elapsed_seconds: float
) -> str:
    level = result.split("]", 1)[0].lstrip("[") if result.startswith("[") else "未知"
    suggestions = {
        "成功": "连接可用，可以开始对话。",
        "配置错误": "检查 Provider、模型名和 API Key 是否填写完整。",
        "认证失败": "检查 API Key、账户权限和所选模型是否可用。",
        "限流/额度": "稍后重试，或检查服务商账户余额与请求额度。",
        "网络错误": "检查网络、代理设置、Base URL 和证书环境。",
        "依赖缺失": "安装提示中提到的 Python 依赖后重启应用。",
        "本地模型错误": "检查模型名称、显存/内存和本地运行配置。",
        "响应异常": "检查服务商返回信息、模型名称和接口兼容性。",
    }
    endpoint = base_url.strip() or ("本地 Hugging Face" if provider == "local_hf" else "服务商默认地址")
    return "\n".join([
        f"测试状态：{level}",
        f"Provider：{provider or '未选择'}",
        f"模型：{model or '未选择'}",
        f"目标：{endpoint}",
        f"耗时：{elapsed_seconds:.2f} 秒",
        f"建议：{suggestions.get(level, '请根据下方详情检查配置。')}",
        "",
        result,
    ])
