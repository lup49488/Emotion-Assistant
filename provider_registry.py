from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from config import (
    ANTHROPIC_BASE_URL,
    ANTHROPIC_MODEL,
    CHAT_MODEL_NAME,
    DEFAULT_API_BASE_URL,
    DEFAULT_API_MODEL,
)


ProviderKind = Literal["local", "openai_compatible", "anthropic"]


def env_or(name: str | None, default: str) -> str:
    if not name:
        return default
    return (os.getenv(name) or "").strip() or default


def unique_choices(values: list[str]) -> list[str]:
    choices: list[str] = []
    for value in values:
        candidate = (value or "").strip()
        if candidate and candidate not in choices:
            choices.append(candidate)
    return choices


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    label: str
    kind: ProviderKind
    api_key_envs: tuple[str, ...] = ("LLM_API_KEY",)
    model_env: str | None = None
    default_model: str = DEFAULT_API_MODEL
    model_choices: tuple[str, ...] = ()
    base_url_env: str | None = None
    default_base_url: str = ""
    show_in_ui: bool = True

    def default_model_value(self) -> str:
        return env_or(self.model_env, self.default_model)

    def model_options(self) -> list[str]:
        return unique_choices([self.default_model_value(), *self.model_choices])

    def base_url_value(self) -> str:
        return env_or(self.base_url_env, self.default_base_url)

    def api_key(self) -> str | None:
        for env_name in self.api_key_envs:
            value = (os.getenv(env_name) or "").strip()
            if value:
                return value
        return None


PROVIDER_DEFINITIONS: dict[str, ProviderDefinition] = {
    "local_hf": ProviderDefinition(
        id="local_hf",
        label="Local Hugging Face",
        kind="local",
        api_key_envs=(),
        model_env="CHAT_MODEL_NAME",
        default_model=CHAT_MODEL_NAME,
        model_choices=(
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
        show_in_ui=False,
    ),
    "anthropic": ProviderDefinition(
        id="anthropic",
        label="Anthropic (Claude)",
        kind="anthropic",
        api_key_envs=("ANTHROPIC_API_KEY", "LLM_API_KEY"),
        model_env="ANTHROPIC_MODEL",
        default_model=ANTHROPIC_MODEL,
        model_choices=(
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-haiku-4-5",
        ),
        base_url_env="ANTHROPIC_BASE_URL",
        default_base_url=ANTHROPIC_BASE_URL,
    ),
    "deepseek": ProviderDefinition(
        id="deepseek",
        label="DeepSeek",
        kind="openai_compatible",
        api_key_envs=("DEEPSEEK_API_KEY", "LLM_API_KEY"),
        model_env="DEEPSEEK_MODEL",
        default_model="deepseek-chat",
        model_choices=("deepseek-chat", "deepseek-reasoner"),
        default_base_url="https://api.deepseek.com",
    ),
    "openai": ProviderDefinition(
        id="openai",
        label="OpenAI",
        kind="openai_compatible",
        api_key_envs=("OPENAI_API_KEY", "LLM_API_KEY"),
        model_env="OPENAI_MODEL",
        default_model="gpt-4.1-mini",
        model_choices=("gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"),
    ),
    "openrouter": ProviderDefinition(
        id="openrouter",
        label="OpenRouter",
        kind="openai_compatible",
        api_key_envs=("OPENROUTER_API_KEY", "LLM_API_KEY"),
        model_env="OPENROUTER_MODEL",
        default_model="openai/gpt-4.1-mini",
        model_choices=(
            "openai/gpt-4.1-mini",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.0-flash-001",
            "deepseek/deepseek-chat",
        ),
        default_base_url="https://openrouter.ai/api/v1",
    ),
    "nvidia_nim": ProviderDefinition(
        id="nvidia_nim",
        label="NVIDIA NIM",
        kind="openai_compatible",
        api_key_envs=("NVIDIA_NIM_API_KEY", "LLM_API_KEY"),
        model_env="NVIDIA_NIM_MODEL",
        default_model="openai/gpt-oss-20b",
        model_choices=(
            "openai/gpt-oss-20b",
            "meta/llama-3.1-8b-instruct",
            "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        ),
        base_url_env="NVIDIA_NIM_BASE_URL",
        default_base_url="https://integrate.api.nvidia.com/v1",
    ),
    "openai_compatible": ProviderDefinition(
        id="openai_compatible",
        label="OpenAI-compatible",
        kind="openai_compatible",
        model_env="LLM_API_MODEL",
        default_model=DEFAULT_API_MODEL,
        model_choices=("deepseek-chat", "qwen-plus", "moonshot-v1-8k"),
        base_url_env="LLM_API_BASE_URL",
        default_base_url=DEFAULT_API_BASE_URL,
    ),
    "custom": ProviderDefinition(
        id="custom",
        label="Custom endpoint",
        kind="openai_compatible",
        model_env="LLM_API_MODEL",
        default_model=DEFAULT_API_MODEL,
        base_url_env="LLM_API_BASE_URL",
        default_base_url=DEFAULT_API_BASE_URL,
    ),
}

PROVIDER_CHOICES = [provider_id for provider_id, definition in PROVIDER_DEFINITIONS.items() if definition.show_in_ui]
API_PROVIDER_IDS = {
    provider_id for provider_id, definition in PROVIDER_DEFINITIONS.items()
    if definition.kind == "openai_compatible"
}
ANTHROPIC_PROVIDER_ID = "anthropic"


def provider_definition(provider: str) -> ProviderDefinition | None:
    return PROVIDER_DEFINITIONS.get((provider or "").strip().lower())


def provider_catalog() -> dict[str, object]:
    providers = []
    for provider_id in PROVIDER_CHOICES:
        definition = PROVIDER_DEFINITIONS[provider_id]
        providers.append({
            "id": definition.id,
            "label": definition.label,
            "kind": definition.kind,
            "models": definition.model_options(),
            "default_model": definition.default_model_value(),
            "default_base_url": definition.base_url_value(),
            "api_key_envs": list(definition.api_key_envs),
        })
    return {"providers": providers}
