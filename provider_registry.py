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


@dataclass(frozen=True)
class ChatCompletionRequestProfile:
    """Per-model constraints for the OpenAI Chat Completions transport."""

    output_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    supports_temperature: bool = True
    supports_top_p: bool = True


DEFAULT_CHAT_COMPLETION_PROFILE = ChatCompletionRequestProfile()
GPT6_CHAT_COMPLETION_PROFILE = ChatCompletionRequestProfile(
    output_token_parameter="max_completion_tokens",
    supports_temperature=False,
    supports_top_p=False,
)


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
    model_request_profiles: tuple[tuple[str, ChatCompletionRequestProfile], ...] = ()
    base_url_env: str | None = None
    default_base_url: str = ""
    show_in_ui: bool = True

    def default_model_value(self) -> str:
        return env_or(self.model_env, self.default_model)

    def model_options(self) -> list[str]:
        return unique_choices([self.default_model_value(), *self.model_choices])

    def chat_completion_parameters(
        self, model: str, *, temperature: float, top_p: float, max_new_tokens: int
    ) -> dict[str, float | int]:
        profile = dict(self.model_request_profiles).get(model, DEFAULT_CHAT_COMPLETION_PROFILE)
        parameters: dict[str, float | int] = {
            profile.output_token_parameter: max_new_tokens,
        }
        if profile.supports_temperature:
            parameters["temperature"] = temperature
        if profile.supports_top_p:
            parameters["top_p"] = top_p
        return parameters

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
            "claude-haiku-4-5",
            "claude-sonnet-5",
            "claude-opus-5",
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
        default_model="deepseek-flash",
        model_choices=("deepseek-flash", "deepseek-v4-pro"),
        default_base_url="https://api.deepseek.com",
    ),
    "openai": ProviderDefinition(
        id="openai",
        label="OpenAI",
        kind="openai_compatible",
        api_key_envs=("OPENAI_API_KEY", "LLM_API_KEY"),
        model_env="OPENAI_MODEL",
        default_model="gpt-6-luna",
        model_choices=(
            "gpt-6-luna",
            "gpt-6-sol",
            "gpt-4o",
        ),
        model_request_profiles=(
            ("gpt-6-luna", GPT6_CHAT_COMPLETION_PROFILE),
            ("gpt-6-sol", GPT6_CHAT_COMPLETION_PROFILE),
            # Astra remains configurable through OPENAI_MODEL or a custom id,
            # but is deliberately not a standard UI choice while costs are reviewed.
            ("gpt-6-astra", GPT6_CHAT_COMPLETION_PROFILE),
        ),
    ),
    "openrouter": ProviderDefinition(
        id="openrouter",
        label="OpenRouter",
        kind="openai_compatible",
        api_key_envs=("OPENROUTER_API_KEY", "LLM_API_KEY"),
        model_env="OPENROUTER_MODEL",
        default_model="openrouter/auto",
        model_choices=(
            "openrouter/auto",
            "openrouter/free",
        ),
        default_base_url="https://openrouter.ai/api/v1",
    ),
    "openai_compatible": ProviderDefinition(
        id="openai_compatible",
        label="OpenAI-compatible",
        kind="openai_compatible",
        model_env="LLM_API_MODEL",
        default_model=DEFAULT_API_MODEL,
        model_choices=("deepseek-flash", "qwen-plus", "moonshot-v1-8k"),
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
