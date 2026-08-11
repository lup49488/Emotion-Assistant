from __future__ import annotations

import os

from config import DEFAULT_API_BASE_URL, DEFAULT_API_MODEL


PROVIDER_CHOICES = [
    "anthropic",
    "deepseek",
    "openai",
    "openrouter",
    "nvidia_nim",
    "openai_compatible",
    "custom",
]

# Dropdown labels. The stored value stays the provider id, so .env files, the API
# contract, and everything keyed on PROVIDER_CHOICES are unaffected — this only
# changes what the operator reads. "anthropic" in particular is hard to find when
# you are looking for Claude.
PROVIDER_LABELS = {
    "anthropic": "Anthropic (Claude)",
    "deepseek": "DeepSeek",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "nvidia_nim": "NVIDIA NIM",
    "openai_compatible": "OpenAI-compatible",
    "custom": "Custom endpoint",
}


def provider_dropdown_choices() -> list[tuple[str, str]]:
    """Return (label, value) pairs for the Provider dropdown."""
    return [(PROVIDER_LABELS.get(provider, provider), provider) for provider in PROVIDER_CHOICES]


PROVIDER_API_KEYS = {
    "anthropic": ["ANTHROPIC_API_KEY", "LLM_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY", "LLM_API_KEY"],
    "openai": ["OPENAI_API_KEY", "LLM_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY", "LLM_API_KEY"],
    "nvidia_nim": ["NVIDIA_NIM_API_KEY", "LLM_API_KEY"],
    "openai_compatible": ["LLM_API_KEY"],
    "custom": ["LLM_API_KEY"],
}


def unique_choices(values: list[str]) -> list[str]:
    choices: list[str] = []
    for value in values:
        if value and value not in choices:
            choices.append(value)
    return choices


MODEL_CHOICES = {
    "local_hf": unique_choices([
        os.getenv("CHAT_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"),
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]),
    "anthropic": unique_choices([
        os.getenv("ANTHROPIC_MODEL", "claude-opus-5"),
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5",
    ]),
    "deepseek": unique_choices([
        os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "deepseek-chat",
        "deepseek-reasoner",
    ]),
    "openai": unique_choices([
        os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-4o",
    ]),
    "openrouter": unique_choices([
        os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
        "openai/gpt-4.1-mini",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.0-flash-001",
        "deepseek/deepseek-chat",
    ]),
    "nvidia_nim": unique_choices([
        os.getenv("NVIDIA_NIM_MODEL", "openai/gpt-oss-20b"),
        "openai/gpt-oss-20b",
        "meta/llama-3.1-8b-instruct",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    ]),
    "openai_compatible": unique_choices([
        DEFAULT_API_MODEL,
        "deepseek-chat",
        "qwen-plus",
        "moonshot-v1-8k",
    ]),
    "custom": unique_choices([
        DEFAULT_API_MODEL,
    ]),
}

DEFAULT_MODELS = {
    provider: choices[0] for provider, choices in MODEL_CHOICES.items()
}

DEFAULT_BASE_URLS = {
    "local_hf": "",
    # Empty means the Anthropic SDK's own endpoint.
    "anthropic": "",
    "deepseek": "https://api.deepseek.com",
    "openai": "",
    "openrouter": "https://openrouter.ai/api/v1",
    "nvidia_nim": os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    "openai_compatible": DEFAULT_API_BASE_URL,
    "custom": DEFAULT_API_BASE_URL,
}
