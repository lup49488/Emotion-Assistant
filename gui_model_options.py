from __future__ import annotations

from provider_registry import PROVIDER_CHOICES, PROVIDER_DEFINITIONS, unique_choices


PROVIDER_LABELS = {
    provider_id: definition.label
    for provider_id, definition in PROVIDER_DEFINITIONS.items()
    if definition.show_in_ui
}


def provider_dropdown_choices() -> list[tuple[str, str]]:
    """Return (label, value) pairs for the Provider dropdown."""
    return [(PROVIDER_LABELS.get(provider, provider), provider) for provider in PROVIDER_CHOICES]


PROVIDER_API_KEYS = {
    provider_id: list(definition.api_key_envs)
    for provider_id, definition in PROVIDER_DEFINITIONS.items()
}

MODEL_CHOICES = {
    provider_id: definition.model_options()
    for provider_id, definition in PROVIDER_DEFINITIONS.items()
}

DEFAULT_MODELS = {
    provider: choices[0] for provider, choices in MODEL_CHOICES.items()
}

DEFAULT_BASE_URLS = {
    provider_id: definition.base_url_value()
    for provider_id, definition in PROVIDER_DEFINITIONS.items()
}
