from __future__ import annotations

import os
from pathlib import Path


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    escaped = False

    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_double:
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                return value[:index].rstrip()
    return value.strip()


def _parse_env_value(raw_value: str) -> str:
    value = _strip_inline_comment(raw_value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
        if raw_value.strip().startswith('"'):
            value = value.encode("utf-8").decode("unicode_escape")
    return value


def load_dotenv(path: Path, *, override: bool = False) -> bool:
    if not path.exists() or not path.is_file():
        return False

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key or (key in os.environ and not override):
            continue
        os.environ[key] = _parse_env_value(raw_value)
    return True


SKIP_DOTENV_ENV_VAR = "CHATBOT_SKIP_DOTENV"


def dotenv_loading_disabled() -> bool:
    return os.getenv(SKIP_DOTENV_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def load_project_env_if_enabled(base_dir: Path) -> list[Path]:
    """Load the project's own .env unless the test guard disables it.

    Every module that loads the project .env at import time must go through this.
    Calling load_project_env() directly injects the developer's real configuration
    into os.environ for the rest of the process — in a test session that leaks into
    every module imported afterwards and makes results depend on file order.
    """
    return [] if dotenv_loading_disabled() else load_project_env(base_dir)


def load_project_env(base_dir: Path) -> list[Path]:
    loaded: list[Path] = []
    protected_keys = set(os.environ)

    env_path = base_dir / ".env"
    if load_dotenv(env_path, override=False):
        loaded.append(env_path)

    local_env_path = base_dir / ".env.local"
    if local_env_path.exists() and local_env_path.is_file():
        for line in local_env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            if "=" not in stripped:
                continue

            key, raw_value = stripped.split("=", 1)
            key = key.strip()
            if not key or key in protected_keys:
                continue
            os.environ[key] = _parse_env_value(raw_value)
        loaded.append(local_env_path)

    return loaded
