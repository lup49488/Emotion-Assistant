"""Small, platform-aware helpers for protecting local application data."""
from __future__ import annotations

import os
from pathlib import Path


def restrict_directory(path: Path) -> None:
    """Create a private directory and tighten its mode where supported."""
    path.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        path.chmod(0o700)


def restrict_file(path: Path) -> None:
    """Tighten an existing sensitive file to owner read/write on POSIX."""
    if os.name != "nt" and path.exists():
        path.chmod(0o600)

