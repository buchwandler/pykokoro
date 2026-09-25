"""Small local configuration and cache-path helpers for the synthesis engine."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir, user_config_dir

from .constants import DEFAULT_CONFIG


def get_user_config_path() -> Path:
    """Return the user-local PyKokoro configuration path."""
    if platform.system() != "Windows":
        custom_dir = Path.home() / ".config" / "pykokoro"
        if custom_dir.exists():
            return custom_dir / "config.json"
    return (
        Path(user_config_dir("pykokoro", appauthor=False, roaming=True, ensure_exists=True))
        / "config.json"
    )


def get_user_cache_path(folder: str | None = None) -> Path:
    """Return the user-local cache directory, optionally creating a subfolder."""
    cache_dir = Path(user_cache_dir("pykokoro", appauthor=False, opinion=True, ensure_exists=True))
    if folder:
        cache_dir = cache_dir / folder
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_config() -> dict[str, Any]:
    """Load user settings, filling omitted values from engine defaults."""
    try:
        config_path = get_user_config_path()
        if config_path.exists():
            user_config = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(user_config, dict):
                return {**DEFAULT_CONFIG, **user_config}
    except (OSError, ValueError):
        pass
    return DEFAULT_CONFIG.copy()


def save_config(config: dict[str, Any]) -> bool:
    """Save user settings, returning False if local storage cannot be written."""
    try:
        config_path = get_user_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return True
    except (OSError, ValueError):
        return False


def reset_config() -> dict[str, Any]:
    """Restore engine defaults and persist them for the current user."""
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG.copy()
