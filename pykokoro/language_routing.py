"""Explicit engine-side automatic pronunciation-language routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kokorog2p.language_codes import normalize_language_code, supported_languages

LanguageRoutingMode = Literal["off", "auto"]
_SUPPORTED_LANGUAGES = frozenset(supported_languages())


@dataclass(frozen=True, slots=True)
class LanguageRoutingConfig:
    """Configure optional automatic language routing for prepared speech."""

    mode: LanguageRoutingMode = "off"
    languages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("off", "auto"):
            raise ValueError("language routing mode must be 'off' or 'auto'")
        if isinstance(self.languages, str):
            raise TypeError("language routing languages must be a sequence, not a string")
        normalized: list[str] = []
        for language in self.languages:
            if not isinstance(language, str) or not language.strip():
                raise ValueError("language routing languages must be non-empty language codes")
            canonical = normalize_language_code(language)
            if canonical not in _SUPPORTED_LANGUAGES:
                raise ValueError(f"unsupported language for routing: {language!r}")
            if canonical not in normalized:
                normalized.append(canonical)
        if self.mode == "auto" and len(normalized) < 2:
            raise ValueError("automatic language routing requires at least two languages")
        object.__setattr__(self, "languages", tuple(normalized))


def resolve_language_routing(
    config: LanguageRoutingConfig | None,
) -> dict[str, object] | None:
    """Return KokoroG2P routing settings from engine configuration only."""
    if config is None or config.mode == "off":
        return None
    return {"mode": config.mode, "languages": config.languages}


__all__ = ["LanguageRoutingConfig", "LanguageRoutingMode", "resolve_language_routing"]
