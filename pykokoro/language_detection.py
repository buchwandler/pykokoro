"""Configuration for optional automatic pronunciation-language routing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from kokorog2p.language_codes import normalize_language_code, supported_languages

LanguageDetectionMode = Literal["off", "auto"]
LanguageDetectionSource = Literal["run", "pipeline", "header", "default"]


@dataclass(frozen=True, slots=True)
class LanguageDetectionConfig:
    """Configure optional automatic language routing in the G2P stage."""

    mode: LanguageDetectionMode = "off"
    languages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("off", "auto"):
            raise ValueError("language detection mode must be 'off' or 'auto'")
        raw_languages: Any = self.languages
        if isinstance(raw_languages, str):
            raise TypeError("language detection languages must be a sequence, not a string")
        supported = set(supported_languages())
        normalized: list[str] = []
        for language in raw_languages:
            if not isinstance(language, str):
                raise TypeError("language detection languages must contain strings")
            canonical = normalize_language_code(language)
            if canonical not in supported:
                raise ValueError(f"unsupported language for detection: {language!r}")
            if canonical not in normalized:
                normalized.append(canonical)
        if self.mode == "auto" and len(normalized) < 2:
            raise ValueError("automatic language detection requires at least two languages")
        object.__setattr__(self, "languages", tuple(normalized))


def resolve_language_detection(
    config: LanguageDetectionConfig | None,
    header: Mapping[str, object],
) -> ResolvedLanguageDetection:
    """Resolve API configuration before a document header hint."""
    if config is not None:
        return ResolvedLanguageDetection(
            mode=config.mode,
            languages=config.languages,
            source="pipeline",
        )
    direct_hint = header.get("language_detection")
    if "language_detection" in header and direct_hint is None:
        return ResolvedLanguageDetection()
    if isinstance(direct_hint, Mapping):
        mode = direct_hint.get("mode", "off")
        languages = direct_hint.get("languages", ())
        if isinstance(mode, str) and isinstance(languages, (list, tuple)):
            normalized = LanguageDetectionConfig(
                mode=cast(LanguageDetectionMode, mode),
                languages=tuple(languages),
            )
            return ResolvedLanguageDetection(
                mode=normalized.mode,
                languages=normalized.languages,
                source="header",
            )
    import ssmd

    hint = ssmd.language_detection_hint(header)
    if hint is None:
        return ResolvedLanguageDetection()
    normalized = LanguageDetectionConfig(mode=hint.mode, languages=tuple(hint.languages))
    return ResolvedLanguageDetection(
        mode=normalized.mode,
        languages=normalized.languages,
        source="header",
    )


@dataclass(frozen=True, slots=True)
class ResolvedLanguageDetection:
    """Effective routing configuration for one synthesis request."""

    mode: LanguageDetectionMode = "off"
    languages: tuple[str, ...] = ()
    source: LanguageDetectionSource = "default"

    def as_routing(self) -> dict[str, object] | None:
        """Return the KokoroG2P routing mapping, or ``None`` when disabled."""
        if self.mode == "off":
            return None
        return {"mode": self.mode, "languages": self.languages}


__all__ = [
    "LanguageDetectionConfig",
    "LanguageDetectionMode",
    "LanguageDetectionSource",
    "ResolvedLanguageDetection",
    "resolve_language_detection",
]
