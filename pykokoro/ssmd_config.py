"""Configuration and pure helpers for SSMD 0.8 document rendering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal

from .exceptions import SSMDDocumentError


@dataclass(frozen=True)
class SSMDPauseOverrides:
    """Optional application-level overrides for document pause defaults."""

    enabled: bool | None = None
    sentence: str | None = None
    paragraph: str | None = None
    voice_change: str | None = None


@dataclass(frozen=True)
class ResolvedPauseDefaults:
    """Validated pause defaults, represented in seconds for the renderer."""

    enabled: bool
    sentence: float | None = None
    paragraph: float | None = None
    voice_change: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sentence": self.sentence,
            "paragraph": self.paragraph,
            "voice_change": self.voice_change,
        }


@dataclass(frozen=True)
class VoiceResolution:
    """Logical voice reference and its final concrete target."""

    reference: str
    target: str
    source: Literal["api", "header", "direct"]


@dataclass(frozen=True)
class SSMDDiagnostic:
    """Stable structured diagnostic emitted while rendering an SSMD document."""

    code: str
    severity: Literal["info", "warn", "error"]
    message: str
    line: int | None = None
    column: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "line": self.line,
            "column": self.column,
        }


@dataclass(frozen=True)
class PauseCandidate:
    """A pause proposal before deterministic boundary reduction."""

    position: int
    duration_s: float
    source: Literal["explicit", "api_default", "header_default", "pipeline_default"]
    kind: Literal["break", "sentence", "paragraph", "voice_change"]
    priority: int


@dataclass(frozen=True)
class SSMDRenderConfig:
    """Renderer-owned controls for consuming SSMD portable metadata.

    ``emphasis_mode`` defaults to ``"plain"``: emphasis metadata is preserved,
    but speech remains unmodified. ``"approximate"`` applies deterministic
    volume-only changes for strong, moderate, and reduced emphasis. ``"warn"``
    preserves unmodified speech and emits one diagnostic per logical source
    segment, while ``"error"`` rejects effectful emphasis before inference.
    SSMD ``emphasis="none"`` is ordinary speech and is accepted silently in
    every mode.
    """

    parse_header: bool = True
    provider: str = "kokoro"
    voice_bindings: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    pause_defaults: SSMDPauseOverrides | None = None
    strict_header: bool = True
    unknown_header: Literal["warn", "error", "ignore"] = "warn"
    missing_voice: Literal["error", "use-default"] = "error"
    validate_profile: bool = True
    emphasis_mode: Literal["plain", "approximate", "warn", "error"] = "plain"
    emphasis_gain_scale: float = 1.0
    audio_source_resolver: Any | None = None
    audio_max_bytes: int = 20_000_000
    audio_max_duration_s: float = 120.0

    def __post_init__(self) -> None:
        if not isinstance(self.parse_header, bool):
            raise TypeError("parse_header must be a boolean")
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider must be a non-empty string")
        if self.unknown_header not in {"warn", "error", "ignore"}:
            raise ValueError("unknown_header must be 'warn', 'error', or 'ignore'")
        if self.missing_voice not in {"error", "use-default"}:
            raise ValueError("missing_voice must be 'error' or 'use-default'")
        if self.emphasis_mode not in {"plain", "approximate", "warn", "error"}:
            raise ValueError("emphasis_mode must be 'plain', 'approximate', 'warn', or 'error'")
        scale: object = self.emphasis_gain_scale
        if isinstance(scale, bool) or not isinstance(scale, Real):
            raise ValueError("emphasis_gain_scale must be finite and between 0.0 and 2.0")
        numeric_scale = float(scale)
        if not math.isfinite(numeric_scale) or not 0.0 <= numeric_scale <= 2.0:
            raise ValueError("emphasis_gain_scale must be finite and between 0.0 and 2.0")
        if isinstance(self.audio_max_bytes, bool) or self.audio_max_bytes <= 0:
            raise ValueError("audio_max_bytes must be a positive integer")
        if self.audio_max_duration_s < 0:
            raise ValueError("audio_max_duration_s must be non-negative")
        _validate_bindings(self.voice_bindings, field_name="voice_bindings")
        if self.pause_defaults is not None and not isinstance(
            self.pause_defaults, SSMDPauseOverrides
        ):
            raise TypeError("pause_defaults must be SSMDPauseOverrides or None")


def _validate_bindings(value: Mapping[str, Mapping[str, str]], *, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    for provider, bindings in value.items():
        if not isinstance(provider, str) or not provider:
            raise ValueError(f"{field_name} provider names must be non-empty strings")
        if not isinstance(bindings, Mapping):
            raise TypeError(f"{field_name}.{provider} must be a mapping")
        for reference, target in bindings.items():
            if not isinstance(reference, str) or not reference:
                raise ValueError(f"{field_name}.{provider} references must be non-empty strings")
            if not isinstance(target, str) or not target:
                raise ValueError(
                    f"{field_name}.{provider}.{reference} target must be a non-empty string"
                )


def _parse_pause_value(value: object, *, field_name: str) -> float:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise SSMDDocumentError(
            f"pause_defaults.{field_name} must use NUMBERms or NUMBERs",
            code="header.pause_duration_invalid",
        )
    try:
        from ssmd.durations import duration_milliseconds

        milliseconds = duration_milliseconds(value)
    except (TypeError, ValueError) as exc:
        raise SSMDDocumentError(
            f"pause_defaults.{field_name} must use NUMBERms or NUMBERs",
            code="header.pause_duration_invalid",
        ) from exc
    return milliseconds / 1000.0


def resolve_pause_defaults(
    header_value: Mapping[str, Any] | None,
    override: SSMDPauseOverrides | None = None,
) -> ResolvedPauseDefaults | None:
    """Validate and merge header pause defaults with API overrides."""

    if not header_value and override is None:
        return None
    if header_value is not None and not isinstance(header_value, Mapping):
        raise SSMDDocumentError(
            "pause_defaults must be a mapping",
            code="header.pause_defaults_invalid",
        )
    data = dict(header_value or {})
    unknown = set(data) - {"enabled", "sentence", "paragraph", "voice_change"}
    if unknown:
        raise SSMDDocumentError(
            f"Unsupported pause_defaults fields: {', '.join(sorted(unknown))}",
            code="header.pause_defaults_invalid",
        )

    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise SSMDDocumentError(
            "pause_defaults.enabled must be a boolean",
            code="header.pause_enabled_invalid",
        )
    if override is not None and override.enabled is not None:
        enabled = override.enabled

    values: dict[str, float | None] = {}
    for name in ("sentence", "paragraph", "voice_change"):
        value = data.get(name)
        if override is not None:
            override_value = getattr(override, name)
            if override_value is not None:
                value = override_value
        values[name] = None if value is None else _parse_pause_value(value, field_name=name)

    if not enabled:
        return ResolvedPauseDefaults(enabled=False)
    if not any(value is not None for value in values.values()):
        raise SSMDDocumentError(
            "enabled pause_defaults require at least one timing field",
            code="header.pause_defaults_empty",
        )
    return ResolvedPauseDefaults(enabled=True, **values)


def resolve_document_voice(
    reference: str,
    *,
    provider: str,
    api_bindings: Mapping[str, Mapping[str, str]],
    header_bindings: Mapping[str, Mapping[str, str]],
) -> VoiceResolution:
    """Resolve a body voice reference using the documented precedence."""

    api_target = api_bindings.get(provider, {}).get(reference)
    if api_target is not None:
        return VoiceResolution(reference, api_target, "api")
    header_target = header_bindings.get(provider, {}).get(reference)
    if header_target is not None:
        return VoiceResolution(reference, header_target, "header")
    return VoiceResolution(reference, reference, "direct")


def copy_public_header(header: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursively copied header suitable for result metadata."""

    def copy(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [copy(item) for item in value]
        if isinstance(value, tuple):
            return [copy(item) for item in value]
        return value

    return copy(dict(header))


__all__ = [
    "PauseCandidate",
    "ResolvedPauseDefaults",
    "SSMDDiagnostic",
    "SSMDPauseOverrides",
    "SSMDRenderConfig",
    "VoiceResolution",
    "copy_public_header",
    "resolve_document_voice",
    "resolve_pause_defaults",
]
