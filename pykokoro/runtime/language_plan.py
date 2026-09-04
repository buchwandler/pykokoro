"""Authoritative language runs for the orchestration pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..constants import ESPEAK_ONLY_LANGUAGES, SUPPORTED_LANGUAGES
from ..types import AnnotationSpan

_LANGUAGE_ALIASES = {
    "en": "en-us",
    "fr": "fr-fr",
    "cmn": "zh",
}


@dataclass(frozen=True, slots=True)
class LanguageRun:
    """A maximal half-open text range with one effective language."""

    char_start: int
    char_end: int
    language: str


def canonicalize_language(language: str) -> str:
    """Normalize and validate one document or span language."""
    if not isinstance(language, str):
        raise TypeError(f"language must be a string, got {type(language)!r}")
    normalized = language.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("language must not be empty")
    normalized = _LANGUAGE_ALIASES.get(normalized, normalized)
    supported = set(SUPPORTED_LANGUAGES) | set(ESPEAK_ONLY_LANGUAGES)
    base_language = normalized.split("-", 1)[0]
    if normalized not in supported and base_language not in supported:
        raise ValueError(f"Unsupported language {language!r}")
    return normalized


def _strictly_contains(
    outer_start: int,
    outer_end: int,
    inner_start: int,
    inner_end: int,
) -> bool:
    return (
        outer_start <= inner_start
        and inner_end <= outer_end
        and (outer_start, outer_end) != (inner_start, inner_end)
    )


def build_language_plan(
    text: str,
    annotations: Sequence[AnnotationSpan],
    *,
    default_language: str,
) -> tuple[LanguageRun, ...]:
    """Build deterministic, non-overlapping language runs for ``text``.

    Explicit ``lang`` annotations override the default only over their own range.
    Nested spans with the same language are harmless. Overlapping spans with
    different effective languages are rejected rather than guessed.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text)!r}")
    default = canonicalize_language(default_language)
    explicit: list[tuple[int, int, str]] = []
    for annotation in annotations:
        language = annotation.attrs.get("lang")
        if language is None:
            continue
        start = annotation.char_start
        end = annotation.char_end
        if not isinstance(start, int) or isinstance(start, bool):
            raise TypeError("language span start must be an integer")
        if not isinstance(end, int) or isinstance(end, bool):
            raise TypeError("language span end must be an integer")
        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"Language span {start}:{end} is outside the supplied text")
        explicit.append((start, end, canonicalize_language(language)))

    for index, (start, end, language) in enumerate(explicit):
        for other_start, other_end, other_language in explicit[index + 1 :]:
            if language == other_language or not (start < other_end and other_start < end):
                continue
            nested = _strictly_contains(start, end, other_start, other_end) or _strictly_contains(
                other_start, other_end, start, end
            )
            if nested:
                continue
            raise ValueError(
                "Conflicting overlapping language spans: "
                f"{start}:{end}={language!r} and "
                f"{other_start}:{other_end}={other_language!r}"
            )
    positions = {0, len(text)}
    for start, end, _language in explicit:
        positions.update((start, end))
    ordered_positions = sorted(positions)
    runs: list[LanguageRun] = []
    for start, end in zip(ordered_positions, ordered_positions[1:], strict=False):
        if end <= start:
            continue
        covering = [span for span in explicit if span[0] <= start and end <= span[1]]
        selected = (
            min(covering, key=lambda span: (span[1] - span[0], -span[0])) if covering else None
        )
        language = selected[2] if selected is not None else default
        if runs and runs[-1].language == language and runs[-1].char_end == start:
            runs[-1] = LanguageRun(runs[-1].char_start, end, language)
        else:
            runs.append(LanguageRun(start, end, language))
    return tuple(runs)


__all__ = ["LanguageRun", "build_language_plan", "canonicalize_language"]
