"""Dependency-light request and result types for Kokoro synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from kokorog2p.language_codes import normalize_language_code, supported_languages

from .voice_manager import VoiceBlend

if TYPE_CHECKING:
    from .types import Trace, WordTiming

_SUPPORTED_LANGUAGES = frozenset(supported_languages())


def _validate_offset(start: int, end: int, text_length: int, label: str) -> None:
    if isinstance(start, bool) or not isinstance(start, int):
        raise TypeError(f"{label}.start must be an integer")
    if isinstance(end, bool) or not isinstance(end, int):
        raise TypeError(f"{label}.end must be an integer")
    if not 0 <= start < end <= text_length:
        raise ValueError(
            f"{label} offsets must satisfy 0 <= start < end <= {text_length}, got [{start}, {end})"
        )


def _normalize_language(language: str, label: str) -> str:
    if not isinstance(language, str) or not language.strip():
        raise ValueError(f"{label} must be a non-empty supported language code")
    normalized = normalize_language_code(language)
    if normalized not in _SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported {label}: {language!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class PronunciationOverride:
    """Source-aligned, already-resolved pronunciation instructions."""

    start: int
    end: int
    phonemes: str | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("pronunciation override start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("pronunciation override end must be an integer")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("pronunciation override must satisfy 0 <= start < end")
        if self.phonemes is not None and (
            not isinstance(self.phonemes, str) or not self.phonemes.strip()
        ):
            raise ValueError("pronunciation override phonemes must be non-empty or None")
        if self.language is not None:
            object.__setattr__(
                self, "language", _normalize_language(self.language, "override language")
            )
        if self.phonemes is None and self.language is None:
            raise ValueError("pronunciation override must set phonemes or language")


@dataclass(frozen=True, slots=True)
class LinguisticToken:
    """Source-aligned linguistic context passed to KokoroG2P."""

    start: int
    end: int
    text: str | None = None
    pos: str | None = None
    tag: str | None = None
    lemma: str | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("linguistic token start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("linguistic token end must be an integer")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("linguistic token must satisfy 0 <= start < end")
        for name in ("text", "pos", "tag", "lemma"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"linguistic token {name} must be a string or None")
        if self.language is not None:
            object.__setattr__(
                self, "language", _normalize_language(self.language, "token language")
            )


@dataclass(frozen=True, slots=True)
class SynthesisSegment:
    """One prepared, independent speech request for a Kokoro synthesis target."""

    id: str
    text: str
    language: str
    voice: str | VoiceBlend | None = None
    pronunciation_overrides: tuple[PronunciationOverride, ...] = ()
    annotations: tuple[LinguisticToken, ...] = ()
    phonemes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("synthesis segment id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("synthesis segment text must be a string")
        object.__setattr__(self, "language", _normalize_language(self.language, "language"))
        if self.voice is not None and not isinstance(self.voice, (str, VoiceBlend)):
            raise TypeError("synthesis segment voice must be a voice name, VoiceBlend, or None")
        if isinstance(self.voice, str) and not self.voice.strip():
            raise ValueError("synthesis segment voice must be non-empty when supplied")
        if self.phonemes is not None and not isinstance(self.phonemes, str):
            raise TypeError("synthesis segment phonemes must be a string or None")

        overrides = tuple(self.pronunciation_overrides)
        annotations = tuple(self.annotations)
        if any(not isinstance(item, PronunciationOverride) for item in overrides):
            raise TypeError("pronunciation_overrides must contain PronunciationOverride values")
        if any(not isinstance(item, LinguisticToken) for item in annotations):
            raise TypeError("annotations must contain LinguisticToken values")
        object.__setattr__(self, "pronunciation_overrides", overrides)
        object.__setattr__(self, "annotations", annotations)

        for index, override in enumerate(overrides):
            _validate_offset(override.start, override.end, len(self.text), f"override[{index}]")
        direct_overrides = sorted(
            (item for item in overrides if item.phonemes is not None), key=lambda item: item.start
        )
        for previous, current in zip(direct_overrides, direct_overrides[1:], strict=False):
            if current.start < previous.end:
                raise ValueError("overlapping direct phoneme overrides are ambiguous")
        if self.phonemes is not None and direct_overrides:
            raise ValueError(
                "whole-request phonemes cannot be combined with span phoneme overrides"
            )

        for index, annotation in enumerate(annotations):
            _validate_offset(
                annotation.start, annotation.end, len(self.text), f"annotation[{index}]"
            )
            if (
                annotation.text is not None
                and self.text[annotation.start : annotation.end] != annotation.text
            ):
                raise ValueError(
                    f"annotation[{index}] text does not match the source text at its offsets"
                )


@dataclass(slots=True)
class RenderedSegment:
    """Engine-local audio and metadata for exactly one synthesis request."""

    id: str
    audio: np.ndarray
    sample_rate: int
    text: str
    language: str
    voice: str | None
    phonemes: str
    token_ids: tuple[int, ...]
    word_timings: tuple[WordTiming, ...] = ()
    diagnostics: tuple[str, ...] = ()
    trace: Trace | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("rendered segment id must be a non-empty string")
        audio = np.asarray(self.audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("rendered segment audio must be mono (one-dimensional)")
        self.audio = audio
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("rendered segment sample_rate must be an integer")
        if self.sample_rate <= 0:
            raise ValueError("rendered segment sample_rate must be positive")
        self.language = _normalize_language(self.language, "rendered segment language")
        self.token_ids = tuple(int(token_id) for token_id in self.token_ids)
        self.word_timings = tuple(self.word_timings)
        self.diagnostics = tuple(self.diagnostics)
        for timing in self.word_timings:
            if timing.start_sample < 0 or timing.end_sample < timing.start_sample:
                raise ValueError("word timing sample bounds must be non-negative and ordered")
            if timing.end_sample > len(self.audio):
                raise ValueError("word timing exceeds rendered audio duration")

    def save_wav(self, path: str | Path) -> None:
        """Write this mono waveform as 32-bit floating-point WAV without clipping."""
        import soundfile

        soundfile.write(path, self.audio, self.sample_rate, subtype="FLOAT")

    def play(self, *, device: int | str | None = None) -> None:
        """Play this rendered waveform using the optional playback dependency."""
        from .playback import play_audio

        play_audio(self.audio, self.sample_rate, device=device)
