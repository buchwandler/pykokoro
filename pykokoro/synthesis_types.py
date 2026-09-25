"""Dependency-light request and result types for Kokoro synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from kokorog2p.language_codes import normalize_language_code, supported_languages

from .exceptions import (
    EmptyTextError,
    InvalidLanguageError,
    InvalidLinguisticTokensError,
    InvalidPronunciationError,
    InvalidRequestError,
    InvalidVoiceError,
)
from .voice_manager import VoiceBlend

if TYPE_CHECKING:
    from .synthesis_identity import SynthesisIdentity
    from .types import Trace, WordTiming
    from .voice_level import VoiceLevelApplication

_SUPPORTED_LANGUAGES = frozenset(supported_languages())


def _validate_offset(
    start: int,
    end: int,
    text_length: int,
    label: str,
    error_type: type[ValueError] = InvalidLinguisticTokensError,
) -> None:
    if isinstance(start, bool) or not isinstance(start, int):
        raise error_type(f"{label}.start must be an integer")
    if isinstance(end, bool) or not isinstance(end, int):
        raise error_type(f"{label}.end must be an integer")
    if not 0 <= start < end <= text_length:
        raise error_type(
            f"{label} offsets must satisfy 0 <= start < end <= {text_length}, got [{start}, {end})"
        )


def _normalize_language(language: str, label: str) -> str:
    if not isinstance(language, str) or not language.strip():
        raise InvalidLanguageError(f"{label} must be a non-empty supported language code")
    normalized = normalize_language_code(language)
    if normalized not in _SUPPORTED_LANGUAGES:
        raise InvalidLanguageError(f"unsupported {label}: {language!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class PronunciationOverride:
    """Source-aligned, already-resolved pronunciation instructions."""

    start: int
    end: int
    phonemes: str | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        _validate_offset(
            self.start, self.end, self.end, "pronunciation override", InvalidPronunciationError
        )
        if self.phonemes is not None and (
            not isinstance(self.phonemes, str) or not self.phonemes.strip()
        ):
            raise InvalidPronunciationError(
                "pronunciation override phonemes must be non-empty or None"
            )
        if self.language is not None:
            object.__setattr__(
                self, "language", _normalize_language(self.language, "override language")
            )
        if self.phonemes is None and self.language is None:
            raise InvalidPronunciationError("pronunciation override must set phonemes or language")


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
    morph: str | None = None

    def __post_init__(self) -> None:
        _validate_offset(
            self.start, self.end, self.end, "linguistic token", InvalidLinguisticTokensError
        )
        for name in ("text", "pos", "tag", "lemma", "morph"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise InvalidLinguisticTokensError(
                    f"linguistic token {name} must be a string or None"
                )
        if self.language is not None:
            object.__setattr__(
                self, "language", _normalize_language(self.language, "token language")
            )


@dataclass(frozen=True, slots=True, init=False)
class SynthesisRequest:
    """One atomic, caller-prepared speech request for a Kokoro synthesis target."""

    id: str
    text: str
    language: str
    voice: str | VoiceBlend | None
    pronunciation_overrides: tuple[PronunciationOverride, ...]
    tokens: tuple[LinguisticToken, ...]
    phonemes: str | None

    def __init__(
        self,
        id: str,
        text: str,
        language: str,
        voice: str | VoiceBlend | None = None,
        pronunciation_overrides: tuple[PronunciationOverride, ...]
        | list[PronunciationOverride] = (),
        annotations: tuple[LinguisticToken, ...] | list[LinguisticToken] | None = None,
        phonemes: str | None = None,
        *,
        tokens: tuple[LinguisticToken, ...] | list[LinguisticToken] | None = None,
    ) -> None:
        if tokens is not None and annotations is not None and tuple(tokens) != tuple(annotations):
            raise InvalidLinguisticTokensError(
                "tokens and legacy annotations cannot provide different values"
            )
        request_tokens = tokens if tokens is not None else annotations or ()
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "voice", voice)
        object.__setattr__(self, "pronunciation_overrides", tuple(pronunciation_overrides))
        object.__setattr__(self, "tokens", tuple(request_tokens))
        object.__setattr__(self, "phonemes", phonemes)
        self.__post_init__()

    @property
    def annotations(self) -> tuple[LinguisticToken, ...]:
        """Deprecated compatibility alias for ``tokens``."""
        return self.tokens

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise InvalidRequestError("synthesis request id must be a non-empty string")
        if not isinstance(self.text, str):
            raise InvalidRequestError("synthesis request text must be a string")
        if not self.text.strip():
            raise EmptyTextError(
                "synthesis text must contain at least one non-whitespace character"
            )
        object.__setattr__(self, "language", _normalize_language(self.language, "language"))
        if self.voice is not None and not isinstance(self.voice, (str, VoiceBlend)):
            raise InvalidVoiceError("voice must be a voice name, VoiceBlend, or None")
        if isinstance(self.voice, str) and not self.voice.strip():
            raise InvalidVoiceError("voice must be non-empty when supplied")
        if self.phonemes is not None and not isinstance(self.phonemes, str):
            raise InvalidPronunciationError("request phonemes must be a string or None")

        overrides = self.pronunciation_overrides
        tokens = self.tokens
        if any(not isinstance(item, PronunciationOverride) for item in overrides):
            raise InvalidPronunciationError(
                "pronunciation_overrides must contain PronunciationOverride values"
            )
        if any(not isinstance(item, LinguisticToken) for item in tokens):
            raise InvalidLinguisticTokensError("tokens must contain LinguisticToken values")

        for index, override in enumerate(overrides):
            _validate_offset(
                override.start,
                override.end,
                len(self.text),
                f"override[{index}]",
                InvalidPronunciationError,
            )
        direct_overrides = sorted(
            (item for item in overrides if item.phonemes is not None), key=lambda item: item.start
        )
        for previous, current in zip(direct_overrides, direct_overrides[1:], strict=False):
            if current.start < previous.end:
                raise InvalidPronunciationError(
                    "overlapping direct phoneme overrides are ambiguous"
                )
        if self.phonemes is not None and direct_overrides:
            raise InvalidPronunciationError(
                "whole-request phonemes cannot be combined with span phoneme overrides"
            )

        previous: LinguisticToken | None = None
        for index, token in enumerate(tokens):
            _validate_offset(
                token.start,
                token.end,
                len(self.text),
                f"token[{index}]",
                InvalidLinguisticTokensError,
            )
            if token.text is not None and self.text[token.start : token.end] != token.text:
                raise InvalidLinguisticTokensError(
                    f"token[{index}] text does not match the source text at its offsets"
                )
            if previous is not None:
                if (token.start, token.end) < (previous.start, previous.end):
                    raise InvalidLinguisticTokensError(
                        "linguistic tokens must be sorted by (start, end)"
                    )
                if token.start < previous.end:
                    raise InvalidLinguisticTokensError(
                        "overlapping linguistic tokens are ambiguous"
                    )
            previous = token


SynthesisSegment = SynthesisRequest


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
    synthesis_identity: SynthesisIdentity | None = None
    voice_level_applications: tuple[VoiceLevelApplication, ...] = ()
    short_sentence_mode: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("rendered segment id must be a non-empty string")
        audio = np.asarray(self.audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("rendered segment audio must be mono (one-dimensional)")
        if audio.size == 0:
            raise ValueError("rendered segment audio must not be empty")
        self.audio = audio
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("rendered segment sample_rate must be an integer")
        if self.sample_rate <= 0:
            raise ValueError("rendered segment sample_rate must be positive")
        if not isinstance(self.text, str):
            raise TypeError("rendered segment text must be a string")
        self.language = _normalize_language(self.language, "rendered segment language")
        self.token_ids = tuple(int(token_id) for token_id in self.token_ids)
        self.word_timings = tuple(self.word_timings)
        self.diagnostics = tuple(self.diagnostics)
        self.voice_level_applications = tuple(self.voice_level_applications)
        if self.short_sentence_mode is not None and not isinstance(self.short_sentence_mode, str):
            raise TypeError("short_sentence_mode must be a string or None")
        previous_char_end = 0
        previous_sample_end = 0
        for timing in self.word_timings:
            if timing.segment_id != self.id:
                raise ValueError("word timing segment_id must match rendered segment id")
            if (
                timing.char_start < 0
                or timing.char_end < timing.char_start
                or timing.char_end > len(self.text)
            ):
                raise ValueError("word timing character bounds must refer to rendered text")
            if self.text[timing.char_start : timing.char_end] != timing.text:
                raise ValueError("word timing text must match rendered text at its offsets")
            if timing.char_start < previous_char_end:
                raise ValueError("word timing character offsets must be monotonic")
            if timing.start_sample < 0 or timing.end_sample < timing.start_sample:
                raise ValueError("word timing sample bounds must be non-negative and ordered")
            if timing.start_sample < previous_sample_end:
                raise ValueError("word timing sample offsets must be monotonic")
            if timing.end_sample > len(self.audio):
                raise ValueError("word timing exceeds rendered audio duration")
            previous_char_end = timing.char_end
            previous_sample_end = timing.end_sample

    def save_wav(self, path: str | Path) -> None:
        """Write this mono waveform as 32-bit floating-point WAV without clipping."""
        import soundfile

        soundfile.write(path, self.audio, self.sample_rate, subtype="FLOAT")

    def play(self, *, device: int | str | None = None) -> None:
        """Play this rendered waveform using the optional playback dependency."""
        from .playback import play_audio

        play_audio(self.audio, self.sample_rate, device=device)
