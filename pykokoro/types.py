"""Request-local engine data structures and timing geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .voice_level import VoiceCalibrationKey


@dataclass(frozen=True, slots=True)
class WordTiming:
    """Model-derived timing for one source-text word or spoken source span."""

    text: str
    char_start: int
    char_end: int
    start_sample: int
    end_sample: int
    segment_id: str
    source: Literal["model_pred_dur"] = "model_pred_dur"

    def start_seconds(self, sample_rate: int) -> float:
        """Return the start offset in seconds for ``sample_rate``."""
        return self.start_sample / sample_rate

    def end_seconds(self, sample_rate: int) -> float:
        """Return the end offset in seconds for ``sample_rate``."""
        return self.end_sample / sample_rate


@dataclass(frozen=True, slots=True)
class G2PAlignmentToken:
    """Normalized KokoroG2P token metadata used for request-local timing alignment."""

    text: str
    phonemes: str
    whitespace: str = ""
    char_start: int | None = None
    char_end: int | None = None
    model_token_count: int | None = None
    model_span_token_count: int | None = None
    pronunciation_source: str | None = None
    pronunciation_provider: str | None = None
    pronunciation_lexicon_id: str | None = None
    pronunciation_requested_language: str | None = None
    pronunciation_source_ipa: str | None = None
    pronunciation_language_markers: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "text": self.text,
            "phonemes": self.phonemes,
            "whitespace": self.whitespace,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "model_token_count": self.model_token_count,
            "pronunciation_source": self.pronunciation_source,
            "pronunciation_provider": self.pronunciation_provider,
            "pronunciation_lexicon_id": self.pronunciation_lexicon_id,
            "pronunciation_requested_language": self.pronunciation_requested_language,
            "pronunciation_source_ipa": self.pronunciation_source_ipa,
            "pronunciation_language_markers": self.pronunciation_language_markers,
        }
        if self.model_span_token_count is not None:
            result["model_span_token_count"] = self.model_span_token_count
        return result


def _exact_timing_geometry(
    token: G2PAlignmentToken | dict[str, Any],
) -> tuple[int, int] | None:
    """Return exact speech and total-span model positions for one alignment item."""
    if isinstance(token, G2PAlignmentToken):
        model_token_count = token.model_token_count
        explicit_span = token.model_span_token_count
        whitespace = token.whitespace
    else:
        model_token_count = token.get("model_token_count")
        explicit_span = token.get("model_span_token_count")
        whitespace = token.get("whitespace") or ""
    if (
        not isinstance(model_token_count, int)
        or isinstance(model_token_count, bool)
        or model_token_count < 0
    ):
        return None
    if explicit_span is not None:
        if (
            not isinstance(explicit_span, int)
            or isinstance(explicit_span, bool)
            or explicit_span < model_token_count
        ):
            return None
        return model_token_count, explicit_span
    return model_token_count, model_token_count + (1 if whitespace else 0)


def _model_span_token_count(token: G2PAlignmentToken | dict[str, Any]) -> int | None:
    """Return model input positions consumed by an alignment item, including whitespace."""
    if isinstance(token, G2PAlignmentToken):
        model_token_count = token.model_token_count
        explicit_span = token.model_span_token_count
        whitespace = token.whitespace
    else:
        model_token_count = token.get("model_token_count")
        explicit_span = token.get("model_span_token_count")
        whitespace = token.get("whitespace") or ""
    if explicit_span is not None:
        if (
            not isinstance(explicit_span, int)
            or isinstance(explicit_span, bool)
            or explicit_span < 0
        ):
            return None
        return explicit_span
    if (
        not isinstance(model_token_count, int)
        or isinstance(model_token_count, bool)
        or model_token_count < 0
    ):
        return None
    return model_token_count + (1 if whitespace else 0)


@dataclass
class PhonemeSegment:
    """One model-ready phoneme chunk belonging to a single synthesis request."""

    id: str
    segment_id: str
    phoneme_id: int
    text: str
    phonemes: str
    tokens: list[int]
    lang: str = "en-us"
    char_start: int = 0
    char_end: int = 0
    engine_metadata: dict[str, Any] | None = field(default=None, repr=False)
    voice_name: str | None = None
    render_voice_key: VoiceCalibrationKey | None = field(default=None, kw_only=True)
    raw_audio: np.ndarray | None = field(default=None, repr=False)
    processed_audio: np.ndarray | None = field(default=None, repr=False)
    alignment_tokens: list[G2PAlignmentToken] = field(
        default_factory=list, repr=False, compare=False, kw_only=True
    )
    word_timings: list[WordTiming] = field(
        default_factory=list, repr=False, compare=False, kw_only=True
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize engine-local phoneme data without document/timeline fields."""
        result: dict[str, Any] = {
            "id": self.id,
            "segment_id": self.segment_id,
            "phoneme_id": self.phoneme_id,
            "text": self.text,
            "phonemes": self.phonemes,
            "tokens": self.tokens,
            "lang": self.lang,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }
        if self.engine_metadata is not None:
            result["engine_metadata"] = self.engine_metadata
        if self.voice_name is not None:
            result["voice_name"] = self.voice_name
        if self.render_voice_key is not None:
            result["render_voice_key"] = str(self.render_voice_key)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhonemeSegment:
        """Deserialize only the current request-local engine representation."""
        return cls(
            id=data["id"],
            segment_id=data["segment_id"],
            phoneme_id=data["phoneme_id"],
            text=data["text"],
            phonemes=data["phonemes"],
            tokens=data["tokens"],
            lang=data.get("lang", "en-us"),
            char_start=data.get("char_start", 0),
            char_end=data.get("char_end", 0),
            engine_metadata=data.get("engine_metadata"),
            voice_name=data.get("voice_name"),
            render_voice_key=(
                None
                if data.get("render_voice_key") is None
                else VoiceCalibrationKey.parse(data["render_voice_key"])
            ),
        )

    def format_readable(self) -> str:
        """Format as human-readable text and phonemes."""
        return f"{self.text} [{self.phonemes}]"


@dataclass(frozen=True)
class TraceEvent:
    stage: str
    name: str
    ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """Structured debugging output and lightweight inference accounting."""

    warnings: list[str] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)
    inference: list[dict[str, Any]] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a named trace counter."""
        self.counters[name] = self.counters.get(name, 0) + amount

    def inference_summary(self) -> dict[str, float | int | None]:
        """Return aggregate acoustic inference metrics without retaining extra arrays."""
        calls = len(self.inference)
        hits = sum(1 for item in self.inference if item.get("cache_hit") is True)
        misses = sum(1 for item in self.inference if item.get("cache_hit") is False)
        runtime_ms = sum(float(item.get("runtime_ms", 0.0)) for item in self.inference)
        audio_seconds = sum(float(item.get("audio_seconds", 0.0)) for item in self.inference)
        summary: dict[str, float | int | None] = {
            "onnx_calls": calls,
            "onnx_cache_hits": hits,
            "onnx_cache_misses": misses,
            "onnx_runtime_ms": runtime_ms,
            "onnx_audio_seconds": audio_seconds,
            "onnx_rtf": runtime_ms / 1000.0 / audio_seconds if audio_seconds else None,
        }
        summary.update(self.counters)
        return summary

    def event_summary(self) -> dict[tuple[str, str], float]:
        """Return aggregate durations grouped by stage and event name."""
        totals: dict[tuple[str, str], float] = {}
        for event in self.events:
            key = (event.stage, event.name)
            totals[key] = totals.get(key, 0.0) + event.ms
        return totals
