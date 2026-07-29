from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class AnnotationSpan:
    """Span-based markup annotation (character offsets refer to clean_text)."""

    char_start: int
    char_end: int
    attrs: dict[str, str]


@dataclass(frozen=True)
class BoundaryEvent:
    """Boundary event for SSMD breaks or markers."""

    pos: int
    kind: Literal["pause", "marker"]
    duration_s: float | None = None
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Segment:
    """A chunk of input text with stable offsets into the document."""

    id: str
    text: str
    char_start: int
    char_end: int
    meta: dict[str, Any] = field(default_factory=dict)
    paragraph_idx: int | None = None
    sentence_idx: int | None = None
    clause_idx: int | None = None


@dataclass
class PhonemeSegment:
    """A segment of text with its phoneme representation.

    Each PhonemeSegment references the originating Segment via segment_id and can
    represent a split portion of a longer segment via phoneme_id.
    """

    id: str
    segment_id: str
    phoneme_id: int
    text: str
    phonemes: str
    tokens: list[int]
    lang: str = "en-us"
    char_start: int = 0
    char_end: int = 0
    paragraph_idx: int | None = None
    sentence_idx: int | None = None
    clause_idx: int | None = None
    pause_before: float = 0.0
    pause_after: float = 0.0
    ssmd_metadata: dict[str, Any] | None = field(default=None, repr=False)
    voice_name: str | None = None
    voice_language: str | None = None
    voice_gender: str | None = None
    voice_variant: str | None = None
    raw_audio: np.ndarray | None = field(default=None, repr=False)
    processed_audio: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "segment_id": self.segment_id,
            "phoneme_id": self.phoneme_id,
            "text": self.text,
            "phonemes": self.phonemes,
            "tokens": self.tokens,
            "lang": self.lang,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "paragraph_idx": self.paragraph_idx,
            "sentence_idx": self.sentence_idx,
            "clause_idx": self.clause_idx,
            "pause_before": self.pause_before,
            "pause_after": self.pause_after,
        }
        if self.ssmd_metadata is not None:
            result["ssmd_metadata"] = self.ssmd_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhonemeSegment:
        """Create from dictionary."""
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
            paragraph_idx=data.get("paragraph_idx"),
            sentence_idx=data.get("sentence_idx"),
            clause_idx=data.get("clause_idx"),
            pause_before=data.get("pause_before", 0.0),
            pause_after=data.get("pause_after", 0.0),
            ssmd_metadata=data.get("ssmd_metadata"),
        )

    def format_readable(self) -> str:
        """Format as human-readable string: text [phonemes]."""
        return f"{self.text} [{self.phonemes}]"


@dataclass(frozen=True)
class TraceEvent:
    stage: str
    name: str
    ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """Structured debugging output."""

    warnings: list[str] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)


@dataclass
class AudioResult:
    audio: np.ndarray
    sample_rate: int
    segments: list[Segment] = field(default_factory=list)
    phoneme_segments: list[PhonemeSegment] = field(default_factory=list)
    trace: Trace | None = None
    document_metadata: dict[str, Any] = field(default_factory=dict)
    markers: list[dict[str, Any]] = field(default_factory=list)

    def save_wav(self, path: str) -> None:
        import soundfile as sf

        sf.write(path, self.audio, self.sample_rate)

    def play(self) -> None:
        """Play audio in a Jupyter notebook."""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice is required for audio playback") from None
        sd.play(self.audio, self.sample_rate)
        sd.wait()


# Backward compatibility aliases
Annotation = AnnotationSpan
