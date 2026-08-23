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
    prosody: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AudioResult:
    """Generated audio and its diagnostic metadata.

    The result owns references to its final waveform and per-segment audio arrays.
    Callers may retain or copy ``audio`` before using :meth:`release_audio`, which is
    destructive for this result object. Segment audio may be absent when compact
    retention is enabled; callers that need raw or processed segment waveforms must
    keep ``PipelineConfig.retain_segment_audio=True``.
    """

    audio: np.ndarray
    sample_rate: int
    segments: list[Segment] = field(default_factory=list)
    phoneme_segments: list[PhonemeSegment] = field(default_factory=list)
    trace: Trace | None = None
    document_metadata: dict[str, Any] = field(default_factory=dict)
    markers: list[dict[str, Any]] = field(default_factory=list)

    def release_segment_audio(self) -> None:
        """Release per-segment raw and processed audio arrays.

        This operation is destructive and idempotent. Segment structure, phonemes,
        tokens, metadata, trace data, and markers remain available.
        """
        for segment in self.phoneme_segments:
            segment.raw_audio = None
            segment.processed_audio = None

    def release_audio(self) -> None:
        """Release the final waveform and all per-segment audio arrays.

        This operation is destructive and idempotent. After it returns, ``save_wav``
        and ``play`` have no waveform to consume. Independently held array references
        remain valid because only references owned by this result are replaced.
        """
        dtype = self.audio.dtype if isinstance(self.audio, np.ndarray) else np.dtype(np.float32)
        self.audio = np.empty(0, dtype=dtype)
        self.release_segment_audio()

    def save_wav(self, path: str) -> None:
        import soundfile as sf

        sf.write(path, self.audio, self.sample_rate)

    def play(self, *, device: int | str | None = None) -> None:
        """Play the generated waveform through the system audio output.

        Playback is blocking, requires the optional ``sounddevice`` dependency,
        and does not create an intermediate audio file.
        """
        from .playback import play_audio

        play_audio(self.audio, self.sample_rate, device=device)


AudioUnitKind = Literal["paragraph", "sentence"]


@dataclass(frozen=True, slots=True)
class AudioUnitDescriptor:
    """Stable, lightweight identity and text metadata for one prepared unit."""

    index: int
    paragraph_idx: int
    char_start: int
    char_end: int
    text: str
    text_hash: str
    segment_ids: tuple[str, ...]
    phoneme_segment_ids: tuple[str, ...]
    marker_names: tuple[str, ...] = ()
    unit_kind: AudioUnitKind = "paragraph"
    sentence_idx: int | None = None


@dataclass(slots=True)
class AudioUnitResult:
    """Audio and metadata owned by one prepared render unit.

    ``release_audio`` and ``release_segment_audio`` only clear references owned by
    this result. Arrays retained independently by a caller remain valid.
    """

    descriptor: AudioUnitDescriptor
    audio: np.ndarray
    sample_rate: int
    segments: list[Segment] = field(default_factory=list)
    phoneme_segments: list[PhonemeSegment] = field(default_factory=list)
    markers: list[dict[str, Any]] = field(default_factory=list)
    trace: Trace | None = None
    document_metadata: dict[str, Any] = field(default_factory=dict)

    def release_segment_audio(self) -> None:
        """Destructively release raw and processed arrays for this unit."""
        for segment in self.phoneme_segments:
            segment.raw_audio = None
            segment.processed_audio = None

    def release_audio(self) -> None:
        """Destructively release final and per-segment audio, idempotently."""
        dtype = self.audio.dtype if isinstance(self.audio, np.ndarray) else np.dtype(np.float32)
        self.audio = np.empty(0, dtype=dtype)
        self.release_segment_audio()

    def play(self, *, device: int | str | None = None) -> None:
        """Play this rendered unit through the system audio output."""
        from .playback import play_audio

        play_audio(self.audio, self.sample_rate, device=device)


# Backward compatibility aliases
Annotation = AnnotationSpan
