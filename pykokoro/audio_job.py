"""Build AudioCompose jobs from PyKokoro rendered segments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from audiocompose import (
    AudioAnchor,
    AudioBufferSource,
    AudioClip,
    AudioJob,
    AudioSpan,
    LoudnessPolicy,
    OutputPolicy,
    Silence,
)

from .loudness_config import LoudnessConfig
from .types import BoundaryEvent, PhonemeSegment
from .utils import seconds_to_samples


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _quantized_seconds(seconds: float, sample_rate: int) -> float:
    return seconds_to_samples(seconds, sample_rate) / sample_rate


def _validate_segment_word_timings(
    segment: PhonemeSegment,
    waveform_length: int,
) -> None:
    for index, timing in enumerate(segment.word_timings):
        if not (0 <= timing.start_sample <= timing.end_sample <= waveform_length):
            raise ValueError(
                "word timing is not segment-local: "
                f"segment={segment.id!r} word_index={index} text={timing.text!r} "
                f"range={timing.start_sample}:{timing.end_sample} "
                f"waveform_length={waveform_length}"
            )


def _loudness_policy(config: LoudnessConfig | None) -> LoudnessPolicy:
    if config is None:
        return LoudnessPolicy()
    return LoudnessPolicy(
        target_lufs=config.target_lufs,
        true_peak_ceiling_dbtp=config.true_peak_ceiling_dbtp,
        peak_policy=config.peak_policy,
    )


def rendered_segments_to_audio_job(
    segments: Sequence[PhonemeSegment],
    *,
    boundaries: Iterable[BoundaryEvent] = (),
    sample_rate: int = 24000,
    loudness: LoudnessConfig | None = None,
    producer: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
    job_id: str | None = None,
) -> AudioJob:
    """Create one deterministic AudioCompose job from processed phoneme segments."""
    items: list[AudioClip | Silence] = []
    marker_events = tuple(boundary for boundary in boundaries if boundary.kind == "marker")
    used_markers: set[int] = set()

    for segment in segments:
        if segment.pause_before > 0:
            items.append(
                Silence(
                    id=f"{segment.id}:pause-before",
                    seconds=_quantized_seconds(segment.pause_before, sample_rate),
                    metadata={"segment_id": segment.id, "position": "before"},
                )
            )
        audio = segment.processed_audio
        if audio is None:
            audio = segment.raw_audio
        if audio is None:
            continue
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        anchors: list[AudioAnchor] = []
        _validate_segment_word_timings(segment, len(waveform))
        for index, boundary in enumerate(marker_events):
            if index in used_markers:
                continue
            if boundary.pos < segment.char_start:
                continue
            if boundary.pos > segment.char_end:
                break
            offset = 0 if boundary.pos <= segment.char_start else len(waveform)
            marker_name = boundary.attrs.get("marker")
            if marker_name:
                anchors.append(AudioAnchor(f"marker:{index}", offset, marker_name))
                used_markers.add(index)

        spans = [
            AudioSpan(
                source_start=timing.char_start,
                source_end=timing.char_end,
                sample_start=timing.start_sample,
                sample_end=timing.end_sample,
                id=f"word:{segment.id}:{index}",
                metadata={"text": timing.text, "segment_id": timing.segment_id},
            )
            for index, timing in enumerate(segment.word_timings)
        ]
        items.append(
            AudioClip(
                id=segment.id,
                source=AudioBufferSource(waveform, sample_rate),
                anchors=tuple(anchors),
                spans=tuple(spans),
                metadata={
                    "segment_id": segment.segment_id,
                    "text": segment.text,
                    "char_start": segment.char_start,
                    "char_end": segment.char_end,
                    "language": segment.lang,
                    "ssmd": _json_safe(segment.ssmd_metadata or {}),
                },
            )
        )
        if segment.pause_after > 0:
            items.append(
                Silence(
                    id=f"{segment.id}:pause-after",
                    seconds=_quantized_seconds(segment.pause_after, sample_rate),
                    metadata={"segment_id": segment.id, "position": "after"},
                )
            )

    return AudioJob(
        items=tuple(items),
        output=OutputPolicy(sample_rate=sample_rate, loudness=_loudness_policy(loudness)),
        producer=_json_safe(dict(producer or {})),
        source=_json_safe(dict(source or {})),
        job_id=job_id,
    )


def waveform_to_audio_job(
    audio: np.ndarray,
    *,
    sample_rate: int,
    markers: Sequence[Mapping[str, Any]] = (),
    word_timings: Sequence[Any] = (),
    loudness: LoudnessConfig | None = None,
    producer: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
    job_id: str | None = None,
) -> AudioJob:
    """Wrap an already rendered waveform for compatibility paths."""
    anchors = tuple(
        AudioAnchor(
            f"marker:{index}",
            int(marker.get("sample_offset", 0)),
            marker.get("name"),
        )
        for index, marker in enumerate(markers)
    )
    spans = tuple(
        AudioSpan(
            source_start=int(timing.char_start),
            source_end=int(timing.char_end),
            sample_start=int(timing.start_sample),
            sample_end=int(timing.end_sample),
            id=f"word:{index}",
            metadata={"text": timing.text, "segment_id": timing.segment_id},
        )
        for index, timing in enumerate(word_timings)
    )
    return AudioJob(
        items=(
            AudioClip(
                id="pykokoro:rendered",
                source=AudioBufferSource(np.asarray(audio, dtype=np.float32), sample_rate),
                anchors=anchors,
                spans=spans,
            ),
        ),
        output=OutputPolicy(sample_rate=sample_rate, loudness=_loudness_policy(loudness)),
        producer=_json_safe(dict(producer or {})),
        source=_json_safe(dict(source or {})),
        job_id=job_id,
    )


__all__ = ["rendered_segments_to_audio_job", "waveform_to_audio_job"]
