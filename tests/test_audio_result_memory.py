from __future__ import annotations

import numpy as np

from pykokoro.types import AudioResult, PhonemeSegment, Segment, Trace, TraceEvent


def _segment(segment_id: str, raw_audio: np.ndarray, processed_audio: np.ndarray) -> PhonemeSegment:
    return PhonemeSegment(
        id=f"{segment_id}_ph0",
        segment_id=segment_id,
        phoneme_id=0,
        text=segment_id,
        phonemes="a",
        tokens=[1, 2],
        char_start=0,
        char_end=len(segment_id),
        ssmd_metadata={"short_sentence": {"enabled": True}},
        raw_audio=raw_audio,
        processed_audio=processed_audio,
    )


def _result() -> tuple[AudioResult, np.ndarray, dict[str, object], list[dict[str, object]]]:
    shared = np.arange(4, dtype=np.float32)
    view_source = np.arange(8, dtype=np.float32)
    view = view_source[2:6]
    independent = np.arange(3, dtype=np.float32)
    metadata = {"title": "Memory test", "pause_defaults": {"sentence": 0.2}}
    markers = [{"name": "middle", "char_offset": 3, "sample_offset": 2}]
    result = AudioResult(
        audio=np.arange(10, dtype=np.float32),
        sample_rate=24_000,
        segments=[Segment(id="seg0", text="seg0", char_start=0, char_end=4)],
        phoneme_segments=[
            _segment("alias", shared, shared),
            _segment("view", view_source, view),
            _segment("independent", np.ones(3, dtype=np.float32), independent),
        ],
        trace=Trace(
            warnings=["warning"],
            events=[TraceEvent(stage="test", name="memory", ms=1.0)],
        ),
        document_metadata=metadata,
        markers=markers,
    )
    return result, metadata, markers, shared


def test_release_segment_audio_preserves_audio_and_metadata() -> None:
    result, metadata, markers, _ = _result()
    original_audio = result.audio.copy()
    original_segments = result.segments.copy()
    original_trace = result.trace
    original_phoneme_metadata = result.phoneme_segments[0].ssmd_metadata

    result.release_segment_audio()

    np.testing.assert_array_equal(result.audio, original_audio)
    assert result.sample_rate == 24_000
    assert result.segments == original_segments
    assert result.document_metadata == metadata
    assert result.markers == markers
    assert result.trace is original_trace
    assert result.phoneme_segments[0].ssmd_metadata == original_phoneme_metadata
    assert all(segment.raw_audio is None for segment in result.phoneme_segments)
    assert all(segment.processed_audio is None for segment in result.phoneme_segments)


def test_release_audio_is_idempotent_and_preserves_external_audio_reference() -> None:
    result, metadata, markers, _ = _result()
    external = result.audio
    original_dtype = result.audio.dtype

    result.release_audio()
    result.release_audio()

    assert result.audio.size == 0
    assert result.audio.dtype == original_dtype
    assert result.sample_rate == 24_000
    assert result.document_metadata == metadata
    assert result.markers == markers
    np.testing.assert_array_equal(external, np.arange(10, dtype=np.float32))
    assert all(segment.raw_audio is None for segment in result.phoneme_segments)
    assert all(segment.processed_audio is None for segment in result.phoneme_segments)
