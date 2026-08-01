"""Tests for structured per-segment prosody diagnostics."""

from __future__ import annotations

import numpy as np

from pykokoro.audio_generator import AudioGenerator
from pykokoro.prosody_config import ProsodyConfig
from pykokoro.short_sentence_handler import SHORT_SENTENCE_META_KEY
from pykokoro.types import PhonemeSegment, Trace


def test_segment_prosody_trace_records_parameters_and_waveform_metrics() -> None:
    time = np.arange(24_000, dtype=np.float32) / 24_000
    source = (0.2 * np.sin(2.0 * np.pi * 180.0 * time)).astype(np.float32)
    segment = PhonemeSegment(
        id="seg0_ph0",
        segment_id="seg0",
        phoneme_id=0,
        text="today",
        phonemes="tədeɪ",
        tokens=[1, 2, 3],
        ssmd_metadata={
            "prosody_rate": "87%",
            "prosody_pitch": "+1.2st",
            SHORT_SENTENCE_META_KEY: {
                "mode": "randomized-phrase",
                "cut_applied": True,
            },
        },
    )
    trace = Trace()
    generator = AudioGenerator.__new__(AudioGenerator)

    rendered = generator._apply_segment_prosody(
        source,
        segment,
        ProsodyConfig(method="wsola", strict=True, fallback_methods=()),
        trace,
    )

    assert rendered.dtype == source.dtype
    assert len(trace.prosody) == 1
    diagnostic = trace.prosody[0]
    assert diagnostic["segment_id"] == "seg0_ph0"
    assert diagnostic["method"] == "wsola"
    assert diagnostic["rate_multiplier"] == 0.87
    assert diagnostic["pitch_semitones"] == 1.2
    assert diagnostic["volume_db"] is None
    assert diagnostic["short_sentence"]["cut_applied"] is True
    assert diagnostic["source"]["samples"] == source.size
    assert diagnostic["output"]["samples"] == round(source.size / 0.87)
    assert diagnostic["source"]["finite"] is True
    assert diagnostic["output"]["finite"] is True
    assert diagnostic["runtime_ms"] >= 0.0


def test_segment_without_prosody_does_not_add_trace_diagnostic() -> None:
    segment = PhonemeSegment(
        id="seg0_ph0",
        segment_id="seg0",
        phoneme_id=0,
        text="plain",
        phonemes="pleɪn",
        tokens=[1],
    )
    trace = Trace()
    generator = AudioGenerator.__new__(AudioGenerator)
    source = np.zeros(32, dtype=np.float32)

    rendered = generator._apply_segment_prosody(source, segment, trace=trace)

    np.testing.assert_array_equal(rendered, source)
    assert trace.prosody == []
