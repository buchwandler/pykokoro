import numpy as np
import pytest
from audiosig import InvalidParameterError

import pykokoro.audio_generator as audio_generator
from pykokoro.audio_generator import resolve_audio_annotation
from pykokoro.exceptions import SSMDDocumentError
from pykokoro.ssmd_config import SSMDRenderConfig
from pykokoro.ssmd_parser import parse_ssmd_document
from pykokoro.stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
from pykokoro.types import PhonemeSegment, Trace


def test_markers_are_preserved_with_clean_text_offsets():
    parsed = parse_ssmd_document("@start Hello @finish")
    assert parsed.segments[0].text == "Hello"
    assert parsed.segments[0].metadata.markers_before == ["start"]
    assert parsed.segments[0].metadata.markers_after == ["finish"]


def test_extensions_are_rejected_for_kokoro_profile():
    with pytest.raises(SSMDDocumentError) as exc:
        parse_ssmd_document("---\nextensions:\n  x: y\n---\nHello")
    assert exc.value.code == "header.extensions_unsupported"


def test_audio_resolver_applies_clip_speed_repeat_and_resampling():
    def resolve(_source):
        return np.arange(100, dtype=np.float32), 100

    audio = resolve_audio_annotation(
        {
            "audio_src": "clip.wav",
            "audio_clip_begin": "100ms",
            "audio_clip_end": "500ms",
            "audio_speed": "200%",
            "audio_repeat_count": "2",
        },
        resolve,
        sample_rate=200,
    )
    assert audio.dtype == np.float32
    assert len(audio) == 80


def test_audio_resolver_delegates_numeric_transforms(monkeypatch):
    source = np.arange(10, dtype=np.float32)
    calls = []

    def fake_speed(audio, factor):
        calls.append(("speed", factor))
        return audio

    def fake_gain(audio, db):
        calls.append(("gain", db))
        return audio

    def fake_resample(audio, *, source_rate, target_rate):
        calls.append(("resample", source_rate, target_rate))
        return audio

    monkeypatch.setattr(audio_generator, "resample_speed", fake_speed)
    monkeypatch.setattr(audio_generator, "apply_gain_db", fake_gain)
    monkeypatch.setattr(audio_generator, "resample", fake_resample)

    resolve_audio_annotation(
        {
            "audio_src": "clip.wav",
            "audio_speed": "200%",
            "audio_sound_level": "+3dB",
        },
        lambda _source: (source, 100),
        sample_rate=200,
    )

    assert calls == [("speed", 2.0), ("gain", 3.0), ("resample", 100, 200)]


def test_audio_resolver_preserves_empty_clip():
    source = np.arange(10, dtype=np.float32)
    result = resolve_audio_annotation(
        {
            "audio_src": "clip.wav",
            "audio_clip_begin": "100ms",
            "audio_clip_end": "100ms",
            "audio_speed": "200%",
        },
        lambda _source: (source, 100),
        sample_rate=200,
    )
    assert result.shape == (0,)
    assert result.dtype == np.float32


def test_audio_signal_error_uses_ssmd_fallback(monkeypatch):
    import pykokoro.stages.audio_postprocessing.onnx as onnx_postprocessing

    def fail(*args, **kwargs):
        raise InvalidParameterError("forced AudioSig failure")

    monkeypatch.setattr(onnx_postprocessing, "resolve_audio_annotation", fail)
    segment = PhonemeSegment(
        id="1",
        segment_id="1",
        phoneme_id=0,
        text="audio",
        phonemes="audio",
        tokens=[],
        ssmd_metadata={"audio_src": "clip.wav"},
    )

    class FakeKokoro:
        def postprocess_audio_segments(self, segments, trim_silence):
            return segments

        def concatenate_audio_segments(self, segments):
            return np.zeros(1, dtype=np.float32)

    cfg = type("Config", (), {
        "generation": type("Generation", (), {"pause_mode": "none"})(),
        "ssmd": SSMDRenderConfig(audio_source_resolver=lambda _source: None),
    })()
    trace = Trace()

    OnnxAudioPostprocessingAdapter(FakeKokoro()).postprocess([segment], cfg, trace)

    assert trace.warnings == ["ssmd.audio_fallback: forced AudioSig failure"]


def test_render_config_is_immutable():
    config = SSMDRenderConfig()
    with pytest.raises(AttributeError):
        config.parse_header = False
