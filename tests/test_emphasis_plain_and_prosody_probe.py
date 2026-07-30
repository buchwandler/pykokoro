from __future__ import annotations

import builtins
from types import SimpleNamespace

import numpy as np

from pykokoro import prosody
from pykokoro.ssmd_config import SSMDRenderConfig
from pykokoro.stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
from pykokoro.types import PhonemeSegment, Trace


class _FakeKokoro:
    def postprocess_audio_segments(self, segments, trim_silence):
        return segments

    def concatenate_audio_segments(self, segments):
        return np.zeros(1, dtype=np.float32)


def _segment() -> PhonemeSegment:
    segment = PhonemeSegment(
        id="1",
        segment_id="1",
        phoneme_id=0,
        text="important",
        phonemes="test",
        tokens=[],
        ssmd_metadata={"emphasis": "strong"},
    )
    segment.raw_audio = np.ones(8, dtype=np.float32)
    return segment


def test_ssmd_emphasis_defaults_to_plain() -> None:
    assert SSMDRenderConfig().emphasis_mode == "plain"


def test_plain_emphasis_leaves_metadata_unmodified_without_warning() -> None:
    segment = _segment()
    trace = Trace()
    adapter = OnnxAudioPostprocessingAdapter(_FakeKokoro())

    cfg = SimpleNamespace(ssmd=SSMDRenderConfig(), generation=SimpleNamespace(pause_mode="none"))
    adapter.postprocess([segment], cfg, trace)

    assert segment.ssmd_metadata == {"emphasis": "strong"}
    assert trace.warnings == []


def test_approximate_emphasis_remains_explicitly_available() -> None:
    segment = _segment()
    trace = Trace()
    adapter = OnnxAudioPostprocessingAdapter(_FakeKokoro())
    cfg = SimpleNamespace(
        ssmd=SSMDRenderConfig(emphasis_mode="approximate"),
        generation=SimpleNamespace(pause_mode="none"),
    )

    adapter.postprocess([segment], cfg, trace)

    assert segment.ssmd_metadata["prosody_volume"] == "loud"
    assert segment.ssmd_metadata["prosody_rate"] == "fast"


def test_librosa_probe_rejects_partial_lazy_install(monkeypatch) -> None:
    class PartialLibrosa:
        stft = object()
        istft = object()
        resample = object()

        @property
        def phase_vocoder(self):
            raise ModuleNotFoundError("No module named 'sklearn'")

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "librosa":
            return PartialLibrosa()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    backend, error = prosody._load_librosa_backend()

    assert backend is None
    assert isinstance(error, ModuleNotFoundError)


def test_rate_fallback_does_not_touch_librosa_effects(monkeypatch) -> None:
    class FakeLibrosa:
        @property
        def effects(self):
            raise AssertionError("librosa.effects must not be used")

        @staticmethod
        def stft(audio):
            return np.asarray(audio)

        @staticmethod
        def phase_vocoder(spectrum, *, rate):
            return spectrum

        @staticmethod
        def istft(spectrum, *, length):
            return np.resize(np.asarray(spectrum), length)

        @staticmethod
        def resample(audio, **kwargs):
            return np.asarray(audio)

    monkeypatch.setattr(prosody, "AUDIOMENTATIONS_AVAILABLE", False)
    monkeypatch.setattr(prosody, "LIBROSA_AVAILABLE", True)
    monkeypatch.setattr(prosody, "librosa", FakeLibrosa())
    audio = np.arange(100, dtype=np.float32)

    result = prosody.apply_rate(audio, "fast")

    assert len(result) == 80
    assert result.dtype == audio.dtype


def test_runtime_optional_backend_failure_is_fail_open(monkeypatch, caplog) -> None:
    monkeypatch.setattr(prosody, "AUDIOMENTATIONS_AVAILABLE", False)
    monkeypatch.setattr(prosody, "LIBROSA_AVAILABLE", True)
    monkeypatch.setattr(prosody, "librosa", object())
    audio = np.arange(100, dtype=np.float32)

    result = prosody.apply_rate(audio, "fast")

    np.testing.assert_array_equal(result, audio)
    assert "Rate adjustment failed" in caplog.text
