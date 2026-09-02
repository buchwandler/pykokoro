"""Tests for prosody configuration propagation through synthesis adapters."""

from __future__ import annotations

import numpy as np

from pykokoro import PipelineConfig, ProsodyConfig
from pykokoro.audio_generator import AudioGenerator
from pykokoro.generation_config import GenerationConfig
from pykokoro.stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
from pykokoro.stages.synth.onnx import OnnxSynthesizerAdapter
from pykokoro.types import PhonemeSegment, Trace


def _segment() -> PhonemeSegment:
    return PhonemeSegment(
        id="segment-1",
        segment_id="segment-1",
        phoneme_id=0,
        text="speech",
        phonemes="speech",
        tokens=[1],
        ssmd_metadata={"prosody_rate": "87%"},
    )


def test_audio_generator_receives_exact_prosody_config(monkeypatch) -> None:
    import pykokoro.audio_generator as audio_generator_module

    seen = []

    def fake_apply(audio, sample_rate, **kwargs):
        seen.append(kwargs["config"])
        return audio

    monkeypatch.setattr(audio_generator_module, "apply_prosody", fake_apply)
    config = ProsodyConfig(method="esola", strict=True)
    generator = object.__new__(AudioGenerator)

    generator._apply_segment_prosody(
        np.ones(100, dtype=np.float32),
        _segment(),
        config,
    )

    assert seen == [config]


def test_split_postprocessing_adapter_forwards_prosody_config() -> None:
    seen = []

    class FakeKokoro:
        def postprocess_audio_segments(self, segments, trim_silence, prosody_config=None):
            _ = trim_silence
            seen.append(prosody_config)
            return segments

        def concatenate_audio_segments(self, segments):
            _ = segments
            return np.zeros(1, dtype=np.float32)

    config = PipelineConfig(prosody=ProsodyConfig(method="esola", strict=True))
    OnnxAudioPostprocessingAdapter(FakeKokoro()).postprocess([], config, Trace())

    assert seen == [config.prosody]


def test_legacy_synth_adapter_forwards_prosody_config() -> None:
    seen = []

    class FakeKokoro:
        def _resolve_voice_style(self, voice):
            _ = voice
            return np.zeros(1, dtype=np.float32)

        def _generate_from_segments(self, *args, **kwargs):
            _ = args
            seen.append(kwargs["prosody_config"])
            return np.zeros(1, dtype=np.float32)

    config = PipelineConfig(
        generation=GenerationConfig(lang="en-us"),
        prosody=ProsodyConfig(method="td_psola", strict=True),
    )
    OnnxSynthesizerAdapter(FakeKokoro()).synthesize([], config, Trace())

    assert seen == [config.prosody]
