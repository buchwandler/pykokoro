from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pykokoro.ssmd_config import SSMDRenderConfig
from pykokoro.stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
from pykokoro.types import PhonemeSegment, Trace


class _FakeKokoro:
    def postprocess_audio_segments(self, segments, trim_silence, prosody_config=None):
        _ = trim_silence, prosody_config
        return segments

    def concatenate_audio_segments(self, segments):
        _ = segments
        return np.zeros(1, dtype=np.float32)


def _segment() -> PhonemeSegment:
    return PhonemeSegment(
        id="1",
        segment_id="1",
        phoneme_id=0,
        text="important",
        phonemes="test",
        tokens=[],
        ssmd_metadata={"emphasis": "strong"},
    )


def test_ssmd_emphasis_defaults_to_plain() -> None:
    assert SSMDRenderConfig().emphasis_mode == "plain"


def test_audio_postprocessing_does_not_apply_policy_late() -> None:
    segment = _segment()
    trace = Trace()
    adapter = OnnxAudioPostprocessingAdapter(_FakeKokoro())
    cfg = SimpleNamespace(
        ssmd=SSMDRenderConfig(emphasis_mode="approximate"),
        generation=SimpleNamespace(pause_mode="none"),
    )

    adapter.postprocess([segment], cfg, trace)

    assert segment.ssmd_metadata == {"emphasis": "strong"}
    assert trace.warnings == []
