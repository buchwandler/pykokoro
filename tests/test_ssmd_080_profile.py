import numpy as np
import pytest

from pykokoro.audio_generator import resolve_audio_annotation
from pykokoro.exceptions import SSMDDocumentError
from pykokoro.ssmd_config import SSMDRenderConfig
from pykokoro.ssmd_parser import parse_ssmd_document


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


def test_render_config_is_immutable():
    config = SSMDRenderConfig()
    with pytest.raises(AttributeError):
        config.parse_header = False
