from __future__ import annotations

import numpy as np
import pytest

from pykokoro.exceptions import ConfigurationError
from pykokoro.prepared_g2p import PreparedSynthesis
from pykokoro.request_renderer import OnnxRequestRenderer
from pykokoro.synthesis_config import SynthesisConfig, resolve_synthesis_config
from pykokoro.synthesis_types import SynthesisSegment
from pykokoro.types import PhonemeSegment
from pykokoro.voice_level import VoiceLevelConfig
from pykokoro.voice_pack import KokoroVoicePack


class FakeG2P:
    def phonemize_context(self, text: str, language: str, config: SynthesisConfig) -> object:
        return object()

    def ids_to_phonemes(self, token_ids: list[int] | tuple[int, ...], target_model: str) -> str:
        return "həˈloʊ"


class FakeBackend:
    def __init__(self, pack: KokoroVoicePack) -> None:
        self.pack = pack
        self.resolve_voice_style_called = False
        self.received_style: np.ndarray | None = None
        self.generated: list[PhonemeSegment] = []

    def preprocess_segments(self, segments, enable_short_sentence, **kwargs):
        return segments

    def resolve_voice_style(self, voice: object) -> np.ndarray:
        self.resolve_voice_style_called = True
        raise AssertionError("KokoroVoicePack must bypass voice-name resolution")

    def generate_raw_audio_segments(self, segments, voice_style, speed, **kwargs):
        self.received_style = voice_style
        self.generated = segments
        assert voice_style is self.pack.data
        assert kwargs["default_voice_name"] is None
        for segment in segments:
            segment.raw_audio = np.asarray([0.1, -0.1], dtype=np.float32)
        return segments

    def postprocess_audio_segments(self, segments, *, trim_silence, **kwargs):
        for segment in segments:
            segment.processed_audio = segment.raw_audio
        return segments


def make_pack(value: float = 0.0) -> KokoroVoicePack:
    data = np.full((510, 1, 256), value, dtype=np.float32)
    return KokoroVoicePack.from_array(data, engine="inno-v0.2")


def test_voice_pack_uses_normal_static_synthesis_without_voice_manager() -> None:
    pack = make_pack()
    request = SynthesisSegment("inno", "Hello", "en-us", voice=pack)
    prepared = PreparedSynthesis(
        request_id=request.id,
        text=request.text,
        language=request.language,
        voice=pack,
        phonemes="həˈloʊ",
        token_ids=(1, 2, 3),
    )
    config = SynthesisConfig(
        voice=pack,
        model_source="github",
        model_variant="v1.0",
        return_trace=True,
    )
    backend = FakeBackend(pack)
    renderer = OnnxRequestRenderer(FakeG2P(), backend_factory=lambda _: backend)

    rendered = renderer.render(prepared, request, config)

    assert backend.resolve_voice_style_called is False
    assert backend.received_style is pack.data
    assert rendered.voice == f"voicepack:{pack.fingerprint}"
    assert rendered.synthesis_identity is not None
    assert rendered.synthesis_identity.voice == f"voicepack:{pack.fingerprint}"
    assert rendered.synthesis_identity.cache_key
    assert rendered.trace is not None
    assert rendered.trace.model["voicepack_fingerprint"] == pack.fingerprint


def test_voice_pack_fingerprint_changes_synthesis_cache_identity() -> None:
    first = make_pack(0.0)
    second = make_pack(0.25)

    assert first.fingerprint != second.fingerprint
    assert first.fingerprint == make_pack(0.0).fingerprint
    request = SynthesisSegment("pack", "Hello", "en-us", voice=first)
    first_prepared = PreparedSynthesis(
        request_id=request.id,
        text=request.text,
        language=request.language,
        voice=first,
        phonemes="hello",
        token_ids=(1, 2),
    )
    second_request = SynthesisSegment("pack", "Hello", "en-us", voice=second)
    second_prepared = PreparedSynthesis(
        request_id=second_request.id,
        text=second_request.text,
        language=second_request.language,
        voice=second,
        phonemes="hello",
        token_ids=(1, 2),
    )
    renderer = OnnxRequestRenderer(FakeG2P(), backend_factory=lambda _: FakeBackend(first))
    first_result = renderer.render(
        first_prepared,
        request,
        SynthesisConfig(voice=first, model_variant="v1.0"),
    )
    second_renderer = OnnxRequestRenderer(FakeG2P(), backend_factory=lambda _: FakeBackend(second))
    second_result = second_renderer.render(
        second_prepared,
        second_request,
        SynthesisConfig(voice=second, model_variant="v1.0"),
    )

    assert first_result.synthesis_identity is not None
    assert second_result.synthesis_identity is not None
    assert first_result.synthesis_identity.cache_key != second_result.synthesis_identity.cache_key


def test_voice_pack_disables_automatic_but_allows_explicit_calibration() -> None:
    pack = make_pack()
    with pytest.raises(ConfigurationError, match="unavailable for reference voices or voice packs"):
        resolve_synthesis_config(
            SynthesisConfig(voice=pack, voice_level=VoiceLevelConfig(mode="calibrated")),
            language="en-us",
        )

    configured = resolve_synthesis_config(
        SynthesisConfig(
            voice=pack,
            voice_level=VoiceLevelConfig(mode="calibrated", gain_db=-2.0),
        ),
        language="en-us",
    )
    assert configured.voice is pack
