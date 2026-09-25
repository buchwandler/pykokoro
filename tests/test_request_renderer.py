from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pykokoro import SynthesisInputTooLongError
from pykokoro.exceptions import InvalidVoiceError
from pykokoro.generation_config import GenerationConfig
from pykokoro.prepared_g2p import PreparedSynthesis
from pykokoro.request_renderer import OnnxRequestRenderer
from pykokoro.short_sentence_handler import ShortSentenceConfig
from pykokoro.synthesis_config import SynthesisConfig
from pykokoro.synthesis_types import SynthesisSegment
from pykokoro.types import G2PAlignmentToken, PhonemeSegment, WordTiming
from pykokoro.voice_level import VoiceLevelConfig


class FakeG2PAdapter:
    def __init__(self) -> None:
        self.context_calls: list[tuple[str, str]] = []
        self.decoded: list[tuple[tuple[int, ...], str]] = []

    def phonemize_context(self, text: str, language: str, config: SynthesisConfig) -> object:
        self.context_calls.append((text, language))
        return object()

    def ids_to_phonemes(self, token_ids: list[int] | tuple[int, ...], target_model: str) -> str:
        self.decoded.append((tuple(token_ids), target_model))
        return f"phonemes:{','.join(str(token_id) for token_id in token_ids)}"


class FakeBackend:
    def __init__(self) -> None:
        self.closed = False
        self.preprocessed: list[PhonemeSegment] = []
        self.generated: list[PhonemeSegment] = []
        self.postprocess_options: dict[str, object] = {}

    def resolve_voice_style(self, voice: str) -> str:
        return f"style:{voice}"

    def preprocess_segments(self, segments, enable_short_sentence, **kwargs):
        self.preprocessed = segments
        self.context_phonemizer = kwargs["context_phonemizer"]
        return segments

    def generate_raw_audio_segments(self, segments, voice_style, speed, **kwargs):
        self.generated = segments
        voice_name = kwargs["default_voice_name"]
        assert voice_style == f"style:{voice_name}"
        assert speed == 1.0
        for segment in segments:
            segment.raw_audio = np.asarray([0.25, -0.25, 0.5], dtype=np.float32)
            segment.word_timings = [WordTiming("Hello", 0, 5, 0, 2, segment.id)]
        return segments

    def postprocess_audio_segments(self, segments, *, trim_silence, **kwargs):
        self.postprocess_options = {"trim_silence": trim_silence, **kwargs}
        for segment in segments:
            segment.processed_audio = segment.raw_audio
        return segments

    def close(self) -> None:
        self.closed = True


def _config(**kwargs: Any) -> SynthesisConfig:
    return SynthesisConfig(
        voice="af_heart",
        model_source="github",
        model_variant="v1.0",
        generation=GenerationConfig(lang="en-us"),
        **kwargs,
    )


def test_renderer_keeps_one_request_independent_and_rebases_timings() -> None:
    adapter = FakeG2PAdapter()
    backend = FakeBackend()
    factory_configs: list[SynthesisConfig] = []
    backend = FakeBackend()

    def backend_factory(config: SynthesisConfig) -> FakeBackend:
        factory_configs.append(config)
        return backend

    renderer = OnnxRequestRenderer(adapter, backend_factory=backend_factory)
    request = SynthesisSegment("request-1", "Hello", "en-us", voice="af_sarah")
    prepared = PreparedSynthesis(
        request_id=request.id,
        text=request.text,
        language=request.language,
        voice="af_sarah",
        phonemes="həlˈoʊ",
        token_ids=(1, 2, 3),
        alignment_tokens=(G2PAlignmentToken("Hello", "həlˈoʊ", char_start=0, char_end=5),),
        diagnostics=("frontend note",),
    )

    rendered = renderer.render(
        prepared,
        request,
        _config(
            return_trace=True,
            voice_level=VoiceLevelConfig(mode="calibrated", gain_db=-2.5),
            waveform_validation="strict",
        ),
    )

    assert rendered.id == request.id
    assert rendered.audio.dtype == np.float32
    assert rendered.audio.tolist() == [0.25, -0.25, 0.5]
    assert rendered.sample_rate == 24_000
    assert rendered.voice == "af_sarah"
    assert rendered.token_ids == (1, 2, 3)
    assert rendered.word_timings == (WordTiming("Hello", 0, 5, 0, 2, request.id),)
    assert rendered.diagnostics == ("frontend note",)
    assert rendered.synthesis_identity is not None
    assert rendered.synthesis_identity.voice == "af_sarah"
    assert rendered.synthesis_identity.language == "en-us"
    assert rendered.synthesis_identity.voice_level_gain_db == -2.5
    assert len(rendered.synthesis_identity.cache_key) == 64
    assert len(rendered.voice_level_applications) == 1
    assert rendered.voice_level_applications[0].applied is True
    assert rendered.voice_level_applications[0].gain_db == -2.5
    assert rendered.voice_level_applications[0].source == "override"
    assert rendered.short_sentence_mode is None
    assert rendered.trace is not None
    assert rendered.trace.warnings == ["frontend note"]
    assert backend.preprocessed[0].tokens == [1, 2, 3]
    assert factory_configs[0].voice == "af_sarah"
    assert factory_configs[0].waveform_validation == "strict"
    assert backend.postprocess_options["trim_silence"] is False
    voice_level_config = backend.postprocess_options["voice_level_config"]
    assert isinstance(voice_level_config, VoiceLevelConfig)
    assert voice_level_config.mode == "calibrated"
    assert voice_level_config.gain_db == -2.5
    assert backend.context_phonemizer("context", "en-us") is not None
    assert adapter.context_calls == [("context", "en-us")]

    renderer.close()
    assert backend.closed


def test_renderer_rejects_oversized_request_before_model_inference() -> None:
    adapter = FakeG2PAdapter()
    backend = FakeBackend()
    renderer = OnnxRequestRenderer(adapter, backend_factory=lambda config: backend)
    text = "x" * 511
    request = SynthesisSegment("long-request", text, "en-us")
    prepared = PreparedSynthesis(
        request_id=request.id,
        text=request.text,
        language=request.language,
        voice="af_heart",
        phonemes=text,
        token_ids=tuple(range(511)),
    )

    with pytest.raises(SynthesisInputTooLongError) as raised:
        renderer.render(prepared, request, _config())

    assert raised.value.text_length == len(text)
    assert raised.value.token_count == 511
    assert raised.value.max_tokens == 510
    assert backend.generated == []


def test_renderer_reports_invalid_voice_as_typed_error() -> None:
    class MissingVoiceBackend(FakeBackend):
        def resolve_voice_style(self, voice: str) -> str:
            raise KeyError(voice)

    request = SynthesisSegment("bad-voice", "Hello", "en-us", voice="not-a-voice")
    prepared = PreparedSynthesis(
        request_id=request.id,
        text=request.text,
        language=request.language,
        voice=request.voice,
        phonemes="hello",
        token_ids=(1, 2),
    )
    renderer = OnnxRequestRenderer(
        FakeG2PAdapter(), backend_factory=lambda config: MissingVoiceBackend()
    )

    with pytest.raises(InvalidVoiceError, match="not-a-voice"):
        renderer.render(prepared, request, _config())


def test_renderer_checks_capacity_after_frontend_postprocessing() -> None:
    class ExpandingBackend(FakeBackend):
        def preprocess_segments(self, segments, enable_short_sentence, **kwargs):
            result = super().preprocess_segments(segments, enable_short_sentence, **kwargs)
            result[0].tokens.append(511)
            return result

    backend = ExpandingBackend()
    renderer = OnnxRequestRenderer(FakeG2PAdapter(), backend_factory=lambda config: backend)
    request = SynthesisSegment("expanded", "Hello", "en-us")
    prepared = PreparedSynthesis(
        request_id=request.id,
        text=request.text,
        language=request.language,
        voice="af_heart",
        phonemes="hello",
        token_ids=tuple(range(510)),
    )

    with pytest.raises(SynthesisInputTooLongError) as raised:
        renderer.render(prepared, request, _config())

    assert raised.value.token_count == 511
    assert raised.value.max_tokens == 510
    assert backend.generated == []


def test_rendered_result_exposes_short_sentence_mode_without_context_text() -> None:
    class WrappedBackend(FakeBackend):
        def preprocess_segments(self, segments, enable_short_sentence, **kwargs):
            result = super().preprocess_segments(segments, enable_short_sentence, **kwargs)
            result[0].engine_metadata = {
                "__short_sentence": {
                    "kind": "wrap",
                    "phrase_template": "synthetic context around {segment}",
                }
            }
            return result

    request = SynthesisSegment("wrapped", "Hello", "en-us")
    prepared = PreparedSynthesis(
        request_id=request.id,
        text=request.text,
        language=request.language,
        voice="af_heart",
        phonemes="həlˈoʊ",
        token_ids=(1, 2, 3),
    )
    renderer = OnnxRequestRenderer(
        FakeG2PAdapter(), backend_factory=lambda config: WrappedBackend()
    )
    result = renderer.render(
        prepared,
        request,
        _config(short_sentence_config=ShortSentenceConfig(resolve_mode="wrap")),
    )

    assert result.text == request.text
    assert result.phonemes == prepared.phonemes
    assert result.short_sentence_mode == "wrap"
    assert result.synthesis_identity is not None
    assert result.synthesis_identity.resolved_short_sentence_mode == "wrap"
    assert "synthetic context" not in result.synthesis_identity.short_sentence
