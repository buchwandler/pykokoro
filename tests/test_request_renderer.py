from __future__ import annotations

from typing import Any

import numpy as np

from pykokoro.generation_config import GenerationConfig
from pykokoro.prepared_g2p import PreparedSynthesis
from pykokoro.request_renderer import OnnxRequestRenderer
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


def test_chunk_split_uses_exact_g2p_alignment_boundaries() -> None:
    token_ids = [10, 11, 12, 13, 14, 15, 16]
    alignments = [
        G2PAlignmentToken("a", "a", char_start=0, char_end=1, model_span_token_count=2),
        G2PAlignmentToken("b", "b", char_start=2, char_end=3, model_span_token_count=2),
        G2PAlignmentToken("c", "c", char_start=4, char_end=5, model_span_token_count=3),
    ]

    groups = OnnxRequestRenderer._split_at_alignment_boundaries(token_ids, alignments, 4)

    assert groups == [
        (token_ids[:4], alignments[:2]),
        (token_ids[4:], alignments[2:]),
    ]
    assert OnnxRequestRenderer._split_at_alignment_boundaries(token_ids, alignments, 2) is None


def test_renderer_stitches_model_sized_chunks_without_losing_local_timings() -> None:
    adapter = FakeG2PAdapter()
    backend = FakeBackend()
    renderer = OnnxRequestRenderer(adapter, backend_factory=lambda config: backend)
    text = "x" * 511
    request = SynthesisSegment("long-request", text, "en-us")
    alignment = tuple(
        G2PAlignmentToken("x", "x", char_start=index, char_end=index + 1, model_span_token_count=1)
        for index in range(len(text))
    )
    prepared = PreparedSynthesis(
        request_id=request.id,
        text=text,
        language=request.language,
        voice="af_heart",
        phonemes=text,
        token_ids=tuple(range(len(text))),
        alignment_tokens=alignment,
    )

    rendered = renderer.render(prepared, request, _config())

    assert [len(segment.tokens) for segment in backend.generated] == [510, 1]
    assert len(rendered.audio) == 6
    assert [timing.start_sample for timing in rendered.word_timings] == [0, 3]
    assert [timing.end_sample for timing in rendered.word_timings] == [2, 5]
    assert all(timing.segment_id == request.id for timing in rendered.word_timings)
    assert [len(ids) for ids, _ in adapter.decoded] == [510, 1]
