from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pykokoro.audio_generator import AudioGenerator
from pykokoro.exceptions import ConfigurationError
from pykokoro.generation_config import GenerationConfig
from pykokoro.prepared_g2p import PreparedSynthesis
from pykokoro.reference_voice import ReferenceVoice
from pykokoro.request_renderer import OnnxRequestRenderer
from pykokoro.synthesis_config import SynthesisConfig
from pykokoro.synthesis_identity import build_synthesis_identity
from pykokoro.synthesis_types import SynthesisSegment
from pykokoro.types import Trace
from pykokoro.voice_level import VoiceLevelConfig

_MODEL_ID = "en-akinvox-cloning-v1"


def _reference_voice(value: float = 0.25) -> ReferenceVoice:
    return ReferenceVoice(
        style=np.full((1, 256), value, dtype=np.float32),
        memory=np.full((1, 4, 192), value, dtype=np.float32),
        memory_mask=np.ones((1, 4), dtype=np.bool_),
        model_id=_MODEL_ID,
        model_fingerprint="onnx-model-fingerprint",
        metadata={
            "reference_audio_sha256": "a" * 64,
            "reference_text_sha256": "b" * 64,
        },
    )


class _Tokenizer:
    def tokenize(self, phonemes: str) -> list[int]:
        return [1, 2, 3]


class _Runtime:
    cache_identity = "reference-runtime"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def infer(
        self,
        token_ids,
        *,
        style=None,
        reference=None,
        speed=1.0,
        seed=None,
    ):
        self.calls.append(
            {
                "token_ids": tuple(token_ids),
                "style": style,
                "reference": reference,
                "speed": speed,
                "seed": seed,
            }
        )
        return SimpleNamespace(
            audio=np.zeros(4000, dtype=np.float32),
            timings=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        )


class _G2P:
    def phonemize_context(self, text: str, language: str, config: SynthesisConfig) -> object:
        return object()


class _ReferenceBackend:
    def __init__(self) -> None:
        self.reference_call: tuple[object, float, int | None] | None = None
        self.postprocess_options: dict[str, object] = {}

    def preprocess_segments(self, segments, enable_short_sentence, **kwargs):
        self.context_phonemizer = kwargs["context_phonemizer"]
        return segments

    def resolve_voice_style(self, voice):
        raise AssertionError("reference rendering must not resolve a static voice style")

    def generate_reference_audio_segments(self, segments, reference_voice, speed, *, seed, trace):
        self.reference_call = (reference_voice, speed, seed)
        for segment in segments:
            segment.raw_audio = np.asarray([0.1, -0.1, 0.2], dtype=np.float32)
        return segments

    def postprocess_audio_segments(self, segments, *, trim_silence, **kwargs):
        self.postprocess_options = {"trim_silence": trim_silence, **kwargs}
        for segment in segments:
            segment.processed_audio = segment.raw_audio
        return segments


def _reference_config(
    *, speed: float = 1.0, voice_level: VoiceLevelConfig | None = None
) -> SynthesisConfig:
    return SynthesisConfig(
        model_source="github",
        model_variant=_MODEL_ID,
        generation=GenerationConfig(random_seed=17, speed=speed),
        return_trace=True,
        voice_level=voice_level or VoiceLevelConfig(),
    )


def test_reference_inference_cache_uses_fingerprint_and_seed_without_style_leakage() -> None:
    runtime = _Runtime()
    generator = AudioGenerator(runtime, _Tokenizer())
    first = _reference_voice()
    other = _reference_voice(0.5)
    trace = Trace()

    generator._run_reference_onnx("abc", first, 1.0, trace, seed=17)
    generator._run_reference_onnx("abc", first, 1.0, seed=17)
    generator._run_reference_onnx("abc", first, 1.0, seed=18)
    generator._run_reference_onnx("abc", other, 1.0, seed=17)

    assert len(runtime.calls) == 3
    assert all(call["style"] is None for call in runtime.calls)
    assert all(call["speed"] == 1.0 for call in runtime.calls)
    assert [call["seed"] for call in runtime.calls] == [17, 18, 17]
    assert all(call["reference"] is first for call in runtime.calls[:2])
    assert runtime.calls[2]["reference"] is other
    assert trace.inference[0]["inputs"].keys() == {"token_ids", "speed"}
    serialized_trace = json.dumps(trace.inference)
    assert "reference_audio_sha256" not in serialized_trace
    assert "reference_text_sha256" not in serialized_trace
    assert "style" not in trace.inference[0]


def test_reference_short_sentence_retry_reuses_the_same_conditioning(monkeypatch) -> None:
    from pykokoro.short_sentence_handler import ShortSentenceApplication

    runtime = _Runtime()
    generator = AudioGenerator(runtime, _Tokenizer(), inference_cache_enabled=False)
    voice = _reference_voice()
    trace = Trace()
    audio, _ = generator._run_reference_onnx("abc", voice, 1.0, trace, seed=7)
    retry_metadata = {
        "kind": "phrase",
        "cutter": "energy-valley",
        "target_start_ts": 1600 / 24000,
        "target_end_ts": 2400 / 24000,
        "previous_token_end_ts": 600 / 24000,
        "next_token_start_ts": 3400 / 24000,
        "has_left_context": True,
        "has_right_context": True,
        "frame_duration_ms": 5,
        "energy_threshold": 0.05,
        "min_silence_seconds": 0.02,
    }
    monkeypatch.setattr(
        "pykokoro.audio_generator.build_short_sentence_phrase_retry",
        lambda *args, **kwargs: ShortSentenceApplication("abc", [1, 2, 3], retry_metadata),
    )
    segment = SimpleNamespace(
        text="Hi!",
        lang="en-us",
        phonemes="abc",
        tokens=[1, 2, 3],
        word_timings=[],
        engine_metadata={
            "__short_sentence": {
                **retry_metadata,
                "target_end_ts": 1600 / 24000,
                "phrase_fallback_templates": ["retry {segment}"],
                "phrase_fallback_tries": 1,
                "fallback_phonemes": "—abc—",
                "fallback_tokens": [1, 2, 3],
            }
        },
    )

    result = generator._prepare_short_sentence_phrase_audio(
        segment,
        audio,
        None,
        1.0,
        trace=trace,
        reference_voice=voice,
        seed=7,
    )

    assert result.size > 0
    assert len(runtime.calls) == 2
    assert all(call["reference"] is voice for call in runtime.calls)
    assert all(call["seed"] == 7 for call in runtime.calls)


def test_reference_renderer_routes_reference_voice_and_keeps_trace_private() -> None:
    voice = _reference_voice()
    request = SynthesisSegment("reference-request", "Hello there.", "en-us", voice=voice)
    prepared = PreparedSynthesis(
        request_id=request.id,
        text=request.text,
        language=request.language,
        voice=None,
        phonemes="həˈloʊ",
        token_ids=(1, 2, 3),
    )
    backend = _ReferenceBackend()
    renderer = OnnxRequestRenderer(_G2P(), backend_factory=lambda config: backend)

    rendered = renderer.render(
        prepared,
        request,
        _reference_config(voice_level=VoiceLevelConfig(gain_db=-1.5)),
    )

    assert backend.reference_call == (voice, 1.0, 17)
    assert rendered.voice == f"reference:{voice.fingerprint}"
    assert rendered.synthesis_identity is not None
    assert rendered.synthesis_identity.voice == f"reference:{voice.fingerprint}"
    assert rendered.voice_level_applications[0].applied is True
    assert rendered.voice_level_applications[0].source == "override"
    assert rendered.trace is not None
    assert rendered.trace.model["voice_mode"] == "reference"
    assert rendered.trace.model["reference_fingerprint"] == voice.fingerprint
    trace_json = json.dumps({"model": rendered.trace.model, "inference": rendered.trace.inference})
    assert "a" * 64 not in trace_json
    assert "b" * 64 not in trace_json
    assert "reference_audio_sha256" not in trace_json
    assert "reference_text_sha256" not in trace_json
    assert "memory" not in trace_json
    renderer.close()


def test_reference_rendering_rejects_auto_calibration_and_non_unit_speed() -> None:
    voice = _reference_voice()
    request = SynthesisSegment("reference-request", "Hello there.", "en-us", voice=voice)
    prepared = PreparedSynthesis(
        request_id=request.id,
        text=request.text,
        language=request.language,
        voice=None,
        phonemes="həˈloʊ",
        token_ids=(1, 2, 3),
    )
    backend = _ReferenceBackend()
    renderer = OnnxRequestRenderer(_G2P(), backend_factory=lambda config: backend)

    with pytest.raises(ConfigurationError, match="automatic voice-level calibration"):
        renderer.render(
            prepared,
            request,
            _reference_config(voice_level=VoiceLevelConfig(mode="calibrated")),
        )
    with pytest.raises(ConfigurationError, match="speed=1.0"):
        renderer.render(prepared, request, _reference_config(speed=1.2))
    assert backend.reference_call is None
    renderer.close()


def test_reference_voice_identity_is_model_bound_and_fingerprint_sensitive() -> None:
    config = SynthesisConfig(model_source="github", model_variant=_MODEL_ID)
    voice = _reference_voice()
    other = _reference_voice(0.5)
    first = build_synthesis_identity(config, language="en-us", voice=voice)
    same = build_synthesis_identity(config, language="en-us", voice=voice)
    different = build_synthesis_identity(config, language="en-us", voice=other)

    assert first.voice == f"reference:{voice.fingerprint}"
    assert first.cache_key == same.cache_key
    assert first.cache_key != different.cache_key
