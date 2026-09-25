from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pykokoro import SynthesisInputTooLongError
from pykokoro.prepared_g2p import PreparedSynthesis
from pykokoro.request_renderer import OnnxRequestRenderer
from pykokoro.synthesis_config import SynthesisConfig
from pykokoro.synthesis_types import SynthesisSegment
from pykokoro.types import G2PAlignmentToken, PhonemeSegment, Trace, WordTiming


class FakeG2PAdapter:
    def __init__(self) -> None:
        self.context_calls: list[tuple[str, str]] = []

    def phonemize_context(self, text: str, language: str, config: SynthesisConfig) -> object:
        self.context_calls.append((text, language))
        return object()


class FakeBackend:
    def __init__(self) -> None:
        self.generated: list[PhonemeSegment] = []

    def resolve_voice_style(self, voice: str) -> str:
        return voice

    def preprocess_segments(self, segments, enable_short_sentence, **kwargs):
        return segments

    def generate_raw_audio_segments(self, segments, voice_style, speed, **kwargs):
        self.generated = list(segments)
        for segment in segments:
            segment.raw_audio = np.zeros(4, dtype=np.float32)
            segment.word_timings = [
                WordTiming(segment.text, segment.char_start, segment.char_end, 1, 3, segment.id)
            ]
        return segments

    def postprocess_audio_segments(self, segments, *, trim_silence, **kwargs):
        for segment in segments:
            segment.processed_audio = segment.raw_audio
        return segments


class FakeRenderer(OnnxRequestRenderer):
    def __init__(self) -> None:
        self.backend = FakeBackend()
        super().__init__(FakeG2PAdapter(), backend_factory=lambda config: self.backend)


def _config(**kwargs: Any) -> SynthesisConfig:
    return SynthesisConfig(
        voice="af_heart",
        model_source="github",
        model_variant="v1.0",
        model_identity="test-model",
        **kwargs,
    )


def _prepared(text: str, token_count: int) -> PreparedSynthesis:
    return PreparedSynthesis(
        request_id="request",
        text=text,
        language="en-us",
        voice="af_heart",
        phonemes="prepared-phonemes",
        token_ids=tuple(range(token_count)),
        alignment_tokens=(
            G2PAlignmentToken(
                text=text,
                phonemes="prepared-phonemes",
                char_start=0,
                char_end=len(text),
                model_token_count=token_count,
                model_span_token_count=token_count,
            ),
        ),
    )


def _request(text: str) -> SynthesisSegment:
    return SynthesisSegment("request", text, "en-us")


def test_atomic_request_keeps_exact_text_in_one_phoneme_segment() -> None:
    request = _request("The caller prepared this exact text.")
    prepared = _prepared(request.text, 2)
    renderer = FakeRenderer()

    result = renderer.render(prepared, request, _config(return_trace=True))

    assert result.text == request.text
    assert len(renderer.backend.generated) == 1
    segment = renderer.backend.generated[0]
    assert segment.text == request.text
    assert (segment.char_start, segment.char_end) == (0, len(request.text))
    assert segment.tokens == [0, 1]
    assert result.trace is not None
    assert result.trace.model["acoustic_chunk_count"] == 1


def test_capacity_boundary_accepts_exact_limit_and_rejects_limit_plus_one() -> None:
    renderer = FakeRenderer()
    request = _request("Hello")

    exact = renderer._build_phoneme_segments(
        request, _prepared(request.text, 2), _config(), max_tokens=2
    )
    assert len(exact) == 1
    assert exact[0].tokens == [0, 1]

    with pytest.raises(SynthesisInputTooLongError) as raised:
        renderer._build_phoneme_segments(
            request, _prepared(request.text, 3), _config(), max_tokens=2
        )

    error = raised.value
    assert error.text_length == len(request.text)
    assert error.token_count == 3
    assert error.max_tokens == 2
    assert error.model_id == "test-model"


def test_oversized_atomic_request_never_imports_or_uses_phrasplit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import: Callable[..., Any] = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "phrasplit" or name.startswith("phrasplit."):
            raise AssertionError("request rendering must not import PhraseSplit")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    request = _request("First. Second.")

    with pytest.raises(SynthesisInputTooLongError):
        FakeRenderer()._build_phoneme_segments(
            request, _prepared(request.text, 5), _config(), max_tokens=4, trace=Trace()
        )
