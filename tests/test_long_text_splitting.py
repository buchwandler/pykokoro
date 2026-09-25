from __future__ import annotations

import builtins
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pykokoro import SynthesisInputTooLongError
from pykokoro.prepared_g2p import PreparedSynthesis
from pykokoro.request_renderer import OnnxRequestRenderer
from pykokoro.synthesis_config import SynthesisConfig
from pykokoro.synthesis_types import LinguisticToken, PronunciationOverride, SynthesisSegment
from pykokoro.types import G2PAlignmentToken, PhonemeSegment, Trace, WordTiming


class FakeG2PAdapter:
    def __init__(self, token_counter: Callable[[str], int] | None = None) -> None:
        self.context_calls: list[tuple[str, str]] = []
        self.phonemized_segments: list[SynthesisSegment] = []
        self._token_counter = token_counter or (lambda text: len(text.split()))

    def phonemize_context(self, text: str, language: str, config: SynthesisConfig) -> object:
        self.context_calls.append((text, language))
        return object()

    def phonemize(self, segment: SynthesisSegment, config: SynthesisConfig) -> PreparedSynthesis:
        self.phonemized_segments.append(segment)
        token_count = self._token_counter(segment.text)
        alignments = (
            (
                G2PAlignmentToken(
                    text=segment.text,
                    phonemes=f"phonemes:{segment.text}",
                    char_start=0,
                    char_end=len(segment.text),
                    model_token_count=token_count,
                    model_span_token_count=token_count,
                ),
            )
            if token_count
            else ()
        )
        return PreparedSynthesis(
            request_id=segment.id,
            text=segment.text,
            language=segment.language,
            voice=segment.voice if isinstance(segment.voice, str) else None,
            phonemes=f"phonemes:{segment.text}",
            token_ids=tuple(range(token_count)),
            alignment_tokens=alignments,
        )


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
    def __init__(self, token_counter: Callable[[str], int] | None = None) -> None:
        self.backend = FakeBackend()
        self.g2p_adapter = FakeG2PAdapter(token_counter)
        super().__init__(self.g2p_adapter, backend_factory=lambda config: self.backend)


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


def _request(text: str, **kwargs: Any) -> SynthesisSegment:
    return SynthesisSegment("request", text, "en-us", **kwargs)


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


def test_oversized_none_mode_never_imports_or_uses_phrasplit(
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


def test_sentence_mode_short_request_does_not_import_phrasplit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import: Callable[..., Any] = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "phrasplit" or name.startswith("phrasplit."):
            raise AssertionError("short request must not import PhraseSplit")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    request = _request("Short request.")

    segments = FakeRenderer()._build_phoneme_segments(
        request, _prepared(request.text, 2), _config(long_text_split="sentence"), max_tokens=4
    )

    assert len(segments) == 1
    assert segments[0].text == request.text


def test_sentence_mode_splits_and_packs_with_simple_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import: Callable[..., Any] = builtins.__import__
    phrasplit_imports: list[str] = []

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "spacy" or name.startswith("spacy."):
            raise AssertionError("simple PhraseSplit mode must not import spaCy")
        if name == "phrasplit":
            phrasplit_imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        "pykokoro.request_renderer.get_model_profile",
        lambda *_: SimpleNamespace(max_tokens=4, sample_rate=24000),
    )
    text = "First sentence. Second sentence. Third sentence."
    request = _request(text)
    renderer = FakeRenderer()

    result = renderer.render(
        _prepared(text, 6),
        request,
        _config(long_text_split="sentence", long_text_use_spacy=False, return_trace=True),
    )

    assert phrasplit_imports
    assert [segment.text for segment in renderer.backend.generated] == [
        "First sentence. Second sentence.",
        " Third sentence.",
    ]
    assert "".join(segment.text for segment in renderer.backend.generated) == text
    assert all(len(segment.tokens) <= 4 for segment in renderer.backend.generated)
    assert result.text == text
    assert result.audio.size == 8
    assert result.trace is not None
    assert result.trace.model["acoustic_chunk_count"] == 2
    assert result.trace.model["model_chunk_token_counts"] == [4, 2]


def test_sentence_mode_rebases_pronunciation_overrides_and_annotations() -> None:
    text = "First. Second."
    request = _request(
        text,
        pronunciation_overrides=(PronunciationOverride(7, 13, phonemes="sɛkənd"),),
        tokens=(LinguisticToken(7, 13, text="Second", pos="NOUN"),),
    )
    renderer = FakeRenderer()

    segments = renderer._build_phoneme_segments(
        request,
        _prepared(text, 4),
        _config(long_text_split="sentence"),
        max_tokens=1,
    )

    assert len(segments) == 2
    assert "".join(segment.text for segment in segments) == text
    assert (segments[1].char_start, segments[1].char_end) == (6, len(text))
    alignment = segments[1].alignment_tokens[0]
    assert (alignment.char_start, alignment.char_end) == (6, len(text))
    rebased = next(
        segment
        for segment in renderer.g2p_adapter.phonemized_segments
        if segment.text == " Second."
    )
    assert (
        rebased.pronunciation_overrides[0].start,
        rebased.pronunciation_overrides[0].end,
    ) == (1, 7)
    assert (rebased.tokens[0].start, rebased.tokens[0].end, rebased.tokens[0].text) == (
        1,
        7,
        "Second",
    )


def test_sentence_mode_long_sentence_falls_back_to_word_boundaries() -> None:
    text = "one two three four five"
    request = _request(text)
    renderer = FakeRenderer()

    segments = renderer._build_phoneme_segments(
        request,
        _prepared(text, 5),
        _config(long_text_split="sentence"),
        max_tokens=2,
    )

    assert "".join(segment.text for segment in segments) == text
    assert len(segments) == 3
    assert all(0 < len(segment.tokens) <= 2 for segment in segments)


def test_sentence_mode_rejects_a_single_word_over_the_model_limit() -> None:
    renderer = FakeRenderer(lambda text: len(text.strip()) if text.strip() else 0)
    request = _request("oversized")

    with pytest.raises(SynthesisInputTooLongError, match="single word") as raised:
        renderer._build_phoneme_segments(
            request,
            _prepared(request.text, len(request.text)),
            _config(long_text_split="sentence"),
            max_tokens=4,
        )

    assert raised.value.token_count == len(request.text)
    assert raised.value.max_tokens == 4
