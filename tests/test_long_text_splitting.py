from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import phrasplit
import pytest

from pykokoro import SynthesisInputTooLongError
from pykokoro.generation_config import GenerationConfig
from pykokoro.prepared_g2p import PreparedSynthesis
from pykokoro.request_renderer import OnnxRequestRenderer
from pykokoro.synthesis_config import SynthesisConfig
from pykokoro.synthesis_types import RenderedSegment, SynthesisSegment
from pykokoro.types import G2PAlignmentToken, Trace, WordTiming


class FakeG2PAdapter:
    def ids_to_phonemes(self, token_ids: list[int] | tuple[int, ...], target_model: str) -> str:
        return f"phonemes:{','.join(str(token_id) for token_id in token_ids)}"

    def phonemize_context(self, text: str, language: str, config: SynthesisConfig) -> object:
        return object()


class FakeBackend:
    def resolve_voice_style(self, voice: str) -> str:
        return voice

    def preprocess_segments(self, segments, enable_short_sentence, **kwargs):
        return segments

    def generate_raw_audio_segments(self, segments, voice_style, speed, **kwargs):
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
        super().__init__(FakeG2PAdapter(), backend_factory=lambda config: FakeBackend())


def _config(mode: str = "sentence", **kwargs: Any) -> SynthesisConfig:
    return SynthesisConfig(
        voice="af_heart",
        generation=GenerationConfig(lang="en-us"),
        model_source="github",
        model_variant="v1.0",
        long_text_split=mode,  # type: ignore[arg-type]
        **kwargs,
    )


def _prepared(
    text: str,
    spans: list[tuple[int, int, int]],
    *,
    phonemes: str | None = None,
) -> PreparedSynthesis:
    alignments = tuple(
        G2PAlignmentToken(
            text=text[start:end],
            phonemes=text[start:end],
            char_start=start,
            char_end=end,
            model_span_token_count=count,
        )
        for start, end, count in spans
    )
    token_count = sum(span[2] for span in spans)
    return PreparedSynthesis(
        request_id="request",
        text=text,
        language="en-us",
        voice="af_heart",
        phonemes=phonemes if phonemes is not None else text,
        token_ids=tuple(range(token_count)),
        alignment_tokens=alignments,
    )


def _request(text: str, *, phonemes: str | None = None) -> SynthesisSegment:
    return SynthesisSegment("request", text, "en-us", phonemes=phonemes)


def _block_phrasplit_import(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import: Callable[..., Any] = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "phrasplit" or name.startswith("phrasplit."):
            raise AssertionError("PhraseSplit must not be imported for this request")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_short_sentence_mode_does_not_import_phrasplit(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_phrasplit_import(monkeypatch)
    prepared = _prepared("Hi.", [(0, 3, 2)])

    chunks = FakeRenderer()._build_phoneme_segments(
        _request("Hi."), prepared, _config(), max_tokens=2
    )

    assert [chunk.tokens for chunk in chunks] == [[0, 1]]


def test_token_mode_never_imports_phrasplit_and_conserves_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_phrasplit_import(monkeypatch)
    text = "abcdef"
    prepared = _prepared(text, [(index, index + 1, 1) for index in range(len(text))])

    chunks = FakeRenderer()._build_phoneme_segments(
        _request(text), prepared, _config("token"), max_tokens=4
    )

    assert [len(chunk.tokens) for chunk in chunks] == [4, 2]
    assert [token for chunk in chunks for token in chunk.tokens] == list(prepared.token_ids)


def test_direct_phoneme_sentence_mode_uses_token_fallback_without_phrasplit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_phrasplit_import(monkeypatch)
    text = "abc def."
    prepared = _prepared(text, [(0, 3, 3), (4, 8, 3)], phonemes="ˈæb ˈdɛf")
    trace = Trace()

    chunks = FakeRenderer()._build_phoneme_segments(
        _request(text, phonemes="ˈæb ˈdɛf"),
        prepared,
        _config(),
        max_tokens=4,
        trace=trace,
    )

    assert [len(chunk.tokens) for chunk in chunks] == [3, 3]
    assert [token for chunk in chunks for token in chunk.tokens] == list(prepared.token_ids)
    assert trace.model["fallback_reason"] == "direct-phoneme-input"


def test_none_mode_renders_short_text_and_rejects_oversized_text_without_phrasplit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_phrasplit_import(monkeypatch)
    renderer = FakeRenderer()
    short = _prepared("Hi.", [(0, 3, 2)])
    assert renderer._build_phoneme_segments(_request("Hi."), short, _config("none"), max_tokens=2)[
        0
    ].tokens == [0, 1]

    long = _prepared("Long.", [(0, 5, 4)])
    with pytest.raises(SynthesisInputTooLongError, match="request.*4 model tokens.*maximum is 2"):
        renderer._build_phoneme_segments(_request("Long."), long, _config("none"), max_tokens=2)


def test_sentence_mode_packs_whole_sentences_without_losing_tokens() -> None:
    text = "Alpha. Beta. Gamma."
    prepared = _prepared(text, [(0, 6, 250), (7, 12, 250), (13, 19, 100)])

    groups, sentence_count, reason, oversized = OnnxRequestRenderer._split_at_sentence_boundaries(
        text=text,
        language="en-us",
        token_ids=list(prepared.token_ids),
        alignments=prepared.alignment_tokens,
        max_tokens=510,
    )

    assert groups is not None
    assert [len(ids) for ids, _ in groups] == [500, 100]
    assert sentence_count == 3
    assert reason is None
    assert not oversized
    assert [token for ids, _ in groups for token in ids] == list(prepared.token_ids)
    assert [alignment.char_start for _, alignments in groups for alignment in alignments] == [
        0,
        7,
        13,
    ]


def test_sentence_packing_does_not_split_to_fill_remaining_budget() -> None:
    text = "First. Second. Third."
    prepared = _prepared(text, [(0, 6, 300), (7, 14, 250), (15, 21, 100)])

    groups, _, _, _ = OnnxRequestRenderer._split_at_sentence_boundaries(
        text=text,
        language="en-us",
        token_ids=list(prepared.token_ids),
        alignments=prepared.alignment_tokens,
        max_tokens=510,
    )

    assert groups is not None
    assert [len(ids) for ids, _ in groups] == [300, 350]


def test_duplicate_sentences_keep_distinct_source_ranges() -> None:
    text = "Repeat this. Repeat this. Final sentence."
    prepared = _prepared(text, [(0, 12, 2), (13, 25, 2), (26, 41, 2)])

    groups, _, _, _ = OnnxRequestRenderer._split_at_sentence_boundaries(
        text=text,
        language="en-us",
        token_ids=list(prepared.token_ids),
        alignments=prepared.alignment_tokens,
        max_tokens=4,
    )

    assert groups is not None
    assert [[item.text for item in alignments] for _, alignments in groups] == [
        ["Repeat this.", "Repeat this."],
        ["Final sentence."],
    ]
    assert groups[0][1][0].char_start == 0
    assert groups[0][1][1].char_start == 13


def test_oversized_sentence_falls_back_to_alignment_token_boundaries() -> None:
    text = "abc def."
    prepared = _prepared(text, [(0, 3, 3), (4, 8, 3)])

    groups, sentence_count, reason, oversized = OnnxRequestRenderer._split_at_sentence_boundaries(
        text=text,
        language="en-us",
        token_ids=list(prepared.token_ids),
        alignments=prepared.alignment_tokens,
        max_tokens=4,
    )

    assert groups is not None
    assert [len(ids) for ids, _ in groups] == [3, 3]
    assert sentence_count == 1
    assert reason is None
    assert oversized
    assert [token for ids, _ in groups for token in ids] == list(prepared.token_ids)


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        (
            "Dr. Smith arrived at 3 p.m. He sat down.",
            "en-us",
            ["Dr. Smith arrived at 3 p.m.", "He sat down."],
        ),
        (
            "Version 3.14 is ready. Continue with the next step.",
            "en-us",
            ["Version 3.14 is ready.", "Continue with the next step."],
        ),
        (
            "Das ist z. B. ein Test. Danach geht es weiter.",
            "de-DE",
            ["Das ist z. B. ein Test.", "Danach geht es weiter."],
        ),
        (
            "First.\n\nSecond!\nThird?",
            "en-gb",
            ["First.", "Second!", "Third?"],
        ),
    ],
)
def test_real_phrasplit_regex_returns_exact_sentence_slices(
    text: str, language: str, expected: list[str]
) -> None:
    sentences = phrasplit.split_with_offsets(
        text,
        mode="sentence",
        use_spacy=False,
        language=language,
    )

    assert [sentence.text for sentence in sentences] == expected
    assert all(text[item.char_start : item.char_end] == item.text for item in sentences)


def test_internal_sentence_path_forces_regex_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    split = phrasplit.split_with_offsets

    def capture(text: str, **kwargs: Any):
        calls.append(kwargs)
        return split(text, **kwargs)

    monkeypatch.setattr(phrasplit, "split_with_offsets", capture)
    text = "First. Second."
    prepared = _prepared(text, [(0, 6, 3), (7, 14, 3)])

    groups, _, _, _ = OnnxRequestRenderer._split_at_sentence_boundaries(
        text=text,
        language="en-us",
        token_ids=list(prepared.token_ids),
        alignments=prepared.alignment_tokens,
        max_tokens=4,
    )

    assert groups is not None
    assert calls == [{"mode": "sentence", "use_spacy": False, "language": "en-us"}]


def test_invalid_alignment_falls_back_to_raw_token_safety_and_trace() -> None:
    text = "long text"
    prepared = replace(_prepared(text, [(0, 4, 2)]), token_ids=tuple(range(7)))
    trace = Trace()
    chunks = FakeRenderer()._build_phoneme_segments(
        _request(text), prepared, _config(), max_tokens=3, trace=trace
    )

    assert [len(chunk.tokens) for chunk in chunks] == [3, 3, 1]
    assert [token for chunk in chunks for token in chunk.tokens] == list(prepared.token_ids)
    assert trace.model["fallback_reason"] == "invalid-g2p-alignment"


def test_sentence_chunks_stitch_one_result_with_rebased_timings() -> None:
    text = "Alpha. Beta. Gamma."
    request = _request(text)
    prepared = _prepared(text, [(0, 6, 250), (7, 12, 250), (13, 19, 100)])

    rendered: RenderedSegment = FakeRenderer().render(prepared, request, _config())

    assert rendered.id == request.id
    assert len(rendered.audio) == 8
    assert [timing.start_sample for timing in rendered.word_timings] == [1, 5]
    assert [timing.end_sample for timing in rendered.word_timings] == [3, 7]
    assert [(timing.char_start, timing.char_end) for timing in rendered.word_timings] == [
        (0, 12),
        (13, 19),
    ]
    assert all(
        timing.start_sample <= timing.end_sample <= len(rendered.audio)
        for timing in rendered.word_timings
    )
