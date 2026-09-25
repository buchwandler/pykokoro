import numpy as np
import pytest

from pykokoro.exceptions import (
    EmptyTextError,
    InvalidLanguageError,
    InvalidLinguisticTokensError,
    InvalidVoiceError,
)
from pykokoro.synthesis_config import SynthesisConfig
from pykokoro.synthesis_types import (
    LinguisticToken,
    PronunciationOverride,
    RenderedSegment,
    SynthesisRequest,
    SynthesisSegment,
)
from pykokoro.types import WordTiming
from pykokoro.voice_level import VoiceLevelConfig


def test_synthesis_segment_normalizes_language_and_freezes_collections():
    request = SynthesisSegment(
        id="line-1",
        text="Hello world",
        language="EN_us",
        pronunciation_overrides=[PronunciationOverride(0, 5, language="de")],
        annotations=[LinguisticToken(0, 5, text="Hello", pos="INTJ")],
    )

    assert request.language == "en-us"
    assert request.pronunciation_overrides[0].language == "de-de"
    assert isinstance(request.pronunciation_overrides, tuple)
    assert isinstance(request.annotations, tuple)


def test_synthesis_segment_rejects_empty_id_and_unsupported_language():
    with pytest.raises(ValueError, match="id"):
        SynthesisSegment(id="  ", text="Hello", language="en-us")
    with pytest.raises(ValueError, match="unsupported"):
        SynthesisSegment(id="line", text="Hello", language="xx-unsupported")


def test_synthesis_request_accepts_tokens_morph_and_legacy_segment_alias():
    token = LinguisticToken(0, 5, text="Hello", pos="INTJ", morph="Number=Sing")
    request = SynthesisRequest(id="line", text="Hello", language="en-us", tokens=(token,))

    assert request.tokens == (token,)
    assert request.annotations is request.tokens
    assert request.tokens[0].morph == "Number=Sing"
    assert isinstance(request, SynthesisSegment)


def test_synthesis_request_rejects_empty_text_with_typed_error():
    with pytest.raises(EmptyTextError, match="non-whitespace"):
        SynthesisRequest(id="line", text=" \n", language="en-us")


def test_synthesis_request_uses_typed_language_and_voice_errors():
    with pytest.raises(InvalidLanguageError):
        SynthesisRequest(id="line", text="Hello", language="xx-unsupported")
    with pytest.raises(InvalidVoiceError):
        SynthesisRequest(id="line", text="Hello", language="en-us", voice="  ")


def test_synthesis_request_rejects_unsorted_and_overlapping_tokens():
    with pytest.raises(InvalidLinguisticTokensError, match="sorted"):
        SynthesisRequest(
            id="line",
            text="Hello world",
            language="en-us",
            tokens=(
                LinguisticToken(6, 11, text="world"),
                LinguisticToken(0, 5, text="Hello"),
            ),
        )
    with pytest.raises(InvalidLinguisticTokensError, match="overlapping"):
        SynthesisRequest(
            id="line",
            text="Hello",
            language="en-us",
            tokens=(LinguisticToken(0, 3, text="Hel"), LinguisticToken(2, 5, text="llo")),
        )


def test_override_and_annotation_offsets_are_bounded_and_source_aligned():
    with pytest.raises(ValueError, match="offsets"):
        SynthesisSegment(
            id="line",
            text="Hello",
            language="en-us",
            pronunciation_overrides=(PronunciationOverride(0, 6, phonemes="hˈɛloʊ"),),
        )
    with pytest.raises(ValueError, match="offsets"):
        SynthesisSegment(
            id="line", text="Hello", language="en-us", annotations=(LinguisticToken(0, 6),)
        )
    with pytest.raises(ValueError, match="does not match"):
        SynthesisSegment(
            id="line",
            text="Hello",
            language="en-us",
            annotations=(LinguisticToken(0, 5, text="World"),),
        )


def test_override_requires_an_effect_and_rejects_ambiguous_phonemes():
    with pytest.raises(ValueError, match="set phonemes or language"):
        PronunciationOverride(0, 1)
    with pytest.raises(ValueError, match="overlapping direct"):
        SynthesisSegment(
            id="line",
            text="abcdef",
            language="en-us",
            pronunciation_overrides=(
                PronunciationOverride(0, 3, phonemes="a"),
                PronunciationOverride(2, 5, phonemes="b"),
            ),
        )
    with pytest.raises(ValueError, match="whole-request phonemes"):
        SynthesisSegment(
            id="line",
            text="Hello",
            language="en-us",
            phonemes="hˈɛloʊ",
            pronunciation_overrides=(PronunciationOverride(0, 5, phonemes="hˈɛloʊ"),),
        )


def test_language_spans_can_overlap_and_only_direct_phonemes_are_ambiguous():
    request = SynthesisSegment(
        id="line",
        text="Hello world",
        language="en-us",
        pronunciation_overrides=(
            PronunciationOverride(0, 8, language="de"),
            PronunciationOverride(4, 11, language="fr"),
        ),
    )

    assert len(request.pronunciation_overrides) == 2


def test_rendered_segment_enforces_mono_float32_and_request_local_timing_bounds():
    timing = WordTiming(
        text="Hello", char_start=0, char_end=5, start_sample=1, end_sample=4, segment_id="line"
    )
    result = RenderedSegment(
        id="line",
        audio=np.array([0, 0.2, -0.2, 0], dtype=np.float64),
        sample_rate=24000,
        text="Hello",
        language="en-us",
        voice="af_sarah",
        phonemes="hˈɛloʊ",
        token_ids=(1, 2),
        word_timings=(timing,),
    )

    assert result.audio.dtype == np.float32
    assert result.audio.ndim == 1
    assert result.sample_rate > 0
    with pytest.raises(ValueError, match="mono"):
        RenderedSegment(
            id="line",
            audio=np.zeros((2, 2)),
            sample_rate=24000,
            text="Hello",
            language="en-us",
            voice=None,
            phonemes="",
            token_ids=(),
        )
    with pytest.raises(ValueError, match="exceeds"):
        RenderedSegment(
            id="line",
            audio=np.zeros(2),
            sample_rate=24000,
            text="Hello",
            language="en-us",
            voice=None,
            phonemes="",
            token_ids=(),
            word_timings=(timing,),
        )


def test_rendered_segment_rejects_invalid_source_and_nonmonotonic_timings() -> None:
    with pytest.raises(ValueError, match="text must match rendered text"):
        RenderedSegment(
            id="line",
            audio=np.ones(8),
            sample_rate=24000,
            text="Hello",
            language="en-us",
            voice=None,
            phonemes="hello",
            token_ids=(1,),
            word_timings=(WordTiming("World", 0, 5, 0, 4, "line"),),
        )

    with pytest.raises(ValueError, match="sample offsets must be monotonic"):
        RenderedSegment(
            id="line",
            audio=np.ones(8),
            sample_rate=24000,
            text="Hello world",
            language="en-us",
            voice=None,
            phonemes="hello world",
            token_ids=(1, 2),
            word_timings=(
                WordTiming("Hello", 0, 5, 4, 6, "line"),
                WordTiming("world", 6, 11, 1, 3, "line"),
            ),
        )


def test_rendered_segment_rejects_empty_audio() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        RenderedSegment(
            id="line",
            audio=np.zeros(0),
            sample_rate=24000,
            text="Hello",
            language="en-us",
            voice=None,
            phonemes="hello",
            token_ids=(1,),
        )


def test_synthesis_config_only_accepts_engine_owned_options():
    config = SynthesisConfig(voice="af_sarah")

    assert config.voice == "af_sarah"
    assert config.voice_level == VoiceLevelConfig()
    with pytest.raises(TypeError, match="overlap_mode"):
        SynthesisConfig(overlap_mode="legacy")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="gain_db"):
        VoiceLevelConfig(gain_db=float("nan"))
