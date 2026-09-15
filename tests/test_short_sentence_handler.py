"""Tests for pykokoro.short_sentence_handler module."""

import pytest

from pykokoro.short_sentence_handler import (
    PhraseResolveMode,
    RandomizedPhraseResolveMode,
    ShortSentenceConfig,
    _phrase_choices,
    _select_phrase_template,
    _terminal_form,
    apply_short_sentence_mode,
    is_segment_empty,
    is_segment_short,
)
from pykokoro.short_sentence_phrases import (
    ShortSentencePhraseSet,
    resolve_short_sentence_phrase_language,
)
from pykokoro.types import PhonemeSegment


def test_phrase_cutters_default_to_timestamp_adaptive_with_compatibility_option() -> None:
    assert PhraseResolveMode().cutter == "timestamp-adaptive"
    assert RandomizedPhraseResolveMode().cutter == "timestamp-adaptive"
    assert PhraseResolveMode(cutter="energy-valley").cutter == "energy-valley"
    assert RandomizedPhraseResolveMode(cutter="energy-valley").cutter == "energy-valley"


def test_randomized_templates_match_terminal_punctuation() -> None:
    mode = RandomizedPhraseResolveMode()
    expected = {
        "Hello.": mode.end_phrases,
        "Why?": mode.question_phrases,
        "Stop!": mode.exclamation_phrases,
        "Wait…": mode.ellipsis_phrases,
        "Go": mode.fragment_phrases,
    }
    for text, templates in expected.items():
        assert _terminal_form(text) in {
            "declarative",
            "question",
            "exclamation",
            "ellipsis",
            "fragment",
        }
        assert _phrase_choices(text, mode) == templates
        if text != "Go":
            assert all(template.endswith("{segment}") for template in templates)


def test_short_sentence_config_rejects_negative_thresholds_and_retries():
    import pytest

    with pytest.raises(ValueError, match="min_phoneme_length"):
        ShortSentenceConfig(min_phoneme_length=-1)
    with pytest.raises(ValueError, match="phrase_fallback_tries"):
        ShortSentenceConfig(phrase_fallback_tries=-1)


def test_phrase_modes_validate_cutter_settings() -> None:
    import pytest

    mode_types = (PhraseResolveMode, RandomizedPhraseResolveMode)
    invalid_values = {
        "frame_duration_ms": 0,
        "energy_threshold": float("nan"),
        "min_silence_seconds": -0.1,
        "search_radius_ms": 0.0,
        "context_guard_ms": -1.0,
        "analysis_window_ms": float("inf"),
    }
    for mode_type in mode_types:
        for field_name, value in invalid_values.items():
            with pytest.raises(ValueError, match=field_name):
                mode_type(**{field_name: value})

    for mode_type in mode_types:
        mode = mode_type()
        assert mode.search_radius_ms == 35.0
        assert mode.context_guard_ms == 8.0
        assert mode.analysis_window_ms == 5.0


def test_short_sentence_config_rejects_unknown_mode_and_malformed_template():
    import pytest

    with pytest.raises(ValueError, match="resolve_mode"):
        ShortSentenceConfig(resolve_mode="missing")
    with pytest.raises(ValueError, match="placeholder"):
        ShortSentenceConfig(
            resolve_mode="phrase",
            resolve_modes={
                "phrase": PhraseResolveMode(neutral_phrase="missing target"),
            },
        )


def make_segment(text: str, phonemes: str) -> PhonemeSegment:
    """Create a PhonemeSegment for tests."""
    return PhonemeSegment(
        id="seg_0_ph0",
        segment_id="seg_0",
        phoneme_id=0,
        text=text,
        phonemes=phonemes,
        tokens=[],
        char_start=0,
        char_end=len(text),
        paragraph_idx=0,
        sentence_idx=0,
        clause_idx=0,
    )


class TestIsSegmentEmpty:
    """Tests for is_segment_empty function."""

    def test_empty_phonemes_returns_false(self):
        """Whitespace-only phonemes should be treated as not empty."""
        segment = make_segment(text="", phonemes="   ")
        assert is_segment_empty(segment) is False

    def test_punctuation_only_short_returns_true(self):
        """Short punctuation-only segments should be considered empty."""
        segment = make_segment(text="!", phonemes="?!")
        assert is_segment_empty(segment) is True

    def test_punctuation_length_at_threshold_returns_false(self):
        """Segments at the length threshold should not count as empty."""
        segment = make_segment(text="!", phonemes="!!!!!")
        config = ShortSentenceConfig(min_phoneme_length=5)
        assert is_segment_empty(segment, config=config) is False

    def test_non_punctuation_returns_false(self):
        """Segments containing letters should not be considered empty."""
        segment = make_segment(text="Hi", phonemes="a!")
        assert is_segment_empty(segment) is False

    def test_disabled_config_returns_false(self):
        """Disabled config should always return False."""
        segment = make_segment(text="!", phonemes="?!")
        config = ShortSentenceConfig(enabled=False)
        assert is_segment_empty(segment, config=config) is False


class TestIsSegmentShort:
    """Tests for is_segment_short function."""

    def test_short_single_word_returns_true(self):
        """Short single-word segments should be considered short."""
        segment = make_segment(text="Go", phonemes="abc")
        assert is_segment_short(segment) is True

    def test_multi_word_can_be_short(self):
        """Multi-word segments can still be considered short."""
        segment = make_segment(text="Go now", phonemes="abc")
        assert is_segment_short(segment) is True

    def test_long_phonemes_return_false(self):
        """Segments with phonemes at the threshold should not be short."""
        segment = make_segment(text="Go", phonemes="abcde")
        config = ShortSentenceConfig(min_phoneme_length=5)
        assert is_segment_short(segment, config=config) is False

    def test_empty_phonemes_returns_false(self):
        """Whitespace-only phonemes should not be considered short."""
        segment = make_segment(text="", phonemes="   ")
        assert is_segment_short(segment) is False

    def test_disabled_config_returns_false(self):
        """Disabled config should always return False."""
        segment = make_segment(text="Go", phonemes="abc")
        config = ShortSentenceConfig(enabled=False)
        assert is_segment_short(segment, config=config) is False


def test_phrase_language_aliases_are_explicit() -> None:
    assert resolve_short_sentence_phrase_language("en-US") == "en"
    assert resolve_short_sentence_phrase_language("de_DE") == "de"
    assert resolve_short_sentence_phrase_language("xx-YY") is None


def test_custom_phrase_catalog_is_selected_for_segment_language() -> None:
    custom = ShortSentencePhraseSet(
        language="xx",
        declarative=("LOCAL {segment}",),
        question=("LOCAL? {segment}",),
        exclamation=("LOCAL! {segment}",),
        ellipsis=("LOCAL… {segment}",),
        fragment=("LOCAL {segment}",),
    )
    config = ShortSentenceConfig(
        phrase_catalog={"xx": custom},
        resolve_mode="phrase",
    )
    segment = make_segment("word", "abc")
    segment.lang = "xx"
    calls: list[tuple[str, str]] = []

    def context_phonemizer(text: str, language: str):
        calls.append((text, language))
        return type("Result", (), {"phonemes": "context", "ids": [1], "tokens": []})()

    result = apply_short_sentence_mode(
        segment,
        segment.phonemes,
        [1],
        config,
        lambda text: [1 for _ in text],
        context_phonemizer=context_phonemizer,
    )
    assert result.metadata is not None
    assert result.metadata["phrase_language"] == "xx"
    assert result.metadata["phrase_template"] == "LOCAL {segment}"
    assert calls == [("LOCAL word", "xx")]


def test_unsupported_phrase_language_uses_wrap_without_english_context() -> None:
    config = ShortSentenceConfig(resolve_mode="phrase")
    segment = make_segment("word", "abc")
    segment.lang = "xx"
    calls: list[tuple[str, str]] = []
    result = apply_short_sentence_mode(
        segment,
        segment.phonemes,
        [1],
        config,
        lambda text: [1 for _ in text],
        context_phonemizer=lambda text, language: calls.append((text, language)),
    )
    assert result.metadata is not None
    assert result.metadata["kind"] == "wrap"
    assert result.metadata["cut_failure_reason"] == "no-localized-phrase-catalog"
    assert calls == []


def test_swedish_builtin_phrase_catalog_is_used() -> None:
    config = ShortSentenceConfig(resolve_mode="phrase")
    segment = make_segment("Ja.", "abc")
    segment.lang = "sv"
    calls: list[tuple[str, str]] = []

    def context_phonemizer(text: str, language: str):
        calls.append((text, language))
        return type(
            "Result",
            (),
            {"phonemes": "context", "ids": [1], "tokens": []},
        )()

    result = apply_short_sentence_mode(
        segment,
        segment.phonemes,
        [1],
        config,
        lambda text: [1 for _ in text],
        context_phonemizer=context_phonemizer,
    )

    assert result.metadata is not None
    assert result.metadata["kind"] == "phrase"
    assert result.metadata["phrase_language"] == "sv"
    assert "Ja." in calls[0][0]
    assert calls[0][1] == "sv"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("はい。", "declarative"),
        ("为什么？", "question"),
        ("لماذا؟", "question"),
        ("止まれ！", "exclamation"),
        ("हाँ।", "declarative"),
        ("等等……", "ellipsis"),
    ],
)
def test_terminal_form_supports_native_punctuation(text: str, expected: str) -> None:
    assert _terminal_form(text) == expected


def test_cutter_override_keeps_german_builtin_carrier_localized() -> None:
    from pykokoro.short_sentence_phrases import GERMAN_SHORT_SENTENCE_PHRASES

    mode = PhraseResolveMode(cutter="energy-valley")
    template = _select_phrase_template("zwölf.", mode, phrase_set=GERMAN_SHORT_SENTENCE_PHRASES)
    assert template == GERMAN_SHORT_SENTENCE_PHRASES.declarative[0]
