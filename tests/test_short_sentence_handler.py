"""Tests for pykokoro.short_sentence_handler module."""

from pykokoro.short_sentence_handler import (
    ShortSentenceConfig,
    is_segment_empty,
    is_segment_short,
)
from pykokoro.types import PhonemeSegment


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
