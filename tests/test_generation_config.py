"""Tests for GenerationConfig dataclass."""

import pytest

from pykokoro.generation_config import GenerationConfig


class TestGenerationConfigCreation:
    """Test GenerationConfig creation and defaults."""

    def test_default_config(self):
        """Test creating config with all defaults."""
        config = GenerationConfig()
        assert config.speed == 1.0
        assert config.lang == "en-us"
        assert config.is_phonemes is False
        assert config.pause_mode == "tts"
        assert config.pause_clause == 0.3
        assert config.pause_sentence == 0.6
        assert config.pause_paragraph == 1.0
        assert config.pause_variance == 0.05
        assert config.random_seed is None
        assert config.enable_short_sentence is None

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = GenerationConfig(
            speed=1.5,
            lang="en-gb",
            is_phonemes=True,
            pause_mode="manual",
            pause_clause=0.25,
            pause_sentence=0.5,
            pause_paragraph=0.8,
            pause_variance=0.1,
            random_seed=42,
            enable_short_sentence=True,
        )
        assert config.speed == 1.5
        assert config.lang == "en-gb"
        assert config.is_phonemes is True
        assert config.pause_mode == "manual"
        assert config.pause_clause == 0.25
        assert config.pause_sentence == 0.5
        assert config.pause_paragraph == 0.8
        assert config.pause_variance == 0.1
        assert config.random_seed == 42
        assert config.enable_short_sentence is True

    def test_partial_config(self):
        """Test creating config with some custom values."""
        config = GenerationConfig(
            speed=1.2,
            pause_mode="manual",
        )
        assert config.speed == 1.2
        assert config.pause_mode == "manual"
        # Defaults for others
        assert config.lang == "en-us"
        assert config.pause_clause == 0.3


class TestGenerationConfigValidation:
    """Test GenerationConfig parameter validation."""

    def test_invalid_speed_negative(self):
        """Test that negative speed raises ValueError."""
        with pytest.raises(ValueError, match="speed must be > 0.0"):
            GenerationConfig(speed=-1.0)

    def test_invalid_speed_zero(self):
        """Test that zero speed raises ValueError."""
        with pytest.raises(ValueError, match="speed must be > 0.0"):
            GenerationConfig(speed=0.0)

    def test_invalid_pause_clause_negative(self):
        """Test that negative pause_clause raises ValueError."""
        with pytest.raises(ValueError, match="pause_clause must be >= 0.0"):
            GenerationConfig(pause_clause=-0.1)

    def test_invalid_pause_sentence_negative(self):
        """Test that negative pause_sentence raises ValueError."""
        with pytest.raises(ValueError, match="pause_sentence must be >= 0.0"):
            GenerationConfig(pause_sentence=-0.5)

    def test_invalid_pause_paragraph_negative(self):
        """Test that negative pause_paragraph raises ValueError."""
        with pytest.raises(ValueError, match="pause_paragraph must be >= 0.0"):
            GenerationConfig(pause_paragraph=-1.0)

    def test_invalid_pause_variance_negative(self):
        """Test that negative pause_variance raises ValueError."""
        with pytest.raises(ValueError, match="pause_variance must be >= 0.0"):
            GenerationConfig(pause_variance=-0.05)

    def test_invalid_pause_mode(self):
        """Test that invalid pause_mode raises ValueError."""
        with pytest.raises(
            ValueError,
            match="pause_mode must be 'tts', 'manual', or 'auto'",
        ):
            GenerationConfig(pause_mode="invalid")  # type: ignore

    def test_invalid_lang_empty(self):
        """Test that empty lang raises ValueError."""
        with pytest.raises(ValueError, match="lang must be a non-empty string"):
            GenerationConfig(lang="")

    def test_valid_zero_pauses(self):
        """Test that zero pause durations are valid."""
        config = GenerationConfig(
            pause_clause=0.0,
            pause_sentence=0.0,
            pause_paragraph=0.0,
            pause_variance=0.0,
        )
        assert config.pause_clause == 0.0
        assert config.pause_sentence == 0.0
        assert config.pause_paragraph == 0.0
        assert config.pause_variance == 0.0


class TestGenerationConfigImmutability:
    """Test that GenerationConfig is frozen/immutable."""

    def test_cannot_modify_speed(self):
        """Test that modifying speed raises an error."""
        config = GenerationConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.speed = 2.0  # type: ignore[misc]

    def test_cannot_modify_lang(self):
        """Test that modifying lang raises an error."""
        config = GenerationConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.lang = "de"  # type: ignore[misc]

    def test_cannot_add_attribute(self):
        """Test that adding new attributes raises an error."""
        config = GenerationConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.new_attr = "value"  # type: ignore[misc]


class TestGenerationConfigMerge:
    """Test merge_with_kwargs functionality."""

    def test_merge_no_kwargs(self):
        """Test merging with no kwargs returns config values."""
        config = GenerationConfig(speed=1.5, lang="de")
        merged = config.merge_with_kwargs()
        assert merged["speed"] == 1.5
        assert merged["lang"] == "de"

    def test_merge_override_speed(self):
        """Test merging with speed override."""
        config = GenerationConfig(speed=1.5)
        merged = config.merge_with_kwargs(speed=2.0)
        assert merged["speed"] == 2.0

    def test_merge_override_multiple(self):
        """Test merging with multiple overrides."""
        config = GenerationConfig(
            speed=1.0,
            lang="en-us",
            pause_mode="tts",
        )
        merged = config.merge_with_kwargs(
            speed=1.5,
            lang="de",
        )
        assert merged["speed"] == 1.5
        assert merged["lang"] == "de"
        assert merged["pause_mode"] == "tts"  # Not overridden

    def test_merge_none_kwargs_ignored(self):
        """Test that None kwargs don't override config."""
        config = GenerationConfig(speed=1.5, lang="de")
        merged = config.merge_with_kwargs(speed=None, lang=None)
        assert merged["speed"] == 1.5  # Not overridden
        assert merged["lang"] == "de"  # Not overridden

    def test_merge_mixed_none_and_values(self):
        """Test merging with mix of None and actual values."""
        config = GenerationConfig(speed=1.0, lang="en-us", pause_mode="tts")
        merged = config.merge_with_kwargs(
            speed=None,  # Should not override
            lang="de",  # Should override
            pause_mode=None,  # Should not override
        )
        assert merged["speed"] == 1.0
        assert merged["lang"] == "de"
        assert merged["pause_mode"] == "tts"

    def test_merge_all_params(self):
        """Test merging all possible parameters."""
        config = GenerationConfig()
        merged = config.merge_with_kwargs(
            speed=1.5,
            lang="de",
            is_phonemes=True,
            pause_mode="manual",
            pause_clause=0.25,
            pause_sentence=0.5,
            pause_paragraph=0.8,
            pause_variance=0.1,
            random_seed=42,
            enable_short_sentence=True,
        )
        assert merged["speed"] == 1.5
        assert merged["lang"] == "de"
        assert merged["is_phonemes"] is True
        assert merged["pause_mode"] == "manual"
        assert merged["pause_clause"] == 0.25
        assert merged["pause_sentence"] == 0.5
        assert merged["pause_paragraph"] == 0.8
        assert merged["pause_variance"] == 0.1
        assert merged["random_seed"] == 42
        assert merged["enable_short_sentence"] is True


class TestGenerationConfigEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_slow_speed(self):
        """Test very slow speed values."""
        config = GenerationConfig(speed=0.1)
        assert config.speed == 0.1

    def test_very_fast_speed(self):
        """Test very fast speed values."""
        config = GenerationConfig(speed=10.0)
        assert config.speed == 10.0

    def test_very_long_pauses(self):
        """Test very long pause durations."""
        config = GenerationConfig(
            pause_clause=5.0,
            pause_sentence=10.0,
            pause_paragraph=20.0,
        )
        assert config.pause_clause == 5.0
        assert config.pause_sentence == 10.0
        assert config.pause_paragraph == 20.0

    def test_large_random_seed(self):
        """Test large random seed values."""
        config = GenerationConfig(random_seed=999999999)
        assert config.random_seed == 999999999

    def test_different_languages(self):
        """Test various language codes."""
        languages = ["en-us", "en-gb", "de", "fr", "es", "ja", "zh", "ko"]
        for lang in languages:
            config = GenerationConfig(lang=lang)
            assert config.lang == lang
