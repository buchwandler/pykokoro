"""Tests for say-as text normalization."""

import pytest

from pykokoro.say_as import (
    normalize_cardinal,
    normalize_characters,
    normalize_date,
    normalize_digits,
    normalize_expletive,
    normalize_fraction,
    normalize_ordinal,
    normalize_say_as,
    normalize_telephone,
    normalize_time,
    normalize_unit,
)


class TestCardinalNormalization:
    """Tests for cardinal number normalization."""

    def test_basic_cardinal(self):
        """Test basic cardinal numbers."""
        assert normalize_cardinal("123", "en-us") == "one hundred and twenty-three"
        assert normalize_cardinal("1", "en-us") == "one"
        assert normalize_cardinal("1000", "en-us") == "one thousand"

    def test_cardinal_with_commas(self):
        """Test cardinal with comma separators."""
        assert "thousand" in normalize_cardinal("1,000", "en-us").lower()

    def test_cardinal_float(self):
        """Test cardinal with decimal numbers."""
        result = normalize_cardinal("3.14", "en-us")
        assert "three" in result.lower()
        assert "point" in result.lower()

    def test_cardinal_negative(self):
        """Test negative cardinal numbers."""
        result = normalize_cardinal("-5", "en-us")
        assert "minus" in result.lower() or "negative" in result.lower()


class TestOrdinalNormalization:
    """Tests for ordinal number normalization."""

    def test_basic_ordinal(self):
        """Test basic ordinal numbers."""
        assert normalize_ordinal("1", "en-us") == "first"
        assert normalize_ordinal("2", "en-us") == "second"
        assert normalize_ordinal("3", "en-us") == "third"
        assert normalize_ordinal("21", "en-us") == "twenty-first"

    def test_ordinal_large_numbers(self):
        """Test ordinal with larger numbers."""
        result = normalize_ordinal("100", "en-us")
        assert "hundredth" in result.lower()


class TestDigitsNormalization:
    """Tests for digits normalization."""

    def test_basic_digits(self):
        """Test basic digit separation."""
        result = normalize_digits("123", "en-us")
        # Should contain "one", "two", "three" separately
        assert "one" in result.lower()
        assert "two" in result.lower()
        assert "three" in result.lower()

    def test_digits_with_non_digits(self):
        """Test digits extraction from mixed text."""
        result = normalize_digits("abc123def", "en-us")
        # Should extract only digits
        assert "one" in result.lower()
        assert "two" in result.lower()
        assert "three" in result.lower()


class TestCharactersNormalization:
    """Tests for character spell-out normalization."""

    def test_basic_characters(self):
        """Test basic character spelling."""
        assert normalize_characters("ABC", "en-us") == "A B C"
        assert normalize_characters("Hello", "en-us") == "H E L L O"

    def test_single_character(self):
        """Test single character."""
        assert normalize_characters("A", "en-us") == "A"

    def test_characters_ignore_whitespace(self):
        """Whitespace should be ignored when spelling characters."""
        assert normalize_characters("A B", "en-us") == "A B"
        assert normalize_characters("A\tB\nC", "en-us") == "A B C"


class TestExpletiveNormalization:
    """Tests for expletive censoring."""

    def test_expletive_censoring(self):
        """Test that expletives are censored."""
        assert normalize_expletive("fuck", "en-us") == "beep"
        assert normalize_expletive("anything", "en-us") == "beep"


class TestFractionNormalization:
    """Tests for fraction normalization."""

    def test_basic_fractions(self):
        """Test basic fractions."""
        assert "half" in normalize_fraction("1/2", "en-us").lower()
        assert "quarter" in normalize_fraction("1/4", "en-us").lower()

    def test_complex_fractions(self):
        """Test complex fractions."""
        result = normalize_fraction("3/4", "en-us")
        assert "three" in result.lower()
        assert "quarter" in result.lower()

    def test_ordinal_fractions(self):
        """Test fractions with ordinal denominators."""
        result = normalize_fraction("2/3", "en-us")
        assert "two" in result.lower()
        assert "third" in result.lower()


class TestTelephoneNormalization:
    """Tests for telephone number normalization."""

    def test_basic_telephone(self):
        """Test basic telephone normalization."""
        result = normalize_telephone("+1-555-0123", "en-us")
        # Should contain "plus" and digit words
        assert "plus" in result.lower()
        assert "one" in result.lower()

    def test_telephone_digits_only(self):
        """Test telephone with just digits."""
        result = normalize_telephone("5550123", "en-us")
        assert "five" in result.lower()


class TestDateNormalization:
    """Tests for date normalization."""

    def test_date_mdy_format(self):
        """Test date in M/D/Y format."""
        result = normalize_date("12/31/2024", "en-us")
        # Should contain month, day, year
        assert "2024" in result or "twenty" in result.lower()

    def test_date_ymd_format(self):
        """Test date in Y-M-D format."""
        result = normalize_date("2024-12-31", "en-us")
        assert len(result) > 0

    def test_invalid_date(self):
        """Test handling of invalid dates."""
        result = normalize_date("invalid", "en-us")
        assert result == "invalid"


class TestTimeNormalization:
    """Tests for time normalization."""

    def test_24hour_time(self):
        """Test 24-hour time format."""
        result = normalize_time("14:30", "en-us")
        assert len(result) > 0

    def test_12hour_time(self):
        """Test 12-hour time format."""
        result = normalize_time("2:30 PM", "en-us")
        assert len(result) > 0

    def test_invalid_time(self):
        """Test handling of invalid times."""
        result = normalize_time("invalid", "en-us")
        assert result == "invalid"


class TestUnitNormalization:
    """Tests for unit normalization."""

    def test_metric_units(self):
        """Test metric units."""
        result = normalize_unit("5kg", "en-us")
        assert "five" in result.lower()
        assert "kilogram" in result.lower()

    def test_imperial_units(self):
        """Test imperial units."""
        result = normalize_unit("10lb", "en-us")
        assert "ten" in result.lower()
        assert "pound" in result.lower()

    def test_singular_units(self):
        """Test singular units."""
        result = normalize_unit("1kg", "en-us")
        assert "one" in result.lower()
        assert "kilogram" in result.lower()
        # Should be singular, not "kilograms"

    def test_decimal_units(self):
        """Test decimal with units."""
        result = normalize_unit("2.5m", "en-us")
        assert "two" in result.lower() or "2.5" in result.lower()
        assert "meter" in result.lower()


class TestNormalizeSayAs:
    """Tests for the main normalize_say_as dispatcher."""

    def test_cardinal_dispatch(self):
        """Test dispatching to cardinal normalizer."""
        result = normalize_say_as("123", "cardinal", lang="en-us")
        assert "hundred" in result.lower()

    def test_ordinal_dispatch(self):
        """Test dispatching to ordinal normalizer."""
        result = normalize_say_as("3", "ordinal", lang="en-us")
        assert result.lower() == "third"

    def test_digits_dispatch(self):
        """Test dispatching to digits normalizer."""
        result = normalize_say_as("123", "digits", lang="en-us")
        assert "one" in result.lower()
        assert "two" in result.lower()

    def test_characters_dispatch(self):
        """Test dispatching to characters normalizer."""
        result = normalize_say_as("ABC", "characters", lang="en-us")
        assert result == "A B C"

    def test_expletive_dispatch(self):
        """Test dispatching to expletive normalizer."""
        result = normalize_say_as("word", "expletive", lang="en-us")
        assert result == "beep"

    def test_telephone_dispatch(self):
        """Test dispatching to telephone normalizer."""
        result = normalize_say_as("+1-555-0123", "telephone", lang="en-us")
        assert "plus" in result.lower()

    def test_unknown_interpret_as(self):
        """Test handling of unknown interpret-as types."""
        result = normalize_say_as("test", "unknown_type", lang="en-us")
        # Should return original text
        assert result == "test"

    def test_number_alias(self):
        """Test that 'number' is an alias for 'cardinal'."""
        result1 = normalize_say_as("123", "number", lang="en-us")
        result2 = normalize_say_as("123", "cardinal", lang="en-us")
        assert result1 == result2

    def test_with_format_parameter(self):
        """Test passing format parameter."""
        # Date with format
        result = normalize_say_as("12/31/2024", "date", lang="en-us", format_str="mdy")
        assert len(result) > 0

    def test_with_detail_parameter(self):
        """Test passing detail parameter."""
        result = normalize_say_as("123", "cardinal", lang="en-us", detail="2")
        assert len(result) > 0


class TestLanguageSupport:
    """Tests for multi-language support."""

    def test_french_cardinal(self):
        """Test French cardinal numbers."""
        try:
            result = normalize_cardinal("123", "fr-fr")
            assert "cent" in result.lower()
        except (LookupError, RuntimeError):
            # If French not installed in num2words, skip
            pytest.skip("French language support not available")

    def test_german_ordinal(self):
        """Test German ordinal numbers."""
        try:
            result = normalize_ordinal("3", "de-de")
            # German ordinal for 3 is "dritte"
            assert len(result) > 0
        except (LookupError, RuntimeError):
            pytest.skip("German language support not available")

    def test_fallback_to_english(self):
        """Test fallback to English for unsupported languages."""
        # For unsupported language, should fall back to English
        result = normalize_cardinal("123", "xx-xx")
        assert len(result) > 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_number_cardinal(self):
        """Test handling of non-numeric input for cardinal."""
        result = normalize_cardinal("abc", "en-us")
        # Should return original text
        assert result == "abc"

    def test_invalid_number_ordinal(self):
        """Test handling of non-numeric input for ordinal."""
        result = normalize_ordinal("abc", "en-us")
        assert result == "abc"

    def test_empty_string(self):
        """Test handling of empty strings."""
        assert normalize_cardinal("", "en-us") == ""
        assert normalize_digits("", "en-us") == ""
        assert normalize_characters("", "en-us") == ""
