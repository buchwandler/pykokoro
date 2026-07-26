"""Say-as text normalization for PyKokoro.

This module handles text normalization based on SSMD/SSML say-as interpret-as types.
It converts text into speakable form for TTS using num2words and babel.

Supported interpret-as types:
    Numbers: cardinal, ordinal, digits, number, fraction
    Text: characters, expletive
    Date/Time: date, time
    Other: telephone, address, unit

Example:
    >>> normalize_say_as("123", "cardinal", lang="en-us")
    'one hundred twenty-three'
    >>> normalize_say_as("3", "ordinal", lang="en-us")
    'third'
    >>> normalize_say_as("ABC", "characters", lang="en-us")
    'A B C'
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from babel import Locale
from babel.dates import format_date, format_time
from num2words import num2words

logger = logging.getLogger(__name__)


def _parse_language_to_locale(lang: str) -> str:
    """Convert language code to locale for num2words/babel.

    Args:
        lang: Language code (e.g., 'en-us', 'fr-fr', 'de-de')

    Returns:
        Locale code for num2words (e.g., 'en', 'fr', 'de')
    """
    # Map common language codes to num2words locales
    lang = lang.lower()

    # Extract base language (en-us -> en)
    base_lang = lang.split("-")[0]

    # Map specific variants
    locale_map = {
        "en": "en",
        "en-us": "en",
        "en-gb": "en_GB",
        "en-au": "en",
        "fr": "fr",
        "fr-fr": "fr",
        "de": "de",
        "de-de": "de",
        "es": "es",
        "es-es": "es",
        "it": "it",
        "pt": "pt",
        "pt-br": "pt_BR",
        "ru": "ru",
        "ja": "ja",
        "zh": "zh",
        "ar": "ar",
        "nl": "nl",
        "pl": "pl",
        "tr": "tr",
    }

    return locale_map.get(lang, locale_map.get(base_lang, "en"))


def normalize_cardinal(text: str, lang: str = "en-us", **kwargs: Any) -> str:
    """Convert number to cardinal form (one, two, three...).

    Args:
        text: Number as string (e.g., "123")
        lang: Language code
        **kwargs: Additional arguments (ignored)

    Returns:
        Cardinal number as words
    """
    try:
        num: float | int
        # Clean the text (remove commas, spaces)
        cleaned = text.replace(",", "").replace(" ", "").strip()

        # Try to parse as integer first
        try:
            num = int(cleaned)
        except ValueError:
            # Try as float
            num = float(cleaned)

        locale = _parse_language_to_locale(lang)
        return num2words(num, lang=locale)
    except (NotImplementedError, ValueError, TypeError) as e:
        logger.warning(f"Failed to convert '{text}' to cardinal: {e}")
        return text


def normalize_ordinal(text: str, lang: str = "en-us", **kwargs: Any) -> str:
    """Convert number to ordinal form (first, second, third...).

    Args:
        text: Number as string (e.g., "3")
        lang: Language code
        **kwargs: Additional arguments (ignored)

    Returns:
        Ordinal number as words
    """
    try:
        # Clean the text
        cleaned = text.replace(",", "").replace(" ", "").strip()
        num = int(cleaned)

        locale = _parse_language_to_locale(lang)
        return num2words(num, lang=locale, to="ordinal")
    except (NotImplementedError, ValueError, TypeError) as e:
        logger.warning(f"Failed to convert '{text}' to ordinal: {e}")
        return text


def normalize_digits(text: str, lang: str = "en-us", **kwargs: Any) -> str:
    """Convert digits to spoken form (1-2-3 -> one two three).

    Args:
        text: Digits as string (e.g., "123")
        lang: Language code
        **kwargs: Additional arguments (ignored)

    Returns:
        Digits spoken separately
    """
    try:
        # Extract only digits
        digits = re.findall(r"\d", text)
        if not digits:
            return text

        locale = _parse_language_to_locale(lang)
        # Convert each digit separately
        words = []
        for digit in digits:
            try:
                word = num2words(int(digit), lang=locale)
                words.append(word)
            except (NotImplementedError, ValueError, TypeError):
                words.append(digit)

        return " ".join(words)
    except (NotImplementedError, ValueError, TypeError) as e:
        logger.warning(f"Failed to convert '{text}' to digits: {e}")
        return text


def normalize_characters(text: str, lang: str = "en-us", **kwargs: Any) -> str:
    """Spell out text character by character (ABC -> A B C).

    Args:
        text: Text to spell out
        lang: Language code
        **kwargs: Additional arguments (ignored)

    Returns:
        Characters separated by spaces
    """
    # Drop whitespace before spelling out characters
    cleaned = re.sub(r"\s+", "", text)
    return " ".join(cleaned.upper())


def normalize_expletive(text: str, lang: str = "en-us", **kwargs: Any) -> str:
    """Censor expletive (profanity filter).

    Args:
        text: Text to censor
        lang: Language code
        **kwargs: Additional arguments (ignored)

    Returns:
        Censored text ("beep")
    """
    # Simple censoring - replace with "beep"
    return "beep"


def normalize_date(
    text: str, lang: str = "en-us", format_str: str | None = None, **kwargs: Any
) -> str:
    """Convert date to spoken form.

    Args:
        text: Date as string (e.g., "12/31/2024", "2024-12-31")
        lang: Language code
        format_str: Date format ("mdy", "dmy", "ymd", or strftime format)
        **kwargs: Additional arguments (ignored)

    Returns:
        Date in spoken form
    """
    try:
        # Parse the date - try common formats
        date_obj = None
        for fmt in [
            "%m/%d/%Y",  # 12/31/2024
            "%d/%m/%Y",  # 31/12/2024
            "%Y-%m-%d",  # 2024-12-31
            "%Y/%m/%d",  # 2024/12/31
            "%m-%d-%Y",  # 12-31-2024
            "%d.%m.%Y",  # 31.12.2024
        ]:
            try:
                date_obj = datetime.strptime(text, fmt)  # noqa: DTZ007
                break
            except ValueError:
                continue

        if not date_obj:
            logger.warning(f"Could not parse date '{text}'")
            return text

        # Get babel locale
        locale_code = _parse_language_to_locale(lang)
        try:
            locale = Locale.parse(locale_code.replace("_", "-"))
        except (ValueError, TypeError):
            locale = Locale.parse("en")

        # Format based on format_str
        if format_str in ["mdy", "dmy", "ymd"]:
            # Use babel's format_date with appropriate format
            babel_format = "long"  # Default to long format
            return format_date(date_obj, format=babel_format, locale=locale)
        else:
            # Use babel's default long format
            return format_date(date_obj, format="long", locale=locale)

    except (NotImplementedError, ValueError, TypeError) as e:
        logger.warning(f"Failed to convert '{text}' to date: {e}")
        return text


def normalize_time(
    text: str, lang: str = "en-us", format_str: str | None = None, **kwargs: Any
) -> str:
    """Convert time to spoken form.

    Args:
        text: Time as string (e.g., "14:30", "2:30 PM")
        lang: Language code
        format_str: Time format (e.g., "hms" for hour-minute-second)
        **kwargs: Additional arguments (ignored)

    Returns:
        Time in spoken form
    """
    try:
        # Parse the time - try common formats
        time_obj = None
        for fmt in [
            "%H:%M:%S",  # 14:30:45
            "%H:%M",  # 14:30
            "%I:%M %p",  # 2:30 PM
            "%I:%M:%S %p",  # 2:30:45 PM
        ]:
            try:
                time_obj = datetime.strptime(text, fmt)  # noqa: DTZ007
                break
            except ValueError:
                continue

        if not time_obj:
            logger.warning(f"Could not parse time '{text}'")
            return text

        # Get babel locale
        locale_code = _parse_language_to_locale(lang)
        try:
            locale = Locale.parse(locale_code.replace("_", "-"))
        except (ValueError, TypeError):
            locale = Locale.parse("en")

        # Format the time
        return format_time(time_obj, format="short", locale=locale)

    except (NotImplementedError, ValueError, TypeError) as e:
        logger.warning(f"Failed to convert '{text}' to time: {e}")
        return text


def normalize_telephone(text: str, lang: str = "en-us", **kwargs: Any) -> str:
    """Format telephone number for speech.

    Args:
        text: Phone number (e.g., "+1-555-0123")
        lang: Language code
        **kwargs: Additional arguments (ignored)

    Returns:
        Phone number formatted for speech
    """
    # Extract digits from phone number
    digits = re.findall(r"\d", text)
    if not digits:
        return text

    # Group digits intelligently
    # For US/international: +1 555 0123 -> "plus one, five five five, oh one two three"
    # Simple approach: just speak each digit
    locale = _parse_language_to_locale(lang)

    words = []
    for i, digit in enumerate(digits):
        # Handle leading + or country code
        if i == 0 and text.strip().startswith("+"):
            words.append("plus")

        # Convert digit
        try:
            if digit == "0":
                words.append("oh" if lang.startswith("en") else num2words(0, lang=locale))
            else:
                words.append(num2words(int(digit), lang=locale))
        except (NotImplementedError, ValueError, TypeError):
            words.append(digit)

    return " ".join(words)


def normalize_address(text: str, lang: str = "en-us", **kwargs: Any) -> str:
    """Format address for speech.

    Args:
        text: Address text
        lang: Language code
        **kwargs: Additional arguments (ignored)

    Returns:
        Address formatted for speech (basic implementation)
    """
    # Basic implementation - just return as-is
    # A full implementation would parse street numbers, abbreviations, etc.
    return text


def normalize_unit(text: str, lang: str = "en-us", **kwargs: Any) -> str:
    """Convert units to spoken form (5kg -> five kilograms).

    Args:
        text: Text with units (e.g., "5kg", "10m")
        lang: Language code
        **kwargs: Additional arguments (ignored)

    Returns:
        Units in spoken form
    """
    # Simple pattern: number + unit
    match = re.match(r"^([\d.]+)\s*([a-zA-Z]+)$", text.strip())
    if not match:
        return text

    number_str, unit = match.groups()

    try:
        # Convert number to words
        num = float(number_str)
        locale = _parse_language_to_locale(lang)
        number_words = num2words(num, lang=locale)

        # Common unit expansions
        unit_map = {
            "kg": "kilograms" if num != 1 else "kilogram",
            "g": "grams" if num != 1 else "gram",
            "m": "meters" if num != 1 else "meter",
            "cm": "centimeters" if num != 1 else "centimeter",
            "mm": "millimeters" if num != 1 else "millimeter",
            "km": "kilometers" if num != 1 else "kilometer",
            "lb": "pounds" if num != 1 else "pound",
            "oz": "ounces" if num != 1 else "ounce",
            "ft": "feet" if num != 1 else "foot",
            "in": "inches" if num != 1 else "inch",
            "mi": "miles" if num != 1 else "mile",
            "l": "liters" if num != 1 else "liter",
            "ml": "milliliters" if num != 1 else "milliliter",
        }

        unit_word = unit_map.get(unit.lower(), unit)
        return f"{number_words} {unit_word}"

    except (NotImplementedError, ValueError, TypeError) as e:
        logger.warning(f"Failed to convert '{text}' to unit: {e}")
        return text


def normalize_fraction(text: str, lang: str = "en-us", **kwargs: Any) -> str:
    """Convert fraction to spoken form (1/2 -> one half).

    Args:
        text: Fraction as string (e.g., "1/2", "3/4")
        lang: Language code
        **kwargs: Additional arguments (ignored)

    Returns:
        Fraction in spoken form
    """
    try:
        # Parse fraction
        match = re.match(r"^(\d+)\s*/\s*(\d+)$", text.strip())
        if not match:
            return text

        numerator = int(match.group(1))
        denominator = int(match.group(2))

        locale = _parse_language_to_locale(lang)

        # Convert numerator to cardinal
        num_words = num2words(numerator, lang=locale)

        # Convert denominator to ordinal (for "half", "third", "quarter")
        if denominator == 2:
            denom_words = "half" if numerator == 1 else "halves"
        elif denominator == 4:
            denom_words = "quarter" if numerator == 1 else "quarters"
        else:
            denom_words = num2words(denominator, lang=locale, to="ordinal")
            if numerator > 1:
                denom_words += "s"

        if numerator == 1:
            return f"one {denom_words}"
        else:
            return f"{num_words} {denom_words}"

    except (NotImplementedError, ValueError, TypeError) as e:
        logger.warning(f"Failed to convert '{text}' to fraction: {e}")
        return text


# Main normalization dispatcher
NORMALIZERS: dict[str, Callable[..., str]] = {
    "cardinal": normalize_cardinal,
    "number": normalize_cardinal,  # alias
    "ordinal": normalize_ordinal,
    "digits": normalize_digits,
    "characters": normalize_characters,
    "expletive": normalize_expletive,
    "date": normalize_date,
    "time": normalize_time,
    "telephone": normalize_telephone,
    "address": normalize_address,
    "unit": normalize_unit,
    "fraction": normalize_fraction,
}


def normalize_say_as(
    text: str,
    interpret_as: str,
    lang: str = "en-us",
    format_str: str | None = None,
    detail: str | None = None,
) -> str:
    """Normalize text based on say-as interpret-as type.

    Args:
        text: Text to normalize
        interpret_as: Interpretation type (cardinal, ordinal, date, etc.)
        lang: Language code (default: "en-us")
        format_str: Format attribute (e.g., "mdy" for dates)
        detail: Detail attribute (additional formatting info)

    Returns:
        Normalized text ready for TTS

    Example:
        >>> normalize_say_as("123", "cardinal", lang="en-us")
        'one hundred twenty-three'
        >>> normalize_say_as("3", "ordinal", lang="en-us")
        'third'
        >>> normalize_say_as("ABC", "characters")
        'A B C'
    """
    normalizer = NORMALIZERS.get(interpret_as.lower())

    if normalizer is None:
        logger.warning(f"Unknown interpret-as type '{interpret_as}', returning original text")
        return text

    try:
        return normalizer(text, lang=lang, format_str=format_str, detail=detail)
    except (NotImplementedError, ValueError, TypeError) as e:
        logger.error(f"Error normalizing '{text}' as '{interpret_as}': {e}")
        return text
