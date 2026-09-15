"""Tests for built-in short-sentence phrase catalogs."""

import pytest

from pykokoro.constants import ESPEAK_ONLY_LANGUAGES, SUPPORTED_LANGUAGES
from pykokoro.short_sentence_phrases import (
    BUILTIN_SHORT_SENTENCE_PHRASES,
    resolve_short_sentence_phrase_language,
    resolve_short_sentence_phrase_set,
)


@pytest.mark.parametrize(
    "language",
    sorted(set(SUPPORTED_LANGUAGES) | set(ESPEAK_ONLY_LANGUAGES)),
)
def test_builtin_phrase_catalog_covers_every_advertised_language(language: str) -> None:
    phrase_set, phrase_language, source = resolve_short_sentence_phrase_set(language)

    assert phrase_set is not None
    assert phrase_language is not None
    assert source == "builtin"


def test_mandarin_alias_resolves_to_chinese_catalog() -> None:
    assert resolve_short_sentence_phrase_language("zh") == "zh"
    assert resolve_short_sentence_phrase_language("cmn") == "zh"
    assert resolve_short_sentence_phrase_language("zh-CN") == "zh"
    assert resolve_short_sentence_phrase_language("zh-TW") == "zh"


def test_european_portuguese_keeps_its_own_catalog() -> None:
    assert resolve_short_sentence_phrase_language("pt") == "pt"
    assert resolve_short_sentence_phrase_language("pt-PT") == "pt-pt"


def test_new_catalogs_have_retry_capacity_for_every_terminal_form() -> None:
    for language, phrase_set in BUILTIN_SHORT_SENTENCE_PHRASES.items():
        if language in {"en", "de"}:
            continue
        assert len(phrase_set.declarative) >= 2
        assert len(phrase_set.question) >= 2
        assert len(phrase_set.exclamation) >= 2
        assert len(phrase_set.ellipsis) >= 2
        assert len(phrase_set.fragment) >= 2


def test_new_catalog_templates_keep_target_at_phrase_end() -> None:
    for language, phrase_set in BUILTIN_SHORT_SENTENCE_PHRASES.items():
        if language in {"en", "de"}:
            continue
        for category in phrase_set.categories():
            for template in category:
                assert template.endswith("{segment}")
                assert template.count("{segment}") == 1
