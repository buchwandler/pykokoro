from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .schema import (
    LOCALE_LANGUAGE,
    SUPPORTED_LANGUAGES,
    HardCase,
    HardCaseError,
    load_jsonl,
    validate_cases,
)

DATA_ROOT = Path(__file__).with_name("data")


def available_languages() -> tuple[str, ...]:
    return SUPPORTED_LANGUAGES


def available_locales(language: str | None = None) -> tuple[str, ...]:
    locales = tuple(LOCALE_LANGUAGE)
    if language is None:
        return locales
    if language not in SUPPORTED_LANGUAGES:
        raise HardCaseError(f"unsupported language: {language!r}")
    return tuple(locale for locale in locales if LOCALE_LANGUAGE[locale] == language)


def load_cases(
    *,
    language: str | None = None,
    locale: str | None = None,
    category: str | None = None,
    case_id: str | None = None,
    case_ids: Iterable[str] | None = None,
) -> list[HardCase]:
    """Load built-in cases in stable path/id order without network or model access."""
    if language is not None and language not in SUPPORTED_LANGUAGES:
        raise HardCaseError(f"unsupported language: {language!r}")
    if locale is not None:
        if locale not in LOCALE_LANGUAGE:
            raise HardCaseError(f"unsupported locale: {locale!r}")
        if language is not None and LOCALE_LANGUAGE[locale] != language:
            raise HardCaseError(f"locale {locale!r} does not belong to language {language!r}")
    wanted_ids = set(case_ids or ())
    if case_id is not None:
        wanted_ids.add(case_id)
    paths = sorted(DATA_ROOT.glob("**/*.jsonl"))
    cases: list[HardCase] = []
    for path in paths:
        loaded = load_jsonl(path)
        for case in loaded:
            if language is not None and case.language != language:
                continue
            if locale is not None and (
                case.language != LOCALE_LANGUAGE[locale] or case.locale not in (None, locale)
            ):
                continue
            if category is not None and case.category != category:
                continue
            if wanted_ids and case.id not in wanted_ids:
                continue
            cases.append(case)
    cases = list(validate_cases(cases))
    cases.sort(key=lambda item: item.id)
    if wanted_ids:
        missing = wanted_ids.difference(case.id for case in cases)
        if missing:
            raise HardCaseError(f"unknown case id: {sorted(missing)[0]!r}")
    return cases


def load_all_cases() -> list[HardCase]:
    return load_cases()


def case_counts() -> dict[str, int]:
    return {language: len(load_cases(language=language)) for language in SUPPORTED_LANGUAGES}


__all__ = [
    "DATA_ROOT",
    "available_languages",
    "available_locales",
    "case_counts",
    "load_all_cases",
    "load_cases",
]
