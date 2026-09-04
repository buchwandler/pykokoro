from __future__ import annotations

from collections.abc import Iterable

from .data import load_cases
from .schema import HardCase, HardCaseError


def select_cases(
    *,
    language: str | None = None,
    locale: str | None = None,
    category: str | None = None,
    case_id: str | None = None,
    case_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[HardCase]:
    if limit is not None and limit < 0:
        raise HardCaseError("limit must be non-negative")
    cases = load_cases(
        language=language, locale=locale, category=category, case_id=case_id, case_ids=case_ids
    )
    return cases if limit is None else cases[:limit]


def list_categories(*, language: str | None = None, locale: str | None = None) -> tuple[str, ...]:
    return tuple(sorted({case.category for case in select_cases(language=language, locale=locale)}))


__all__ = ["list_categories", "select_cases"]
