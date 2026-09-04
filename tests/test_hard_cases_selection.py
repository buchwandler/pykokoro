from __future__ import annotations

import pytest

from benchmarks.hard_cases.schema import HardCaseError
from benchmarks.hard_cases.selection import list_categories, select_cases


def test_selection_filters_category_and_limit() -> None:
    cases = select_cases(language="de", category="compounds", limit=2)
    assert len(cases) == 2
    assert all(case.category == "compounds" for case in cases)


def test_selection_rejects_unknown_case() -> None:
    with pytest.raises(HardCaseError, match="unknown case"):
        select_cases(case_id="does_not_exist")


def test_categories_are_stable() -> None:
    assert list_categories(language="en") == tuple(sorted(list_categories(language="en")))
