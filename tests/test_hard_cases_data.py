from __future__ import annotations

from benchmarks.hard_cases.data import case_counts, load_cases


def test_builtin_corpus_is_large_enough_and_deterministic() -> None:
    assert case_counts() == {"en": 130, "de": 120}
    assert [case.id for case in load_cases(language="en")[:3]] == [
        "en_gb_001",
        "en_gb_002",
        "en_gb_003",
    ]
    assert len({case.id for case in load_cases()}) == 250


def test_builtin_corpus_covers_language_specific_categories() -> None:
    assert "heteronyms" in {case.category for case in load_cases(language="en")}
    assert "compounds" in {case.category for case in load_cases(language="de")}
    assert len(load_cases(locale="en-US")) == 120
    assert len(load_cases(locale="de-DE")) == 120
