from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from benchmarks.polynorm_data import (
    POLYNORM_COMMIT,
    PolyNormDataError,
    PolyNormLicenseError,
    default_cache_root,
    load_cases,
    locale_cache_path,
)


def _write_locale_fixture(cache_dir: Path, locale: str, rows: list[str]) -> Path:
    path = locale_cache_path(locale, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_load_cases_parses_and_orders_deterministically(tmp_path: Path) -> None:
    _write_locale_fixture(
        tmp_path,
        "en-US",
        [
            '{"index":"10","category":"Date","original_text":"ten","normalized_text":"ten"}',
            '{"index":"2","category":"Cardinal","original_text":"two","normalized_text":"two"}',
        ],
    )

    cases = load_cases(locales=["en-US"], cache_dir=tmp_path, offline=True)

    assert [case.case_id for case in cases] == ["en-US:2", "en-US:10"]


def test_load_cases_skips_empty_placeholder(tmp_path: Path) -> None:
    _write_locale_fixture(
        tmp_path,
        "en-US",
        [
            '{"index":"1","category":"Date","original_text":"one","normalized_text":"one"}',
            '{"index":"2","category":"Abbreviation","original_text":"","normalized_text":""}',
            '{"index":"3","category":"Date","original_text":"three","normalized_text":"three"}',
        ],
    )

    cases = load_cases(locales=["en-US"], cache_dir=tmp_path, offline=True)

    assert [case.case_id for case in cases] == ["en-US:1", "en-US:3"]


@pytest.mark.parametrize(
    ("original_text", "normalized_text", "invalid_field"),
    [("", "spoken", "original_text"), ("source", "", "normalized_text")],
)
def test_load_cases_rejects_partially_empty_text(
    tmp_path: Path,
    original_text: str,
    normalized_text: str,
    invalid_field: str,
) -> None:
    _write_locale_fixture(
        tmp_path,
        "en-US",
        [
            f'{{"index":"1","category":"Date","original_text":"{original_text}","normalized_text":"{normalized_text}"}}',
        ],
    )

    with pytest.raises(PolyNormDataError, match=invalid_field):
        load_cases(locales=["en-US"], cache_dir=tmp_path, offline=True)


def test_load_cases_rejects_missing_fields(tmp_path: Path) -> None:
    _write_locale_fixture(
        tmp_path,
        "en-US",
        ['{"index":"1","category":"Date","original_text":"x"}'],
    )

    with pytest.raises(PolyNormDataError, match="normalized_text"):
        load_cases(locales=["en-US"], cache_dir=tmp_path, offline=True)


def test_load_cases_rejects_invalid_json(tmp_path: Path) -> None:
    _write_locale_fixture(tmp_path, "en-US", ['{"index":'])

    with pytest.raises(PolyNormDataError, match="Invalid JSON"):
        load_cases(locales=["en-US"], cache_dir=tmp_path, offline=True)


def test_load_cases_validates_locale(tmp_path: Path) -> None:
    with pytest.raises(PolyNormDataError, match="Unsupported PolyNorm locale"):
        load_cases(locales=["xx-YY"], cache_dir=tmp_path, offline=True)


def test_load_cases_filters_category_case_and_limit(tmp_path: Path) -> None:
    _write_locale_fixture(
        tmp_path,
        "en-US",
        [
            '{"index":"1","category":"Date","original_text":"A","normalized_text":"A"}',
            '{"index":"2","category":"Cardinal","original_text":"B","normalized_text":"bee"}',
            '{"index":"3","category":"Cardinal","original_text":"C","normalized_text":"see"}',
        ],
    )

    category_cases = load_cases(
        locales=["en-US"],
        category="Cardinal",
        cache_dir=tmp_path,
        offline=True,
    )
    case_filtered = load_cases(
        case_ids=["en-US:3"],
        cache_dir=tmp_path,
        offline=True,
    )
    limited = load_cases(locales=["en-US"], cache_dir=tmp_path, offline=True, limit=1)

    assert [case.case_id for case in category_cases] == ["en-US:2", "en-US:3"]
    assert [case.case_id for case in case_filtered] == ["en-US:3"]
    assert [case.case_id for case in limited] == ["en-US:1"]


def test_load_cases_offline_requires_existing_cache(tmp_path: Path) -> None:
    with pytest.raises(PolyNormDataError, match="Offline mode requested"):
        load_cases(locales=["en-US"], cache_dir=tmp_path, offline=True)


def test_first_download_requires_explicit_license_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (
        b'{"index":"1","category":"Date","original_text":"05/20/2023",'
        b'"normalized_text":"May twentieth"}\n'
    )

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self) -> bytes:
            return payload

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _Response())

    with pytest.raises(PolyNormLicenseError):
        load_cases(locales=["en-US"], cache_dir=tmp_path)

    cases = load_cases(locales=["en-US"], cache_dir=tmp_path, accept_license=True)

    assert [case.case_id for case in cases] == ["en-US:1"]


def test_cache_root_is_commit_scoped(tmp_path: Path) -> None:
    root = default_cache_root(tmp_path)

    assert root.parts[-2:] == ("polynorm", POLYNORM_COMMIT)
