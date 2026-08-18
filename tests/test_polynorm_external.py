from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.polynorm_data import PolyNormDataError, load_cases, locale_cache_path
from benchmarks.polynorm_phoneme import main as polynorm_main


def _write_cached_row(cache_dir: Path) -> None:
    path = locale_cache_path("en-US", cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"index":"1","category":"Date","original_text":"2","normalized_text":"two"}\n',
        encoding="utf-8",
    )


def _load_cached_cases_or_skip(cache_dir: Path):
    try:
        return load_cases(locales=["en-US"], cache_dir=cache_dir, offline=True, limit=1)
    except PolyNormDataError as exc:  # pragma: no cover - exercised by pytest skip
        pytest.skip(str(exc))


@pytest.mark.slow
@pytest.mark.polynorm
@pytest.mark.external_data
def test_external_bridge_skips_when_cache_is_missing(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception, match="Offline mode requested"):
        _load_cached_cases_or_skip(tmp_path)


@pytest.mark.slow
@pytest.mark.polynorm
@pytest.mark.external_data
def test_external_bridge_runs_from_existing_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    results_dir = tmp_path / "results"
    _write_cached_row(cache_dir)

    exit_code = polynorm_main(
        [
            "--offline",
            "--locale",
            "en-US",
            "--limit",
            "1",
            "--results-dir",
            str(results_dir),
            "--cache-dir",
            str(cache_dir),
        ]
    )

    assert exit_code == 0
    assert (results_dir / "summary.json").exists()
