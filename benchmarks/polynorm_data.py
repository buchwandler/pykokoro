from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

POLYNORM_REPOSITORY = "https://github.com/apple/ml-speech-polynorm-bench"
POLYNORM_COMMIT = "f3c67e047bea6b7c40bc2466c0fdaad51d8ce67d"
POLYNORM_LICENSE = "CC BY-NC-ND 4.0"
POLYNORM_TO_PYKOKORO_LANGUAGE = {
    "de-DE": "de",
    "en-US": "en-us",
    "es-MX": "es",
    "fr-FR": "fr-fr",
    "it-IT": "it",
}


class PolyNormDataError(RuntimeError):
    """Raised when the external PolyNorm dataset cannot be prepared or parsed."""


class PolyNormLicenseError(PolyNormDataError):
    """Raised when a first download is requested without explicit acknowledgement."""


@dataclass(frozen=True, slots=True)
class PolyNormCase:
    polynorm_locale: str
    index: str
    category: str
    original_text: str
    normalized_text: str

    @property
    def case_id(self) -> str:
        return f"{self.polynorm_locale}:{self.index}"

    @property
    def is_transformation(self) -> bool:
        return self.original_text != self.normalized_text


def default_cache_root(cache_dir: str | Path | None = None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "pykokoro"
    return root / "polynorm" / POLYNORM_COMMIT


def locale_dataset_url(locale: str) -> str:
    _validate_locale(locale)
    return (
        "https://raw.githubusercontent.com/apple/ml-speech-polynorm-bench/"
        f"{POLYNORM_COMMIT}/polynorm_bench/{locale}/{locale}_groundtruth.jsonl"
    )


def locale_cache_path(locale: str, cache_dir: str | Path | None = None) -> Path:
    _validate_locale(locale)
    return default_cache_root(cache_dir) / "polynorm_bench" / locale / f"{locale}_groundtruth.jsonl"


def ensure_locale_dataset(
    locale: str,
    *,
    cache_dir: str | Path | None = None,
    accept_license: bool = False,
    offline: bool = False,
    refresh: bool = False,
    timeout: int = 30,
) -> Path:
    path = locale_cache_path(locale, cache_dir)
    if path.exists() and not refresh:
        return path
    if offline:
        if refresh:
            raise PolyNormDataError("Offline mode cannot refresh the PolyNorm cache.")
        raise PolyNormDataError(f"Offline mode requested but no cached PolyNorm data exists for {locale}.")
    if not accept_license:
        raise PolyNormLicenseError(
            "PolyNorm download requires explicit acknowledgement of the "
            f"{POLYNORM_LICENSE} licence via --accept-license."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(locale_dataset_url(locale), timeout=timeout) as response:
        payload = response.read()
    path.write_bytes(payload)
    return path


def load_cases(
    *,
    locales: Iterable[str] | None = None,
    category: str | None = None,
    case_ids: Iterable[str] | None = None,
    limit: int | None = None,
    cache_dir: str | Path | None = None,
    accept_license: bool = False,
    offline: bool = False,
    refresh: bool = False,
) -> list[PolyNormCase]:
    requested_case_ids = tuple(case_ids or ())
    resolved_locales = _resolve_locales(locales, requested_case_ids)
    wanted_case_ids = set(requested_case_ids)
    cases: list[PolyNormCase] = []

    for locale in resolved_locales:
        path = ensure_locale_dataset(
            locale,
            cache_dir=cache_dir,
            accept_license=accept_license,
            offline=offline,
            refresh=refresh,
        )
        cases.extend(_parse_jsonl(path, locale))

    cases.sort(key=_case_sort_key)
    if category is not None:
        cases = [case for case in cases if case.category == category]
    if wanted_case_ids:
        cases = [case for case in cases if case.case_id in wanted_case_ids]
    if limit is not None:
        if limit < 0:
            raise PolyNormDataError("limit must be non-negative")
        cases = cases[:limit]
    return cases


def _resolve_locales(
    locales: Iterable[str] | None,
    case_ids: Iterable[str],
) -> tuple[str, ...]:
    if locales is not None:
        requested = tuple(locales)
        if not requested:
            raise PolyNormDataError("At least one locale is required when locales is provided.")
        for locale in requested:
            _validate_locale(locale)
        return tuple(dict.fromkeys(requested))

    derived = []
    for case_id in case_ids:
        locale, _, index = case_id.partition(":")
        if not locale or not index:
            raise PolyNormDataError(
                f"Case selector {case_id!r} must use the '<locale>:<index>' format."
            )
        _validate_locale(locale)
        derived.append(locale)
    if derived:
        return tuple(dict.fromkeys(derived))
    return tuple(POLYNORM_TO_PYKOKORO_LANGUAGE)


def _parse_jsonl(path: Path, locale: str) -> list[PolyNormCase]:
    cases: list[PolyNormCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PolyNormDataError(f"Invalid JSON in {path} line {line_number}.") from exc
        if not isinstance(payload, dict):
            raise PolyNormDataError(f"Expected an object in {path} line {line_number}.")
        cases.append(_parse_case_payload(payload, locale=locale, path=path, line_number=line_number))
    return cases


def _parse_case_payload(
    payload: dict[str, object],
    *,
    locale: str,
    path: Path,
    line_number: int,
) -> PolyNormCase:
    required = ("index", "category", "original_text", "normalized_text")
    values: dict[str, str] = {}
    for key in required:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise PolyNormDataError(f"Missing or invalid {key!r} in {path} line {line_number}.")
        values[key] = value
    return PolyNormCase(
        polynorm_locale=locale,
        index=values["index"],
        category=values["category"],
        original_text=values["original_text"],
        normalized_text=values["normalized_text"],
    )


def _validate_locale(locale: str) -> None:
    if locale not in POLYNORM_TO_PYKOKORO_LANGUAGE:
        supported = ", ".join(sorted(POLYNORM_TO_PYKOKORO_LANGUAGE))
        raise PolyNormDataError(f"Unsupported PolyNorm locale {locale!r}. Supported: {supported}")


def _case_sort_key(case: PolyNormCase) -> tuple[str, tuple[int, str], str, str, str]:
    return (
        case.polynorm_locale,
        _index_sort_key(case.index),
        case.category,
        case.original_text,
        case.normalized_text,
    )


def _index_sort_key(value: str) -> tuple[int, str]:
    if value.isdigit():
        return int(value), value
    return 10**9, value
