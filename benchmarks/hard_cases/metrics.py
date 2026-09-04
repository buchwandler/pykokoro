from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return asdict(value) if is_dataclass(value) else dict(value)


def summarize(
    results: Iterable[Any], *, environment: dict[str, Any] | None = None
) -> dict[str, Any]:
    items = list(results)
    active = [item for item in items if not getattr(item, "quarantined", False)]
    failures = [item for item in active if getattr(item, "failed", False)]
    passed = [item for item in active if not getattr(item, "failed", False)]

    def rate(values: list[Any], predicate: str) -> float | None:
        selected = [
            getattr(item, predicate)
            for item in values
            if getattr(item, predicate, None) is not None
        ]
        return sum(selected) / len(selected) if selected else None

    return {
        "schema_version": 1,
        "counts": {
            "cases_total": len(items),
            "cases_passed": len(passed),
            "cases_failed": len(failures),
            "cases_quarantined": len(items) - len(active),
            "cases_skipped": 0,
        },
        "metrics": {
            "spoken_text_exact": rate(active, "spoken_text_pass"),
            "raw_phoneme_exact": rate(active, "raw_phoneme_exact"),
            "semantic_phoneme_exact": rate(active, "semantic_phoneme_exact"),
            "token_exact": rate(active, "token_exact"),
            "critical_pronunciation_pass": rate(active, "critical_pronunciation_pass"),
            "segment_plan_pass": rate(active, "plan_pass"),
        },
        "failure_ids": sorted(f"{item.level}:{item.case_id}" for item in failures),
        "likely_owner": dict(Counter(item.likely_owner for item in failures)),
        "per_language": _group_rates(active, "language"),
        "per_locale": _group_rates(active, "locale"),
        "per_category": _group_rates(active, "category"),
        "environment": environment or {},
    }


def _group_rates(items: list[Any], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Any]] = {}
    for item in items:
        key = getattr(item, field, None) or "none"
        groups.setdefault(str(key), []).append(item)
    return {
        key: {
            "cases": len(group),
            "failed": sum(bool(item.failed) for item in group),
            "semantic_phoneme_exact_rate": _safe_rate(group, "semantic_phoneme_exact"),
        }
        for key, group in sorted(groups.items())
    }


def _safe_rate(items: list[Any], field: str) -> float | None:
    values = [getattr(item, field) for item in items if getattr(item, field, None) is not None]
    return sum(values) / len(values) if values else None


def result_dict(result: Any) -> dict[str, Any]:
    value = _as_dict(result)
    # Text and phoneme arrays are useful in cases.jsonl but can be omitted from compact summaries.
    return _jsonable(value)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["result_dict", "summarize"]
