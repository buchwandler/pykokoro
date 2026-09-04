from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def config_fingerprint(config: Any) -> str:
    payload = config if isinstance(config, dict) else getattr(config, "__dict__", str(config))
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def make_baseline(
    summary: dict[str, Any],
    *,
    case_ids: Iterable[str] = (),
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case_set_fingerprint": hashlib.sha256("\n".join(sorted(case_ids)).encode()).hexdigest(),
        "config_hash": (environment or summary.get("environment", {})).get("config_hash"),
        "environment": environment or summary.get("environment", {}),
        "summary": summary.get("metrics", {}),
        "failure_ids": list(summary.get("failure_ids", ())),
        "quarantine": summary.get("counts", {}).get("cases_quarantined", 0),
    }


def load_baseline(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported hard-cases baseline")
    return value


def save_baseline(path: str | Path, baseline: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(baseline, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


def compare_baselines(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    old = set(before.get("failure_ids", ()))
    new = set(after.get("failure_ids", ()))
    before_metrics = before.get("summary", {})
    after_metrics = after.get("summary", {})
    return {
        "new_failures": sorted(new - old),
        "resolved_failures": sorted(old - new),
        "remaining_failures": sorted(old & new),
        "delta_metrics": {
            key: float(after_metrics[key]) - float(before_metrics[key])
            for key in set(before_metrics) & set(after_metrics)
            if isinstance(before_metrics[key], (int, float))
            and isinstance(after_metrics[key], (int, float))
        },
    }


__all__ = [
    "compare_baselines",
    "config_fingerprint",
    "load_baseline",
    "make_baseline",
    "save_baseline",
]
