from __future__ import annotations

import json
import platform
import sys
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

from .baseline import config_fingerprint
from .metrics import result_dict, summarize


def environment_fingerprint(config: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = config or {}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pykokoro": _version("pykokoro"),
        "kokorog2p": _version("kokorog2p"),
        "spokenform": _version("spokenform"),
        "phrasplit": _version("phrasplit"),
        "config_hash": config_fingerprint(payload),
        "config": payload,
    }


def write_reports(
    directory: str | Path, results: Iterable[Any], *, environment: dict[str, Any] | None = None
) -> dict[str, Path]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    items = list(results)
    summary = summarize(items, environment=environment)
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cases_path = destination / "cases.jsonl"
    cases_path.write_text(
        "".join(
            json.dumps(result_dict(item), sort_keys=True, ensure_ascii=False) + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    failures = [item for item in items if getattr(item, "failed", False)]
    failures_jsonl = destination / "failures.jsonl"
    failures_jsonl.write_text(
        "".join(
            json.dumps(result_dict(item), sort_keys=True, ensure_ascii=False) + "\n"
            for item in failures
        ),
        encoding="utf-8",
    )
    failures_md = destination / "failures.md"
    lines = ["# Hard-case failures", "", f"Failures: {len(failures)}", ""]
    for item in failures:
        lines.extend(
            [
                f"## `{item.case_id}`",
                "",
                f"- owner: `{item.likely_owner}`",
                f"- level: `{item.level}`",
            ]
        )
        if item.error:
            lines.append(f"- error: {item.error}")
        for error in item.expectation_errors:
            lines.append(f"- finding: {error}")
        lines.append("")
    failures_md.write_text("\n".join(lines), encoding="utf-8")
    environment_path = destination / "environment.json"
    environment_path.write_text(
        json.dumps(environment or {}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "summary": summary_path,
        "cases": cases_path,
        "failures": failures_jsonl,
        "failures_md": failures_md,
        "environment": environment_path,
    }


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


__all__ = ["environment_fingerprint", "write_reports"]
