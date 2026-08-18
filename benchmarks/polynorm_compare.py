from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two PolyNorm benchmark summaries.")
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--allow-incompatible", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    comparison = compare_summaries(before, after, allow_incompatible=args.allow_incompatible)
    print(render_comparison(comparison))
    return 0


def compare_summaries(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    allow_incompatible: bool = False,
) -> dict[str, Any]:
    if not allow_incompatible:
        for key in ("schema_version",):
            if before.get(key) != after.get(key):
                raise ValueError(f"Incompatible benchmark summaries: {key} differs.")
        if before.get("dataset", {}).get("commit") != after.get("dataset", {}).get("commit"):
            raise ValueError("Incompatible benchmark summaries: dataset commit differs.")
        if before.get("config_hash") != after.get("config_hash"):
            raise ValueError("Incompatible benchmark summaries: config hash differs.")

    return {
        "new_failures": sorted(
            set(after.get("failure_ids", [])) - set(before.get("failure_ids", []))
        ),
        "resolved_failures": sorted(
            set(before.get("failure_ids", [])) - set(after.get("failure_ids", []))
        ),
        "remaining_failures": sorted(
            set(before.get("failure_ids", [])) & set(after.get("failure_ids", []))
        ),
        "delta_metrics": {
            "raw_phoneme_exact": _delta_metric(before, after, "raw_phoneme_exact"),
            "semantic_phoneme_exact": _delta_metric(before, after, "semantic_phoneme_exact"),
            "token_exact": _delta_metric(before, after, "token_exact"),
            "token_error_rate": _delta_metric(before, after, "token_error_rate"),
        },
        "per_locale": _delta_bucket_map(before.get("per_locale", {}), after.get("per_locale", {})),
        "per_category": _delta_bucket_map(
            before.get("per_category", {}),
            after.get("per_category", {}),
        ),
        "likely_owner": _delta_count_map(
            before.get("likely_owner", {}),
            after.get("likely_owner", {}),
        ),
    }


def render_comparison(comparison: dict[str, Any]) -> str:
    lines = [
        "PolyNorm benchmark comparison",
        "",
        f"new failures: {len(comparison['new_failures'])}",
        f"resolved failures: {len(comparison['resolved_failures'])}",
        f"remaining failures: {len(comparison['remaining_failures'])}",
        "",
    ]
    for name, delta in comparison["delta_metrics"].items():
        lines.append(f"delta {name}: {delta:+g}")
    for section_name in ("per_locale", "per_category", "likely_owner"):
        lines.append("")
        lines.append(f"{section_name}:")
        for key, delta in sorted(comparison[section_name].items()):
            lines.append(f"  {key}: {delta:+g}")
    return "\n".join(lines)


def _delta_metric(before: dict[str, Any], after: dict[str, Any], name: str) -> float:
    return float(after.get("metrics", {}).get(name, 0)) - float(
        before.get("metrics", {}).get(name, 0)
    )


def _delta_bucket_map(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, float]:
    keys = set(before) | set(after)
    return {
        key: float(after.get(key, {}).get("semantic_phoneme_exact_rate", 0))
        - float(before.get(key, {}).get("semantic_phoneme_exact_rate", 0))
        for key in keys
    }


def _delta_count_map(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, float]:
    keys = set(before) | set(after)
    return {key: float(after.get(key, 0) - before.get(key, 0)) for key in keys}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
