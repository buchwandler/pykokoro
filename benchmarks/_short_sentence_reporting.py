"""Model-free reporting helpers for short-sentence benchmark results."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

SUCCESSFUL_CUT_STRATEGIES = frozenset(
    {
        "energy-valley",
        "timestamp-smooth",
        "timestamp-smooth-relaxed",
        "timestamp-anchor",
    }
)


def classify_short_sentence_outcome(
    configured_policy: str,
    metadata: dict[str, object] | None,
) -> str:
    """Classify the final result of one rendered short-sentence case."""
    if configured_policy == "disabled":
        return "disabled"
    if not metadata:
        return "unresolved"

    kind = metadata.get("kind")
    fallback_used = metadata.get("fallback_used")
    cut_strategy = metadata.get("cut_strategy")
    failure_reason = metadata.get("timing_failure_reason") or metadata.get("cut_failure_reason")

    if configured_policy == "wrap" and kind == "wrap":
        return "wrap-configured"
    if kind == "wrap":
        if failure_reason == "no-localized-phrase-catalog":
            return "wrap-no-localized-catalog"
        return "wrap-runtime-resolution"
    if fallback_used == "wrap" or cut_strategy == "wrap":
        return "wrap-fallback"
    if cut_strategy in SUCCESSFUL_CUT_STRATEGIES:
        if fallback_used == "phrase":
            return "phrase-cut-retry"
        return "phrase-cut-initial"
    return "unresolved"


def normalize_short_sentence_row(row: dict[str, Any]) -> dict[str, Any]:
    """Add final-case fields to a raw latency row without discarding raw fields."""
    normalized = dict(row)
    policy = str(row.get("configured_policy", row.get("policy", "")))
    metadata = row.get("short_sentence_metadata")
    if not isinstance(metadata, dict):
        metadata = {
            key: row[key]
            for key in (
                "kind",
                "fallback_used",
                "cut_strategy",
                "timing_failure_reason",
                "cut_failure_reason",
                "failure_stage",
            )
            if key in row
        }
    outcome = classify_short_sentence_outcome(policy, metadata)
    normalized.update(
        {
            "configured_policy": policy,
            "final_outcome": outcome,
            "phrase_cut_succeeded": outcome in {"phrase-cut-initial", "phrase-cut-retry"},
            "phrase_cut_used_retry": outcome == "phrase-cut-retry",
            "actual_cut_strategy": metadata.get("cut_strategy"),
            "fallback_used": metadata.get("fallback_used", row.get("fallback_used")),
            "retry_attempts": metadata.get("retry_attempts", row.get("retry_attempts", 0)),
            "cut_failure_reason": metadata.get("cut_failure_reason", row.get("cut_failure_reason")),
            "timing_failure_reason": metadata.get(
                "timing_failure_reason", row.get("timing_failure_reason")
            ),
            "failure_stage": metadata.get("failure_stage", row.get("failure_stage")),
        }
    )
    return normalized


def _integer(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def aggregate_short_sentence_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate normalized final cases and independent attempt-level counters."""
    normalized_rows = [normalize_short_sentence_row(row) for row in rows]
    policies: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "cases": 0,
            "phrase_cut_successes": 0,
            "initial_successes": 0,
            "retry_successes": 0,
            "wrap_fallbacks": 0,
            "unresolved": 0,
        }
    )
    actual_strategies: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    attempts = {
        "initial_phrase_attempts": 0,
        "phrase_retries": 0,
        "cut_failures": 0,
        "wrap_fallback_renders": 0,
        "onnx_calls": 0,
    }
    for row in normalized_rows:
        policy = row["configured_policy"]
        outcome = row["final_outcome"]
        summary = policies[policy]
        summary["cases"] += 1
        if row["phrase_cut_succeeded"]:
            summary["phrase_cut_successes"] += 1
            if outcome == "phrase-cut-initial":
                summary["initial_successes"] += 1
            else:
                summary["retry_successes"] += 1
            strategy = row.get("actual_cut_strategy")
            if strategy in SUCCESSFUL_CUT_STRATEGIES:
                actual_strategies[str(strategy)] += 1
        elif outcome == "wrap-fallback":
            summary["wrap_fallbacks"] += 1
        elif outcome == "unresolved":
            summary["unresolved"] += 1

        reason = row.get("timing_failure_reason") or row.get("cut_failure_reason")
        if isinstance(reason, str) and reason:
            failure_reasons[reason] += 1
        for key in attempts:
            attempts[key] += _integer(row, key)

    for summary in policies.values():
        cases = summary["cases"]
        summary["phrase_cut_success_rate"] = (
            summary["phrase_cut_successes"] / cases if cases else 0.0
        )
    return {
        "cases": dict(policies),
        "actual_cut_strategies": dict(actual_strategies),
        "failure_reasons": dict(failure_reasons),
        "attempts": attempts,
        "rows": normalized_rows,
    }


def build_short_sentence_summary(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable summary while retaining normalized raw rows."""
    summary = aggregate_short_sentence_rows(rows)
    result: dict[str, Any] = {
        "schema": "pykokoro.short-sentence-benchmark.v1",
        "cases": summary["cases"],
        "actual_cut_strategies": summary["actual_cut_strategies"],
        "failure_reasons": summary["failure_reasons"],
        "attempts": summary["attempts"],
        "rows": summary["rows"],
    }
    if metadata:
        result["run"] = dict(metadata)
    return result


def _percentage(value: int, total: int) -> str:
    return f"{value / total * 100:.1f}%" if total else "-"


def format_short_sentence_table(summary: dict[str, Any]) -> str:
    """Format a compact human-readable report from ``build_short_sentence_summary``."""
    lines = ["Short-sentence benchmark summary"]
    run = summary.get("run", {})
    if isinstance(run, dict):
        for label, key in (("Voice", "voice"), ("Model", "model")):
            if run.get(key) is not None:
                lines.append(f"{label}: {run[key]}")
    lines.extend(
        [
            "",
            "Configured policy      Cases  Phrase cut  Initial  Retry  Wrap fallback  Unresolved",
        ]
    )
    for policy, values in summary.get("cases", {}).items():
        cases = int(values["cases"])
        lines.append(
            f"{policy:<22}{cases:>5}  {values['phrase_cut_successes']:>11}  "
            f"{values['initial_successes']:>7}  {values['retry_successes']:>5}  "
            f"{values['wrap_fallbacks']:>13}  {values['unresolved']:>10}"
        )
    lines.extend(["", "Phrase-cut success rate"])
    for policy, values in summary.get("cases", {}).items():
        if values["phrase_cut_successes"]:
            lines.append(
                f"{policy:<22}{values['phrase_cut_successes']}/{values['cases']}  "
                f"{_percentage(values['phrase_cut_successes'], values['cases'])}"
            )
    lines.extend(["", "Actual successful cut strategies"])
    for strategy, count in summary.get("actual_cut_strategies", {}).items():
        lines.append(f"{strategy:<30}{count}")
    lines.extend(["", "Phrase attempt accounting"])
    labels = {
        "initial_phrase_attempts": "Initial phrase attempts",
        "phrase_retries": "Phrase retries",
        "cut_failures": "Cut failures",
        "wrap_fallback_renders": "Wrap fallback renders",
    }
    for key, label in labels.items():
        lines.append(f"{label:<30}{summary.get('attempts', {}).get(key, 0)}")
    lines.extend(["", "Failure reasons"])
    for reason, count in summary.get("failure_reasons", {}).items():
        lines.append(f"{reason:<30}{count}")
    return "\n".join(lines)


__all__ = [
    "SUCCESSFUL_CUT_STRATEGIES",
    "aggregate_short_sentence_rows",
    "build_short_sentence_summary",
    "classify_short_sentence_outcome",
    "format_short_sentence_table",
    "normalize_short_sentence_row",
]
