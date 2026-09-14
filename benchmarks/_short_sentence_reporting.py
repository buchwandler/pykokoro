"""Model-free reporting helpers for short-sentence benchmark results."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any

SUCCESSFUL_CUT_STRATEGIES = frozenset(
    {"energy-valley", "timestamp-smooth", "timestamp-smooth-relaxed", "timestamp-anchor"}
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
    cut_strategy = metadata.get("cut_strategy", metadata.get("actual_cut_strategy"))
    failure_reason = metadata.get("timing_failure_reason") or metadata.get("cut_failure_reason")
    if configured_policy == "wrap" and kind == "wrap":
        return "wrap-configured"
    if kind == "wrap":
        return (
            "wrap-no-localized-catalog"
            if failure_reason == "no-localized-phrase-catalog"
            else "wrap-runtime-resolution"
        )
    if fallback_used == "wrap" or cut_strategy == "wrap":
        return "wrap-fallback"
    if cut_strategy in SUCCESSFUL_CUT_STRATEGIES:
        return "phrase-cut-retry" if fallback_used == "phrase" else "phrase-cut-initial"
    return "unresolved"


def _attempt_history(metadata: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("short_sentence_attempts", row.get("attempt_history", []))
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def normalize_short_sentence_row(row: dict[str, Any]) -> dict[str, Any]:
    """Add final-case and attempt-level fields without discarding raw fields."""
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
                "phrase_template",
                "phrase_language",
                "cutter",
                "configured_cutter",
            )
            if key in row
        }
    history = _attempt_history(metadata, row)
    outcome = classify_short_sentence_outcome(policy, metadata)
    successful_attempts = [item for item in history if item.get("succeeded") is True]
    failed_attempts = [item for item in history if item.get("succeeded") is False]
    success = successful_attempts[0] if successful_attempts else None
    attempt_count = len(history)
    if not attempt_count:
        attempt_count = _integer(row, "initial_phrase_attempts") + _integer(row, "phrase_retries")
    success_ordinal = success.get("ordinal") if success else None
    if success_ordinal is None and success is not None:
        success_ordinal = _integer(success, "attempt") + 1
    normalized.update(
        {
            "configured_policy": policy,
            "final_outcome": outcome,
            "phrase_cut_succeeded": outcome in {"phrase-cut-initial", "phrase-cut-retry"},
            "phrase_cut_used_retry": outcome == "phrase-cut-retry",
            "phrase_attempt_count": attempt_count,
            "failed_phrase_attempt_count": len(failed_attempts),
            "success_attempt_ordinal": success_ordinal,
            "success_template": (success or {}).get(
                "phrase_template", metadata.get("phrase_template")
            )
            if success is not None or outcome.startswith("phrase-cut")
            else None,
            "configured_cutter": metadata.get("configured_cutter", metadata.get("cutter")),
            "actual_cut_strategy": metadata.get(
                "actual_cut_strategy", metadata.get("cut_strategy")
            ),
            "fallback_used": metadata.get("fallback_used", row.get("fallback_used")),
            "retry_attempts": metadata.get("retry_attempts", row.get("retry_attempts", 0)),
            "cut_failure_reason": metadata.get("cut_failure_reason", row.get("cut_failure_reason")),
            "timing_failure_reason": metadata.get(
                "timing_failure_reason", row.get("timing_failure_reason")
            ),
            "failure_stage": metadata.get("failure_stage", row.get("failure_stage")),
            "attempt_history": history,
        }
    )
    return normalized


def _integer(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def aggregate_short_sentence_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate final cases and independent attempt-level counters."""
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
    success_by_ordinal: dict[str, Counter[str]] = defaultdict(Counter)
    success_by_template: Counter[str] = Counter()
    failure_by_stage: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    attempts = {
        "initial_phrase_attempts": 0,
        "phrase_retries": 0,
        "cut_failures": 0,
        "wrap_fallback_renders": 0,
        "onnx_calls": 0,
    }
    budgets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "cases": 0,
            "phrase_cut_successes": 0,
            "wrap_fallbacks": 0,
            "onnx_calls": [],
            "wall_ms": [],
        }
    )
    for row in normalized_rows:
        policy = str(row["configured_policy"])
        outcome = str(row["final_outcome"])
        summary = policies[policy]
        summary["cases"] += 1
        if row["phrase_cut_succeeded"]:
            summary["phrase_cut_successes"] += 1
            if outcome == "phrase-cut-initial":
                summary["initial_successes"] += 1
            else:
                summary["retry_successes"] += 1
        elif outcome == "wrap-fallback":
            summary["wrap_fallbacks"] += 1
        elif outcome == "unresolved":
            summary["unresolved"] += 1

        history = row.get("attempt_history", [])
        for attempt in history if isinstance(history, list) else []:
            if attempt.get("succeeded") is True:
                strategy = attempt.get("actual_cut_strategy", attempt.get("cut_strategy"))
                if isinstance(strategy, str) and strategy:
                    actual_strategies[strategy] += 1
                    success_by_ordinal[policy][
                        str(attempt.get("ordinal", _integer(attempt, "attempt") + 1))
                    ] += 1
                template = attempt.get("phrase_template")
                if isinstance(template, str) and template:
                    success_by_template[template] += 1
            else:
                stage = attempt.get("failure_stage")
                reason = attempt.get("failure_reason")
                if isinstance(stage, str) and stage:
                    failure_by_stage[stage] += 1
                if isinstance(reason, str) and reason:
                    failure_reasons[reason] += 1
        if not history:
            strategy = row.get("actual_cut_strategy")
            if isinstance(strategy, str) and strategy in SUCCESSFUL_CUT_STRATEGIES:
                actual_strategies[strategy] += 1
            reason = row.get("timing_failure_reason") or row.get("cut_failure_reason")
            if isinstance(reason, str) and reason:
                failure_reasons[reason] += 1
            stage = row.get("failure_stage")
            if isinstance(stage, str) and stage:
                failure_by_stage[stage] += 1

        for key in attempts:
            attempts[key] += _integer(row, key)
        budget = row.get("fallback_retries", row.get("phrase_fallback_tries"))
        if isinstance(budget, int) and not isinstance(budget, bool):
            bucket = budgets[str(budget)]
            bucket["cases"] += 1
            bucket["phrase_cut_successes"] += int(bool(row["phrase_cut_succeeded"]))
            bucket["wrap_fallbacks"] += int(outcome == "wrap-fallback")
            bucket["onnx_calls"].append(_integer(row, "onnx_calls"))
            wall_ms = row.get("wall_ms")
            if isinstance(wall_ms, (int, float)):
                bucket["wall_ms"].append(float(wall_ms))

    for summary in policies.values():
        cases = summary["cases"]
        summary["phrase_cut_success_rate"] = (
            summary["phrase_cut_successes"] / cases if cases else 0.0
        )
    retry_comparison: dict[str, dict[str, Any]] = {}
    for budget, value in budgets.items():
        cases = value["cases"]
        retry_comparison[budget] = {
            "fallback_retries": int(budget),
            "max_phrase_attempts": int(budget) + 1,
            "cases": cases,
            "phrase_cut_successes": value["phrase_cut_successes"],
            "phrase_cut_success_rate": value["phrase_cut_successes"] / cases if cases else 0.0,
            "wrap_fallbacks": value["wrap_fallbacks"],
            "mean_onnx_calls": sum(value["onnx_calls"]) / len(value["onnx_calls"])
            if value["onnx_calls"]
            else None,
            "p50_wall_ms": median(value["wall_ms"]) if value["wall_ms"] else None,
        }
    return {
        "cases": dict(policies),
        "actual_cut_strategies": dict(actual_strategies),
        "success_by_attempt_ordinal": {
            policy: dict(counts) for policy, counts in success_by_ordinal.items()
        },
        "success_by_strategy": dict(actual_strategies),
        "success_by_template": dict(success_by_template),
        "failure_by_stage": dict(failure_by_stage),
        "failure_reasons": dict(failure_reasons),
        "attempts": attempts,
        "retry_budget_comparison": retry_comparison,
        "rows": normalized_rows,
    }


def build_short_sentence_summary(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable schema-v2 summary retaining normalized rows."""
    summary = aggregate_short_sentence_rows(rows)
    result: dict[str, Any] = {"schema": "pykokoro.short-sentence-benchmark.v2", **summary}
    if metadata:
        result["run"] = dict(metadata)
    return result


def _percentage(value: int, total: int) -> str:
    return f"{value / total * 100:.1f}%" if total else "-"


def format_short_sentence_table(summary: dict[str, Any]) -> str:
    """Format a compact human-readable report with case and attempt sections."""
    lines = ["Short-sentence benchmark summary"]
    run = summary.get("run", {})
    if isinstance(run, dict):
        if run.get("voice") is not None:
            lines.append(f"Voice: {run['voice']}")
        if run.get("model") is not None:
            lines.append(f"Model: {run['model']}")
        if run.get("fallback_retries") is not None:
            retries = int(run["fallback_retries"])
            lines.append(f"Fallback retries: {retries}")
            lines.append(f"Maximum phrase attempts per short segment: {retries + 1}")
    lines.extend(
        ["", "Configured policy      Cases  Phrase cut  Initial  Retry  Wrap fallback  Unresolved"]
    )
    for policy, values in summary.get("cases", {}).items():
        lines.append(
            f"{policy:<22}{values['cases']:>5}  {values['phrase_cut_successes']:>11}  "
            f"{values['initial_successes']:>7}  {values['retry_successes']:>5}  "
            f"{values['wrap_fallbacks']:>13}  {values['unresolved']:>10}"
        )
    lines.extend(["", "Phrase-cut success rate"])
    for policy, values in summary.get("cases", {}).items():
        lines.append(
            f"{policy:<22}{values['phrase_cut_successes']}/{values['cases']}  {_percentage(values['phrase_cut_successes'], values['cases'])}"
        )
    lines.extend(["", "Success by attempt ordinal"])
    for policy, values in summary.get("success_by_attempt_ordinal", {}).items():
        lines.append(
            f"{policy:<22}"
            + "  ".join(
                f"#{ordinal}: {count}"
                for ordinal, count in sorted(values.items(), key=lambda item: int(item[0]))
            )
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
    lines.extend(["", "Failure stage"])
    for stage, count in summary.get("failure_by_stage", {}).items():
        lines.append(f"{stage:<30}{count}")
    lines.extend(["", "Failure reasons"])
    for reason, count in summary.get("failure_reasons", {}).items():
        lines.append(f"{reason:<30}{count}")
    if summary.get("retry_budget_comparison"):
        lines.extend(["", "Retry-budget comparison"])
        lines.append(
            "Retries  Max attempts  Cases  Phrase success  Wrap fallback  Mean ONNX calls  p50 ms"
        )
        for value in summary["retry_budget_comparison"].values():
            lines.append(
                f"{value['fallback_retries']:>7}  {value['max_phrase_attempts']:>12}  {value['cases']:>5}  {value['phrase_cut_successes']:>14}  {value['wrap_fallbacks']:>13}  {value['mean_onnx_calls']!s:>16}  {value['p50_wall_ms']!s:>6}"
            )
    return "\n".join(lines)


__all__ = [
    "SUCCESSFUL_CUT_STRATEGIES",
    "aggregate_short_sentence_rows",
    "build_short_sentence_summary",
    "classify_short_sentence_outcome",
    "format_short_sentence_table",
    "normalize_short_sentence_row",
]
