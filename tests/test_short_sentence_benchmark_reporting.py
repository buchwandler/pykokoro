from __future__ import annotations

import pytest

pytest.importorskip("pykokoro.stages.doc_parsers.plain")


from benchmarks._short_sentence_reporting import (
    aggregate_short_sentence_rows,
    build_short_sentence_summary,
    classify_short_sentence_outcome,
    format_short_sentence_table,
)


def test_classify_short_sentence_outcomes() -> None:
    assert classify_short_sentence_outcome("disabled", None) == "disabled"
    assert classify_short_sentence_outcome("wrap", {"kind": "wrap"}) == "wrap-configured"
    assert (
        classify_short_sentence_outcome(
            "energy-valley", {"kind": "phrase", "cut_strategy": "energy-valley"}
        )
        == "phrase-cut-initial"
    )
    assert (
        classify_short_sentence_outcome(
            "phrase", {"kind": "phrase", "cut_strategy": "timestamp-smooth"}
        )
        == "phrase-cut-initial"
    )
    assert (
        classify_short_sentence_outcome(
            "phrase",
            {
                "kind": "phrase",
                "cut_strategy": "timestamp-smooth-relaxed",
                "fallback_used": "phrase",
            },
        )
        == "phrase-cut-retry"
    )
    assert (
        classify_short_sentence_outcome(
            "phrase",
            {"kind": "phrase", "cut_strategy": "timestamp-anchor", "fallback_used": "phrase"},
        )
        == "phrase-cut-retry"
    )
    assert (
        classify_short_sentence_outcome(
            "phrase", {"kind": "phrase", "cut_strategy": "wrap", "fallback_used": "wrap"}
        )
        == "wrap-fallback"
    )
    assert (
        classify_short_sentence_outcome(
            "phrase", {"kind": "wrap", "cut_failure_reason": "no-localized-phrase-catalog"}
        )
        == "wrap-no-localized-catalog"
    )
    assert classify_short_sentence_outcome("phrase", None) == "unresolved"


def test_aggregation_keeps_final_case_denominator_separate_from_attempts() -> None:
    rows = [
        {
            "policy": "timestamp-adaptive",
            "short_sentence_metadata": {
                "kind": "phrase",
                "cut_strategy": "timestamp-smooth-relaxed",
                "fallback_used": "phrase",
            },
            "initial_phrase_attempts": 1,
            "phrase_retries": 1,
            "cut_failures": 2,
            "wrap_fallback_renders": 0,
        },
        {
            "policy": "timestamp-adaptive",
            "short_sentence_metadata": {
                "kind": "phrase",
                "cut_strategy": "wrap",
                "fallback_used": "wrap",
                "cut_failure_reason": "left-search-invalid",
            },
            "initial_phrase_attempts": 1,
            "phrase_retries": 0,
            "cut_failures": 1,
            "wrap_fallback_renders": 1,
        },
    ]

    summary = aggregate_short_sentence_rows(rows)
    policy = summary["cases"]["timestamp-adaptive"]
    assert policy["cases"] == 2
    assert policy["phrase_cut_successes"] == 1
    assert policy["retry_successes"] == 1
    assert policy["wrap_fallbacks"] == 1
    assert policy["phrase_cut_success_rate"] == 0.5
    assert summary["attempts"]["cut_failures"] == 3
    assert summary["actual_cut_strategies"] == {"timestamp-smooth-relaxed": 1}
    assert summary["failure_reasons"] == {"left-search-invalid": 1}


def test_summary_and_table_include_run_metadata_and_sections() -> None:
    summary = build_short_sentence_summary(
        [{"policy": "disabled"}], metadata={"voice": "af_sarah", "model": "v1.0"}
    )
    table = format_short_sentence_table(summary)
    assert summary["schema"] == "pykokoro.short-sentence-benchmark.v2"
    assert "Voice: af_sarah" in table
    assert "Configured policy" in table
    assert "Phrase attempt accounting" in table
    assert "Failure reasons" in table


def test_attempt_history_reports_success_ordinal_strategy_template_and_stages() -> None:
    rows = [
        {
            "policy": "energy-valley",
            "fallback_retries": 5,
            "short_sentence_metadata": {
                "kind": "phrase",
                "cut_strategy": "timestamp-smooth",
                "fallback_used": "phrase",
                "short_sentence_attempts": [
                    {
                        "attempt": 0,
                        "ordinal": 1,
                        "succeeded": False,
                        "failure_stage": "timing-alignment",
                        "failure_reason": "timing-model-position-mismatch",
                        "phrase_template": "A {segment}",
                    },
                    {
                        "attempt": 1,
                        "ordinal": 2,
                        "succeeded": True,
                        "actual_cut_strategy": "timestamp-smooth",
                        "phrase_template": "B {segment}",
                    },
                ],
            },
        }
    ]
    summary = aggregate_short_sentence_rows(rows)
    normalized = summary["rows"][0]
    assert normalized["phrase_attempt_count"] == 2
    assert normalized["success_attempt_ordinal"] == 2
    assert summary["success_by_attempt_ordinal"]["energy-valley"] == {"2": 1}
    assert summary["success_by_strategy"] == {"timestamp-smooth": 1}
    assert summary["success_by_template"] == {"B {segment}": 1}
    assert summary["failure_by_stage"] == {"timing-alignment": 1}
    assert summary["retry_budget_comparison"]["5"]["max_phrase_attempts"] == 6
