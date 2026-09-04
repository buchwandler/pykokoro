from __future__ import annotations

from types import SimpleNamespace

from benchmarks.hard_cases.acoustic import evaluate_audio
from benchmarks.hard_cases.baseline import compare_baselines
from benchmarks.hard_cases.metrics import summarize


def test_baseline_comparison_reports_deltas() -> None:
    before = {"failure_ids": ["frontend:old"], "summary": {"token_exact": 0.5}}
    after = {"failure_ids": ["frontend:new"], "summary": {"token_exact": 0.75}}
    result = compare_baselines(before, after)
    assert result["new_failures"] == ["frontend:new"]
    assert result["resolved_failures"] == ["frontend:old"]
    assert result["delta_metrics"]["token_exact"] == 0.25


def test_acoustic_health_is_not_human_likeness() -> None:
    result = evaluate_audio(
        SimpleNamespace(audio=[0.0, 0.1], sample_rate=2, word_timings=[], phoneme_segments=[])
    )
    assert result.status == "pass"
    assert result.duration_s == 1.0
    assert not hasattr(result, "human_likeness")


def test_summary_counts_failures() -> None:
    item = SimpleNamespace(
        failed=True,
        quarantined=False,
        level="frontend",
        case_id="en_1",
        language="en",
        locale="en-US",
        category="normalization",
        likely_owner="spokenform",
        spoken_text_pass=False,
        raw_phoneme_exact=False,
        semantic_phoneme_exact=False,
        token_exact=False,
        critical_pronunciation_pass=None,
    )
    assert summarize([item])["counts"]["cases_failed"] == 1
