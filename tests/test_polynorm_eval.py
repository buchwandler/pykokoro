from __future__ import annotations

import json
from pathlib import Path

from benchmarks.polynorm_compare import compare_summaries
from benchmarks.polynorm_data import PolyNormCase
from benchmarks.polynorm_eval import (
    PhonemeObservation,
    aggregate_results,
    evaluate_case,
    semantic_phoneme_key,
    write_failure_reports,
)


class _FakeHarness:
    def __init__(self, mapping: dict[str, PhonemeObservation | Exception]) -> None:
        self.mapping = mapping

    def phonemize(self, text: str) -> PhonemeObservation:
        observation = self.mapping[text]
        if isinstance(observation, Exception):
            raise observation
        return observation


def _case(
    original_text: str = "A",
    normalized_text: str = "A",
    *,
    index: str = "1",
    category: str = "Date",
) -> PolyNormCase:
    return PolyNormCase(
        polynorm_locale="en-US",
        index=index,
        category=category,
        original_text=original_text,
        normalized_text=normalized_text,
    )


def _observation(
    phonemes: str,
    tokens: tuple[int, ...],
    *,
    warnings: tuple[str, ...] = (),
    segments: int = 1,
) -> PhonemeObservation:
    return PhonemeObservation(
        phonemes=phonemes,
        tokens=tokens,
        segment_count=segments,
        warnings=warnings,
    )


def _environment() -> dict[str, str]:
    return {"config_hash": "cfg"}


def test_evaluate_case_reports_raw_and_token_exactness() -> None:
    case = _case()
    harness = _FakeHarness({"A": _observation("abc", (1, 2))})

    result = evaluate_case(case, harness, pipeline="plain")

    assert result.raw_phoneme_exact is True
    assert result.semantic_phoneme_exact is True
    assert result.token_exact is True
    assert result.failed is False


def test_evaluate_case_distinguishes_presentation_only_phoneme_mismatch() -> None:
    case = _case(original_text="orig", normalized_text="norm")
    harness = _FakeHarness(
        {
            "orig": _observation("a-b", (1, 2)),
            "norm": _observation("ab", (1, 2)),
        }
    )

    result = evaluate_case(case, harness, pipeline="plain")

    assert result.raw_phoneme_exact is False
    assert result.semantic_phoneme_exact is True
    assert result.failed is False
    assert semantic_phoneme_key("a-b") == semantic_phoneme_key("ab")


def test_evaluate_case_reports_semantic_mismatch() -> None:
    case = _case(original_text="orig", normalized_text="norm")
    harness = _FakeHarness(
        {
            "orig": _observation("abc", (1, 2)),
            "norm": _observation("xyz", (3, 4)),
        }
    )

    result = evaluate_case(case, harness, pipeline="plain")

    assert result.semantic_phoneme_exact is False
    assert result.token_exact is False
    assert result.failed is True


def test_evaluate_case_reports_token_only_mismatch() -> None:
    case = _case(original_text="orig", normalized_text="norm")
    harness = _FakeHarness(
        {
            "orig": _observation("abc", (1, 2)),
            "norm": _observation("abc", (1, 3)),
        }
    )

    result = evaluate_case(case, harness, pipeline="plain")

    assert result.semantic_phoneme_exact is True
    assert result.token_exact is False
    assert result.failed is True


def test_evaluate_case_captures_exceptions_as_pipeline_failures() -> None:
    case = _case(original_text="orig", normalized_text="norm")
    harness = _FakeHarness({"orig": RuntimeError("boom"), "norm": _observation("abc", (1,))})

    result = evaluate_case(case, harness, pipeline="plain")

    assert result.original_error == "RuntimeError: boom"
    assert result.failed is True
    assert result.likely_owner == "pykokoro_pipeline"


def test_evaluate_case_aggregates_warnings() -> None:
    case = _case(original_text="orig", normalized_text="norm")
    harness = _FakeHarness(
        {
            "orig": _observation("abc", (1,), warnings=("one",)),
            "norm": _observation("abc", (1,), warnings=("two",)),
        }
    )

    result = evaluate_case(case, harness, pipeline="plain")

    assert result.warnings == ("one", "two")
    assert result.warning_count == 2


def test_aggregate_results_tracks_kind_quarantine_and_owner_counts(tmp_path: Path) -> None:
    ok_case = _case(index="1")
    mismatch_case = _case(
        original_text="SOURCE_TOKEN_ALPHA",
        normalized_text="TARGET_TOKEN_BETA",
        index="2",
    )
    quarantined_case = _case(
        original_text="QUARANTINE_SOURCE_GAMMA",
        normalized_text="QUARANTINE_TARGET_DELTA",
        index="3",
    )
    harness = _FakeHarness(
        {
            "A": _observation("abc", (1, 2)),
            "SOURCE_TOKEN_ALPHA": _observation("abc", (1,)),
            "TARGET_TOKEN_BETA": _observation("xyz", (2,)),
            "QUARANTINE_SOURCE_GAMMA": _observation("qqq", (9,)),
            "QUARANTINE_TARGET_DELTA": _observation("zzz", (8,)),
        }
    )
    ok_result = evaluate_case(ok_case, harness, pipeline="plain")
    mismatch_result = evaluate_case(
        mismatch_case,
        harness,
        pipeline="plain",
        direct_kokorog2p=lambda text: _observation("abc", (1,)),
    )
    quarantined_result = evaluate_case(
        quarantined_case,
        harness,
        pipeline="plain",
        quarantine={"en-US:3": "known upstream mismatch"},
    )

    summary = aggregate_results(
        [ok_result, mismatch_result, quarantined_result],
        environment=_environment(),
        baseline_failure_ids=["plain:en-US:old"],
    )
    write_failure_reports(tmp_path, [ok_result, mismatch_result, quarantined_result])
    dumped = json.dumps(summary, ensure_ascii=False)

    assert summary["counts"]["total_rows"] == 3
    assert summary["counts"]["evaluated_rows"] == 2
    assert summary["counts"]["quarantined_rows"] == 1
    assert summary["per_kind"]["identity"]["cases"] == 1
    assert summary["per_kind"]["transformation"]["cases"] == 1
    assert summary["likely_owner"]["pykokoro_pipeline"] == 1
    assert "known upstream mismatch" not in dumped
    assert mismatch_case.original_text not in dumped
    assert mismatch_case.normalized_text not in dumped


def test_evaluate_case_uses_direct_diagnostics_for_likely_owner() -> None:
    case = _case(original_text="orig", normalized_text="norm")
    harness = _FakeHarness(
        {
            "orig": _observation("abc", (1,)),
            "norm": _observation("xyz", (2,)),
        }
    )

    pipeline_owner = evaluate_case(
        case,
        harness,
        pipeline="plain",
        direct_kokorog2p=lambda text: _observation("same", (1,)),
    )
    upstream_owner = evaluate_case(
        case,
        harness,
        pipeline="plain",
        direct_kokorog2p=lambda text: _observation(text, (1 if text == "orig" else 2,)),
    )

    assert pipeline_owner.likely_owner == "pykokoro_pipeline"
    assert upstream_owner.likely_owner == "kokorog2p_or_spokenform"


def test_compare_summaries_reports_failure_and_metric_deltas() -> None:
    before = {
        "schema_version": 1,
        "dataset": {"commit": "abc"},
        "config_hash": "cfg",
        "metrics": {"raw_phoneme_exact": 1, "semantic_phoneme_exact": 1, "token_exact": 1, "token_error_rate": 0.0},
        "failure_ids": ["plain:en-US:1"],
        "per_locale": {"en-US": {"semantic_phoneme_exact_rate": 1.0}},
        "per_category": {"Date": {"semantic_phoneme_exact_rate": 1.0}},
        "likely_owner": {"pykokoro_pipeline": 1},
    }
    after = {
        "schema_version": 1,
        "dataset": {"commit": "abc"},
        "config_hash": "cfg",
        "metrics": {"raw_phoneme_exact": 2, "semantic_phoneme_exact": 3, "token_exact": 2, "token_error_rate": 0.25},
        "failure_ids": ["plain:en-US:2"],
        "per_locale": {"en-US": {"semantic_phoneme_exact_rate": 0.5}},
        "per_category": {"Date": {"semantic_phoneme_exact_rate": 0.5}},
        "likely_owner": {"kokorog2p_or_spokenform": 1},
    }

    comparison = compare_summaries(before, after)

    assert comparison["new_failures"] == ["plain:en-US:2"]
    assert comparison["resolved_failures"] == ["plain:en-US:1"]
    assert comparison["delta_metrics"]["semantic_phoneme_exact"] == 2.0
