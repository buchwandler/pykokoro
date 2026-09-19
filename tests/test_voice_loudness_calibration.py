from __future__ import annotations

import pytest

pytest.importorskip("pykokoro.stages.doc_parsers.plain")


import json
import math
from pathlib import Path


from benchmarks.voice_loudness_calibration import (
    PRODUCTION_CALIBRATION_PATH,
    build_runtime_calibration,
    validate_measurement_report,
    write_runtime_calibration,
)
from pykokoro.voice_level import load_voice_calibrations

CATALOG_SOURCE = Path("data/voice_level_calibration_from_measurements.json")


def _aggregate(
    voice: str = "voice",
    *,
    status: str = "eligible",
    median_lufs: float = -24.0,
    mad_lu: float = 0.0,
    repeat_count: int = 3,
    max_true_peak_dbtp: float = -2.0,
) -> dict[str, object]:
    return {
        "model_source": "github",
        "model_id": "v1.0",
        "quality": "fp32",
        "voice": voice,
        "locale": "en-US",
        "median_lufs": median_lufs,
        "mad_lu": mad_lu,
        "repeat_count": repeat_count,
        "max_true_peak_dbtp": max_true_peak_dbtp,
        "status": status,
    }


def _report(*aggregates: dict[str, object]) -> dict[str, object]:
    return {
        "schema": 2,
        "corpus": "test-corpus",
        "policy": {
            "repeats": 3,
            "reference_lufs": -24.0,
            "calibration_peak_ceiling_dbtp": -1.0,
            "max_boost_db": 8.0,
            "max_attenuation_db": 12.0,
            "max_mad_lu": 0.75,
        },
        "generated_with": {"pykokoro": "test", "audiosig": "test", "spokenform": "test"},
        "aggregates": list(aggregates),
    }


def test_complete_aggregate_is_converted_to_runtime_schema_one() -> None:
    result = build_runtime_calibration(_report(_aggregate(median_lufs=-22.0)))
    record = result["voices"]["github:v1.0:fp32:voice"]
    assert result["schema"] == 1
    assert record == {
        "gain_db": -2.0,
        "measured_lufs": -22.0,
        "reference_lufs": -24.0,
        "mad_lu": 0.0,
        "samples": 3,
        "method": "bs1770",
        "corpus_version": "test-corpus",
    }


def test_duplicate_voice_keys_are_rejected() -> None:
    aggregate = _aggregate()
    with pytest.raises(ValueError, match="duplicate"):
        validate_measurement_report(_report(aggregate, dict(aggregate)))


def test_unsupported_measurement_schema_is_rejected() -> None:
    report = _report(_aggregate())
    report["schema"] = 1
    with pytest.raises(ValueError, match="schema 2"):
        validate_measurement_report(report)


def test_non_finite_numeric_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_measurement_report(_report(_aggregate(median_lufs=math.nan)))


def test_insufficient_repeats_are_not_promoted() -> None:
    result = build_runtime_calibration(_report(_aggregate(repeat_count=2)))
    assert result["voices"] == {}


def test_high_mad_is_not_promoted() -> None:
    result = build_runtime_calibration(_report(_aggregate(mad_lu=0.76)))
    assert result["voices"] == {}


def test_reviewed_large_attenuation_is_promoted() -> None:
    result = build_runtime_calibration(
        _report(_aggregate(status="extreme_gain", median_lufs=-12.0))
    )
    assert result["voices"]["github:v1.0:fp32:voice"]["gain_db"] == -12.0


def test_missing_review_status_is_not_silently_promoted() -> None:
    aggregate = _aggregate()
    del aggregate["status"]
    with pytest.raises(ValueError, match="requires a status"):
        validate_measurement_report(_report(aggregate))


def test_output_ordering_is_deterministic() -> None:
    result = build_runtime_calibration(_report(_aggregate("z"), _aggregate("a")))
    assert list(result["voices"]) == ["github:v1.0:fp32:a", "github:v1.0:fp32:z"]


def test_output_can_be_loaded_by_runtime_loader(tmp_path: Path) -> None:
    report_path = tmp_path / "measurements.json"
    output_path = tmp_path / "calibration.json"
    report_path.write_text(json.dumps(_report(_aggregate())), encoding="utf-8")
    write_runtime_calibration(report_path, output_path)
    assert load_voice_calibrations(output_path).voices


def test_supplied_catalog_has_exactly_216_runtime_records() -> None:
    catalog = load_voice_calibrations(CATALOG_SOURCE)
    assert len(catalog.voices) == 216


def test_runtime_records_have_no_candidate_only_fields() -> None:
    result = build_runtime_calibration(_report(_aggregate()))
    assert set(result["voices"]["github:v1.0:fp32:voice"]) == {
        "gain_db",
        "measured_lufs",
        "reference_lufs",
        "mad_lu",
        "samples",
        "method",
        "corpus_version",
    }


def test_writer_protects_production_catalog(tmp_path: Path) -> None:
    report_path = tmp_path / "measurements.json"
    report_path.write_text(json.dumps(_report(_aggregate())), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing"):
        write_runtime_calibration(report_path, PRODUCTION_CALIBRATION_PATH)
