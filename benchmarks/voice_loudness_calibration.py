"""Promote voice loudness benchmark reports into runtime calibration data."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pykokoro.voice_level import VoiceCalibrationKey

PRODUCTION_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1] / "pykokoro" / "data" / "voice_level_calibration.json"
)
REVIEWED_STATUSES = frozenset({"eligible", "extreme_gain"})
_RUNTIME_FIELDS = {
    "gain_db",
    "measured_lufs",
    "reference_lufs",
    "mad_lu",
    "samples",
    "method",
    "corpus_version",
}


def load_measurement_report(path: Path) -> dict[str, Any]:
    """Load a JSON measurement report and validate its top-level object type."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("measurement report must be an object")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _key(row: Mapping[str, Any]) -> VoiceCalibrationKey:
    fields = ("model_source", "model_id", "quality", "voice")
    values = [row.get(field) for field in fields]
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError("aggregate rows must have non-empty voice identity fields")
    return VoiceCalibrationKey(*values)


def _policy_limits(policy: Mapping[str, Any]) -> tuple[float, float]:
    if "max_boost_db" in policy and "max_attenuation_db" in policy:
        return _finite(policy["max_boost_db"], "policy.max_boost_db"), _finite(
            policy["max_attenuation_db"], "policy.max_attenuation_db"
        )
    if "max_abs_requested_gain_db" in policy:
        limit = _finite(policy["max_abs_requested_gain_db"], "policy.max_abs_requested_gain_db")
        return limit, limit
    raise ValueError("measurement policy requires split gain limits or max_abs_requested_gain_db")


def validate_measurement_report(report: Mapping[str, Any]) -> None:
    """Validate report provenance and the aggregate shape used for promotion."""
    if not isinstance(report, Mapping):
        raise ValueError("measurement report must be an object")
    if report.get("schema") != 2:
        raise ValueError("measurement report must use schema 2")
    if not isinstance(report.get("corpus"), str) or not report["corpus"]:
        raise ValueError("measurement report requires a non-empty corpus")
    generated_with = report.get("generated_with")
    if not isinstance(generated_with, Mapping):
        raise ValueError("measurement report requires generated_with provenance")
    if any(not isinstance(value, str) or not value for value in generated_with.values()):
        raise ValueError("generated_with provenance values must be non-empty strings")

    policy = report.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("measurement report requires a policy object")
    repeats = policy.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("policy.repeats must be a positive integer")
    _finite(policy.get("reference_lufs"), "policy.reference_lufs")
    _finite(policy.get("calibration_peak_ceiling_dbtp"), "policy.calibration_peak_ceiling_dbtp")
    _finite(policy.get("max_mad_lu"), "policy.max_mad_lu")
    _policy_limits(policy)

    aggregates = report.get("aggregates")
    if not isinstance(aggregates, list):
        raise ValueError("measurement report requires an aggregates list")
    seen: set[VoiceCalibrationKey] = set()
    for index, aggregate in enumerate(aggregates):
        if not isinstance(aggregate, Mapping):
            raise ValueError(f"aggregate {index} must be an object")
        key = _key(aggregate)
        if key in seen:
            raise ValueError(f"duplicate aggregate voice key: {key}")
        seen.add(key)
        status = aggregate.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"aggregate {key} requires a status")
        repeat_count = aggregate.get("repeat_count")
        if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 0:
            raise ValueError(f"aggregate {key} has an invalid repeat_count")
        for field in ("median_lufs", "mad_lu", "max_true_peak_dbtp"):
            value = aggregate.get(field)
            if value is None and status == "non_finite_loudness":
                continue
            _finite(value, f"aggregate {key}.{field}")


def _runtime_record(
    aggregate: Mapping[str, Any], report: Mapping[str, Any], gain_db: float
) -> dict[str, Any]:
    policy = report["policy"]
    return {
        "gain_db": gain_db,
        "measured_lufs": _finite(aggregate["median_lufs"], "median_lufs"),
        "reference_lufs": _finite(policy["reference_lufs"], "reference_lufs"),
        "mad_lu": _finite(aggregate["mad_lu"], "mad_lu"),
        "samples": aggregate["repeat_count"],
        "method": "bs1770",
        "corpus_version": report["corpus"],
    }


def build_runtime_calibration(
    report: Mapping[str, Any],
    *,
    reviewed_statuses: frozenset[str] = REVIEWED_STATUSES,
) -> dict[str, Any]:
    """Build a deterministic schema-1 runtime catalog from reviewed aggregates."""
    validate_measurement_report(report)
    policy = report["policy"]
    repeats = policy["repeats"]
    reference_lufs = _finite(policy["reference_lufs"], "policy.reference_lufs")
    peak_ceiling = _finite(
        policy["calibration_peak_ceiling_dbtp"], "policy.calibration_peak_ceiling_dbtp"
    )
    max_mad_lu = _finite(policy["max_mad_lu"], "policy.max_mad_lu")
    voices: dict[str, dict[str, Any]] = {}

    for aggregate in report["aggregates"]:
        if aggregate["status"] not in reviewed_statuses:
            continue
        if aggregate["repeat_count"] != repeats:
            continue
        mad_lu = _finite(aggregate["mad_lu"], "aggregate.mad_lu")
        if mad_lu > max_mad_lu:
            continue
        measured_lufs = _finite(aggregate["median_lufs"], "aggregate.median_lufs")
        max_true_peak = _finite(aggregate["max_true_peak_dbtp"], "aggregate.max_true_peak_dbtp")
        requested_gain = reference_lufs - measured_lufs
        gain_db = (
            min(requested_gain, peak_ceiling - max_true_peak)
            if requested_gain > 0
            else requested_gain
        )
        key = str(_key(aggregate))
        voices[key] = _runtime_record(aggregate, report, gain_db)

    return {
        "schema": 1,
        "method": "bs1770",
        "corpus": report["corpus"],
        "reference_lufs": reference_lufs,
        "generated_with": dict(report["generated_with"]),
        "voices": {key: voices[key] for key in sorted(voices)},
    }


def write_runtime_calibration(report_path: Path, output_path: Path) -> None:
    """Write a deterministic runtime catalog without replacing packaged production data."""
    if output_path.resolve() == PRODUCTION_CALIBRATION_PATH.resolve():
        raise ValueError("refusing to overwrite packaged production calibration data")
    report = load_measurement_report(report_path)
    catalog = build_runtime_calibration(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(catalog, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
