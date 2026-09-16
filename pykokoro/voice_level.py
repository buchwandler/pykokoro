"""Typed loading and resolution of offline voice loudness calibration."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from audiosig import apply_gain_db

from .loudness_config import LoudnessConfig

if TYPE_CHECKING:
    from .types import Trace
_CALIBRATION_METHODS = {"bs1770"}


@dataclass(frozen=True, slots=True)
class VoiceCalibrationKey:
    """Identity used to prevent calibration reuse across model variants."""

    model_source: str
    model_id: str
    quality: str
    voice: str

    def __str__(self) -> str:
        return f"{self.model_source}:{self.model_id}:{self.quality}:{self.voice}"

    @classmethod
    def parse(cls, value: str) -> VoiceCalibrationKey:
        parts = value.split(":", 3)
        if len(parts) != 4 or not all(parts):
            raise ValueError(f"invalid voice calibration key: {value!r}")
        return cls(*parts)


@dataclass(frozen=True, slots=True)
class VoiceLevelCalibration:
    """One reviewed static gain for one model and voice identity."""

    gain_db: float
    measured_lufs: float | None = None
    reference_lufs: float | None = None
    mad_lu: float | None = None
    samples: int = 0
    method: str = "bs1770"
    corpus_version: str | None = None


@dataclass(frozen=True, slots=True)
class VoiceCalibrationCatalog:
    """Validated calibration records and their measurement provenance."""

    schema: int
    method: str
    corpus: str
    reference_lufs: float
    generated_with: Mapping[str, Any]
    voices: Mapping[VoiceCalibrationKey, VoiceLevelCalibration]


_DEFAULT_CALIBRATION_PATH = Path(__file__).with_name("data") / "voice_level_calibration.json"


def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate calibration key: {key!r}")
        result[key] = value
    return result


def _record(value: object, key: VoiceCalibrationKey, corpus: str) -> VoiceLevelCalibration:
    if not isinstance(value, dict):
        raise ValueError(f"calibration record for {key} must be an object")
    allowed = {
        "gain_db",
        "measured_lufs",
        "reference_lufs",
        "mad_lu",
        "samples",
        "method",
        "corpus_version",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"calibration record for {key} has unknown fields: {sorted(unknown)}")
    if "gain_db" not in value or "samples" not in value:
        raise ValueError(f"calibration record for {key} requires gain_db and samples")
    samples = value["samples"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise ValueError(f"calibration record for {key} has invalid samples")
    method = value.get("method", "bs1770")
    if method not in _CALIBRATION_METHODS:
        raise ValueError(f"calibration record for {key} has invalid method: {method!r}")
    corpus_version = value.get("corpus_version", corpus)
    if not isinstance(corpus_version, str) or not corpus_version:
        raise ValueError(f"calibration record for {key} has invalid corpus_version")
    return VoiceLevelCalibration(
        gain_db=_finite(value["gain_db"], f"calibration record for {key}.gain_db"),
        measured_lufs=(
            None
            if value.get("measured_lufs") is None
            else _finite(value["measured_lufs"], f"{key}.measured_lufs")
        ),
        reference_lufs=(
            None
            if value.get("reference_lufs") is None
            else _finite(value["reference_lufs"], f"{key}.reference_lufs")
        ),
        mad_lu=None if value.get("mad_lu") is None else _finite(value["mad_lu"], f"{key}.mad_lu"),
        samples=samples,
        method=method,
        corpus_version=corpus_version,
    )


def load_voice_calibrations(path: str | Path | None = None) -> VoiceCalibrationCatalog:
    """Load and validate packaged or explicitly supplied calibration data."""

    calibration_path = Path(path) if path is not None else _DEFAULT_CALIBRATION_PATH
    with calibration_path.open(encoding="utf-8") as stream:
        data = json.load(stream, object_pairs_hook=_object_pairs)
    if not isinstance(data, dict):
        raise ValueError("voice calibration data must be an object")
    required = {"schema", "method", "corpus", "reference_lufs", "generated_with", "voices"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"voice calibration data is missing fields: {sorted(missing)}")
    unknown = set(data) - required
    if unknown:
        raise ValueError(f"voice calibration data has unknown fields: {sorted(unknown)}")
    if data["schema"] != 1:
        raise ValueError(f"unsupported voice calibration schema: {data['schema']!r}")
    if data["method"] not in _CALIBRATION_METHODS:
        raise ValueError(f"unsupported voice calibration method: {data['method']!r}")
    corpus = data["corpus"]
    if not isinstance(corpus, str) or not corpus:
        raise ValueError("voice calibration corpus must be a non-empty string")
    generated_with = data["generated_with"]
    if not isinstance(generated_with, dict):
        raise ValueError("generated_with must be an object")
    voices_data = data["voices"]
    if not isinstance(voices_data, dict):
        raise ValueError("voices must be an object")
    records: dict[VoiceCalibrationKey, VoiceLevelCalibration] = {}
    for raw_key, value in voices_data.items():
        if not isinstance(raw_key, str):
            raise ValueError("voice calibration keys must be strings")
        key = VoiceCalibrationKey.parse(raw_key)
        if key in records:
            raise ValueError(f"duplicate calibration key: {raw_key!r}")
        records[key] = _record(value, key, corpus)
    return VoiceCalibrationCatalog(
        schema=1,
        method=data["method"],
        corpus=corpus,
        reference_lufs=_finite(data["reference_lufs"], "reference_lufs"),
        generated_with=generated_with,
        voices=records,
    )


def resolve_voice_calibration(
    catalog: VoiceCalibrationCatalog,
    key: VoiceCalibrationKey | None,
) -> VoiceLevelCalibration | None:
    """Return a calibration record, or ``None`` for custom and unmeasured voices."""

    return None if key is None else catalog.voices.get(key)


@lru_cache(maxsize=1)
def default_voice_calibration() -> VoiceCalibrationCatalog:
    """Return the validated packaged calibration catalog."""
    return load_voice_calibrations()


def apply_voice_level_calibration(
    audio: np.ndarray,
    config: LoudnessConfig,
    key: VoiceCalibrationKey | None,
    *,
    external_audio: bool = False,
    trace: Trace | None = None,
    catalog: VoiceCalibrationCatalog | None = None,
    segment_id: str | None = None,
) -> np.ndarray:
    """Apply one static voice gain without measuring the runtime waveform."""
    mode = config.voice_leveling
    source = "none"
    calibration = None
    if not external_audio:
        if config.voice_gain_db is not None:
            gain_db = config.voice_gain_db
            source = "override"
        elif mode == "calibrated" and key is not None:
            calibration = resolve_voice_calibration(catalog or default_voice_calibration(), key)
            gain_db = calibration.gain_db if calibration is not None else 0.0
            source = "registry" if calibration is not None else "none"
        else:
            gain_db = 0.0
    else:
        gain_db = 0.0
        source = "external_audio"
    if trace is not None:
        trace.model.setdefault("voice_leveling", []).append(
            {
                "segment_id": segment_id,
                "mode": mode,
                "voice_key": None if key is None else str(key),
                "gain_db": gain_db,
                "source": source,
                "method": None if calibration is None else calibration.method,
                "corpus": None if calibration is None else calibration.corpus_version,
            }
        )
    return audio if gain_db == 0.0 else apply_gain_db(audio, gain_db)
