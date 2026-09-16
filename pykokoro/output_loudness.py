"""Optional one-pass loudness normalization for complete PyKokoro output."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from audiosig import apply_gain_db, measure_loudness

from .exceptions import ConfigurationError
from .loudness_config import LoudnessConfig

if TYPE_CHECKING:
    from .types import Trace


class LoudnessNormalizationError(ConfigurationError):
    """The requested complete-output target violates the peak policy."""


def apply_complete_output_loudness(
    audio: np.ndarray,
    sample_rate: int,
    config: LoudnessConfig,
    trace: Trace | None = None,
) -> np.ndarray:
    """Apply one static gain to a complete waveform when a target is configured."""

    if config.target_lufs is None or audio.size == 0:
        return audio

    metrics = measure_loudness(audio, sample_rate=sample_rate)
    if not math.isfinite(metrics.integrated_lufs):
        if trace is not None:
            trace.warnings.append(
                "output_loudness: digital silence has no integrated LUFS; unchanged"
            )
            trace.model.setdefault("output_loudness", {})
            trace.model["output_loudness"].update(
                {"target_lufs": config.target_lufs, "applied_gain_db": 0.0, "target_reached": False}
            )
        return audio

    requested_gain_db = config.target_lufs - metrics.integrated_lufs
    safe_gain_db = math.inf
    if config.true_peak_ceiling_dbtp is not None and math.isfinite(metrics.true_peak_dbtp):
        safe_gain_db = config.true_peak_ceiling_dbtp - metrics.true_peak_dbtp
    if config.peak_policy == "error" and requested_gain_db > safe_gain_db:
        raise LoudnessNormalizationError(
            "Complete-output loudness target exceeds true-peak ceiling: "
            f"measured LUFS={metrics.integrated_lufs:.3f}, target LUFS={config.target_lufs:.3f}, "
            f"requested gain={requested_gain_db:.3f} dB, measured true peak={metrics.true_peak_dbtp:.3f} dBTP, "
            f"ceiling={config.true_peak_ceiling_dbtp!r} dBTP, maximum safe gain={safe_gain_db:.3f} dB"
        )
    applied_gain_db = (
        requested_gain_db if requested_gain_db <= 0.0 else min(requested_gain_db, safe_gain_db)
    )
    normalized = apply_gain_db(audio, applied_gain_db)
    if trace is not None:
        trace.model["output_loudness"] = {
            "integrated_lufs_before": metrics.integrated_lufs,
            "target_lufs": config.target_lufs,
            "requested_gain_db": requested_gain_db,
            "applied_gain_db": applied_gain_db,
            "sample_peak_before_dbfs": metrics.sample_peak_dbfs,
            "true_peak_before_dbtp": metrics.true_peak_dbtp,
            "true_peak_ceiling_dbtp": config.true_peak_ceiling_dbtp,
            "target_reached": math.isclose(applied_gain_db, requested_gain_db, abs_tol=1e-9),
            "peak_policy": config.peak_policy,
        }
    return np.asarray(normalized).astype(audio.dtype, copy=False)
