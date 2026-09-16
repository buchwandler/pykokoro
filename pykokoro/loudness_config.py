"""Configuration for voice leveling and complete-output loudness."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

VoiceLevelingMode = Literal["off", "calibrated"]
PeakPolicy = Literal["reduce_gain", "error"]


@dataclass(frozen=True)
class LoudnessConfig:
    """Output-level policy, separate from expressive prosody controls."""

    voice_leveling: VoiceLevelingMode = "off"
    target_lufs: float | None = None
    true_peak_ceiling_dbtp: float | None = -1.0
    peak_policy: PeakPolicy = "reduce_gain"
    voice_gain_db: float | None = None

    def __post_init__(self) -> None:
        if self.voice_leveling not in {"off", "calibrated"}:
            raise ValueError(
                f"voice_leveling must be 'off' or 'calibrated', got {self.voice_leveling!r}"
            )
        if self.peak_policy not in {"reduce_gain", "error"}:
            raise ValueError(
                f"peak_policy must be 'reduce_gain' or 'error', got {self.peak_policy!r}"
            )
        for name in ("target_lufs", "true_peak_ceiling_dbtp", "voice_gain_db"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite real number or None, got {value!r}")
