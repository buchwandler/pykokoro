"""Public options for voice enrollment engines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .exceptions import InvalidVoiceError

VoiceEnrollmentEngine = Literal["inno", "akinvox"]


@dataclass(frozen=True, slots=True)
class InnoEnrollmentOptions:
    """Tuning options supported by the Inno v0.2 enrollment engine."""

    fmax: float | None = None

    def __post_init__(self) -> None:
        if self.fmax is not None and (
            isinstance(self.fmax, bool)
            or not isinstance(self.fmax, (int, float))
            or not math.isfinite(self.fmax)
            or self.fmax <= 0
        ):
            raise InvalidVoiceError("Inno fmax must be a finite positive number")

    def to_runtime_options(self) -> dict[str, float]:
        if self.fmax is None:
            return {}
        return {"fmax": float(self.fmax)}
