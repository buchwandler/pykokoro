"""Kokoro inference controls for request-centric speech synthesis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from .exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Acoustic speed, default language, and short-sentence inference settings."""

    speed: float = 1.0
    lang: str | None = None
    random_seed: int | None = None
    enable_short_sentence: bool | None = None

    def __post_init__(self) -> None:
        if isinstance(self.speed, bool) or not isinstance(self.speed, Real):
            raise ConfigurationError("speed must be a real number greater than zero")
        if not math.isfinite(float(self.speed)) or self.speed <= 0:
            raise ConfigurationError("speed must be finite and greater than zero")
        if self.lang is not None and (not isinstance(self.lang, str) or not self.lang.strip()):
            raise ValueError("lang must be a non-empty string or None")
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int)
        ):
            raise ValueError("random_seed must be an integer or None")
        if self.enable_short_sentence is not None and not isinstance(
            self.enable_short_sentence, bool
        ):
            raise ValueError("enable_short_sentence must be a boolean or None")
