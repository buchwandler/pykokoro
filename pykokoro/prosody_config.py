"""Configuration for AudioSig-backed speech prosody processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProsodyMethod = Literal[
    "phase_vocoder",
    "wsola",
    "esola",
    "td_psola",
    "psola",
]
AudioSigProsodyMethod = Literal["phase_vocoder", "wsola", "esola", "td_psola"]


@dataclass(frozen=True)
class ProsodyConfig:
    """Configuration for post-synthesis SSMD prosody processing.

    ``psola`` is accepted as a user-facing alias for AudioSig's ``td_psola``.
    """

    method: ProsodyMethod = "wsola"
    fallback_methods: tuple[ProsodyMethod, ...] = (
        "wsola",
        "phase_vocoder",
    )
    strict: bool = False
    clip: bool = False

    # AudioSig compositor/resampler parameters.
    n_fft: int = 2048
    hop_length: int | None = None
    filter_width: int = 32
    rolloff: float = 0.945
    boundary_blend_ms: float = 5.0

    def __post_init__(self) -> None:
        allowed = {
            "phase_vocoder",
            "wsola",
            "esola",
            "td_psola",
            "psola",
        }
        if self.method not in allowed:
            raise ValueError(f"unsupported prosody method: {self.method!r}")
        if any(method not in allowed for method in self.fallback_methods):
            raise ValueError("fallback_methods contains an unsupported method")
        if self.n_fft < 2:
            raise ValueError("n_fft must be at least 2")
        if self.hop_length is not None:
            if self.hop_length <= 0:
                raise ValueError("hop_length must be positive")
            if self.hop_length > self.n_fft:
                raise ValueError("hop_length must not exceed n_fft")
        if self.filter_width <= 0:
            raise ValueError("filter_width must be positive")
        if not 0.0 < self.rolloff <= 1.0:
            raise ValueError("rolloff must be in the interval (0, 1]")
        if self.boundary_blend_ms < 0.0:
            raise ValueError("boundary_blend_ms must be non-negative")


def canonical_prosody_method(method: ProsodyMethod) -> AudioSigProsodyMethod:
    """Return the AudioSig backend name for a configured method."""

    return "td_psola" if method == "psola" else method
