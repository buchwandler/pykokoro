"""SSMD emphasis capability policy and metadata resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .constants import EMPHASIS_GAIN_DB
from .exceptions import CapabilityError
from .types import PhonemeSegment, Trace

if TYPE_CHECKING:
    from .pipeline_config import PipelineConfig

EmphasisMode = Literal["plain", "approximate", "warn", "error"]


@dataclass(frozen=True)
class EmphasisDecision:
    """The resolved action for one SSMD emphasis value."""

    level: str | None
    volume: str | None = None
    warning_code: str | None = None
    reject: bool = False


def _format_gain_db(value: float) -> str:
    """Format a gain in the compact form accepted by AudioSig."""

    if value > 0:
        return f"+{value:g}dB"
    return f"{value:g}dB"


def resolve_emphasis(
    level: object,
    mode: EmphasisMode,
    *,
    gain_scale: float = 1.0,
) -> EmphasisDecision:
    """Resolve an SSMD emphasis value without touching audio or metadata."""

    if level is None or level == "" or level == "none":
        return EmphasisDecision(level="none")

    normalized = str(level)
    if mode == "plain":
        return EmphasisDecision(level=normalized)
    if mode == "approximate":
        base_gain = EMPHASIS_GAIN_DB.get(normalized)
        return EmphasisDecision(
            level=normalized,
            volume=None if base_gain is None else _format_gain_db(base_gain * gain_scale),
        )
    if mode == "warn":
        return EmphasisDecision(level=normalized, warning_code="ssmd.emphasis_unsupported")
    if mode == "error":
        return EmphasisDecision(level=normalized, reject=True)
    raise ValueError(f"Unsupported emphasis mode: {mode!r}")


def apply_emphasis_policy(
    phoneme_segments: list[PhonemeSegment],
    cfg: PipelineConfig,
    trace: Trace,
) -> None:
    """Apply emphasis policy before audio generation.

    Approximation metadata is copied to every phoneme batch, while warnings and
    errors are evaluated once per logical source segment.
    """

    seen_segment_ids: set[str] = set()
    for segment in phoneme_segments:
        metadata = segment.ssmd_metadata or {}
        emphasis = metadata.get("emphasis")
        decision = resolve_emphasis(
            emphasis,
            cfg.ssmd.emphasis_mode,
            gain_scale=cfg.ssmd.emphasis_gain_scale,
        )
        if decision.level == "none":
            continue

        if segment.segment_id not in seen_segment_ids:
            seen_segment_ids.add(segment.segment_id)
            if decision.reject:
                raise CapabilityError(
                    f"SSMD emphasis level {decision.level!r} is unsupported by "
                    "Kokoro with emphasis_mode='error'"
                )
            if decision.warning_code is not None:
                trace.warnings.append(f"{decision.warning_code}: using unmodified speech")

        if decision.volume is not None:
            metadata.setdefault("prosody_volume", decision.volume)
