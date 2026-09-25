"""Shared timestamp-window helpers for short-sentence phrase cutters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pykokoro.constants import SAMPLE_RATE


@dataclass(frozen=True)
class BoundaryWindows:
    """Timestamp-derived legal windows for phrase extraction boundaries."""

    target_start: int
    target_end: int
    left_window: tuple[int, int] | None
    right_window: tuple[int, int] | None
    has_left_context: bool
    has_right_context: bool


def boundary_windows_from_metadata(
    audio_length: int,
    metadata: dict[str, object],
) -> BoundaryWindows | None:
    """Build legal cut windows from target and neighboring token timestamps."""
    target_start = _sample_index(metadata.get("target_start_ts"), audio_length)
    target_end = _sample_index(metadata.get("target_end_ts"), audio_length)
    if target_start is None or target_end is None:
        record_cut_failure(metadata, "missing-target-timestamps", "target-boundary")
        return None
    if target_end <= target_start:
        record_cut_failure(metadata, "invalid-target-range", "target-boundary")
        return None

    has_left_context = bool(metadata.get("has_left_context", True))
    has_right_context = bool(metadata.get("has_right_context", True))
    left_window = None
    right_window = None

    if has_left_context:
        previous_end = _sample_index(metadata.get("previous_token_end_ts"), audio_length)
        if previous_end is None:
            record_cut_failure(metadata, "left-context-timestamp-missing", "cut-window")
            return None
        if previous_end > target_start:
            record_cut_failure(metadata, "left-search-invalid", "cut-window")
            return None
        left_window = (previous_end, target_start)

    if has_right_context:
        next_start = _sample_index(metadata.get("next_token_start_ts"), audio_length)
        if next_start is None:
            record_cut_failure(metadata, "right-context-timestamp-missing", "cut-window")
            return None
        if next_start < target_end:
            record_cut_failure(metadata, "right-search-invalid", "cut-window")
            return None
        right_window = (target_end, next_start)

    return BoundaryWindows(
        target_start=target_start,
        target_end=target_end,
        left_window=left_window,
        right_window=right_window,
        has_left_context=has_left_context,
        has_right_context=has_right_context,
    )


def record_cut_failure(metadata: dict[str, object], reason: str, stage: str = "cut-window") -> None:
    """Record the first cut failure and its processing stage."""
    metadata.setdefault("cut_failure_reason", reason)
    metadata.setdefault("failure_stage", stage)


def _sample_index(value: object, audio_length: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    sample = round(numeric * SAMPLE_RATE)
    tolerance = 2
    if sample < -tolerance or sample > audio_length + tolerance:
        return None
    return min(audio_length, max(0, sample))
