"""Timestamp-guided short-sentence phrase cutter with smooth boundaries."""

from __future__ import annotations

from typing import Final

import numpy as np
from audiosig import find_smooth_cut_point

from pykokoro.constants import SAMPLE_RATE

from .energy_valley import find_energy_valley_cut_bounds
from .shared import BoundaryWindows, boundary_windows_from_metadata

_SEARCH_RADIUS_SAMPLES: Final = round(SAMPLE_RATE * 0.035)
_CONTEXT_GUARD_SAMPLES: Final = round(SAMPLE_RATE * 0.008)
_ANALYSIS_WINDOW_SAMPLES: Final = round(SAMPLE_RATE * 0.005)


def cut_with_timestamp_adaptive(
    audio: np.ndarray,
    metadata: dict[str, object],
) -> np.ndarray | None:
    """Cut phrase audio using strict valleys, then legal smooth boundaries."""
    cut_bounds = find_timestamp_adaptive_cut_bounds(audio, metadata)
    if cut_bounds is None:
        return None
    left_cut, right_cut = cut_bounds
    metadata["cut_left"] = left_cut
    metadata["cut_right"] = right_cut
    return audio[left_cut:right_cut]


def find_timestamp_adaptive_cut_bounds(
    audio: np.ndarray,
    metadata: dict[str, object],
) -> tuple[int, int] | None:
    """Return legal phrase boundaries, preferring the existing strict cutter."""
    windows = boundary_windows_from_metadata(len(audio), metadata)
    if windows is None:
        return None

    strict_bounds = find_energy_valley_cut_bounds(audio, metadata)
    if strict_bounds is not None:
        metadata["cut_strategy"] = "energy-valley"
        return strict_bounds

    left_result = _find_left_cut(audio, windows, metadata)
    right_result = _find_right_cut(audio, windows, metadata)
    if left_result is None or right_result is None:
        return None
    left_cut, left_strategy = left_result
    right_cut, right_strategy = right_result

    if not 0 <= left_cut <= right_cut <= len(audio):
        metadata["cut_failure_reason"] = "cut-order-invalid"
        return None
    if windows.has_left_context:
        if windows.left_window is None:
            metadata["cut_failure_reason"] = "left-search-invalid"
            return None
        previous_end, target_start = windows.left_window
        if not previous_end <= left_cut <= target_start:
            metadata["cut_failure_reason"] = "left-cut-outside-safe-range"
            return None
    elif left_cut != 0:
        metadata["cut_failure_reason"] = "left-cut-outside-safe-range"
        return None
    if windows.has_right_context:
        if windows.right_window is None:
            metadata["cut_failure_reason"] = "right-search-invalid"
            return None
        target_end, next_start = windows.right_window
        if not target_end <= right_cut <= next_start:
            metadata["cut_failure_reason"] = "right-cut-outside-safe-range"
            return None
    elif right_cut != len(audio):
        metadata["cut_failure_reason"] = "right-cut-outside-safe-range"
        return None
    if right_cut <= left_cut:
        metadata["cut_failure_reason"] = "empty-target-audio"
        return None

    strategies = {left_strategy, right_strategy}
    if "timestamp-anchor" in strategies:
        metadata["cut_strategy"] = "timestamp-anchor"
    elif "timestamp-smooth-relaxed" in strategies:
        metadata["cut_strategy"] = "timestamp-smooth-relaxed"
    else:
        metadata["cut_strategy"] = "timestamp-smooth"
    return left_cut, right_cut


def _find_left_cut(
    audio: np.ndarray,
    windows: BoundaryWindows,
    metadata: dict[str, object],
) -> tuple[int, str] | None:
    if not windows.has_left_context:
        return 0, "no-context"
    if windows.left_window is None:
        metadata["cut_failure_reason"] = "left-search-invalid"
        return None
    interval = _search_interval(
        windows.left_window,
        anchor=windows.target_start,
        side="left",
        audio_length=len(audio),
    )
    if interval is None:
        metadata["cut_failure_reason"] = "left-search-invalid"
        return None
    start, end, strategy = interval
    cut = find_smooth_cut_point(
        _as_float_audio(audio),
        start=start,
        end=end,
        anchor=windows.target_start,
        window_length=_ANALYSIS_WINDOW_SAMPLES,
    )
    if cut is None:
        metadata["cut_failure_reason"] = "left-search-invalid"
        return None
    return cut, strategy


def _find_right_cut(
    audio: np.ndarray,
    windows: BoundaryWindows,
    metadata: dict[str, object],
) -> tuple[int, str] | None:
    if not windows.has_right_context:
        return len(audio), "no-context"
    if windows.right_window is None:
        metadata["cut_failure_reason"] = "right-search-invalid"
        return None
    interval = _search_interval(
        windows.right_window,
        anchor=windows.target_end,
        side="right",
        audio_length=len(audio),
    )
    if interval is None:
        metadata["cut_failure_reason"] = "right-search-invalid"
        return None
    start, end, strategy = interval
    cut = find_smooth_cut_point(
        _as_float_audio(audio),
        start=start,
        end=end,
        anchor=windows.target_end,
        window_length=_ANALYSIS_WINDOW_SAMPLES,
    )
    if cut is None:
        metadata["cut_failure_reason"] = "right-search-invalid"
        return None
    return cut, strategy


def _search_interval(
    window: tuple[int, int],
    *,
    anchor: int,
    side: str,
    audio_length: int,
) -> tuple[int, int, str] | None:
    safe_low, safe_high = window
    guard = min(_CONTEXT_GUARD_SAMPLES, (safe_high - safe_low) // 3)
    if side == "left":
        start = max(safe_low, anchor - _SEARCH_RADIUS_SAMPLES)
        end = min(safe_high, anchor - guard)
    else:
        start = max(safe_low, anchor + guard)
        end = min(safe_high, anchor + _SEARCH_RADIUS_SAMPLES)
    if end > start:
        return start, end, "timestamp-smooth"

    relaxed_start = max(safe_low, anchor - _SEARCH_RADIUS_SAMPLES)
    relaxed_end = min(audio_length, safe_high + 1, anchor + _SEARCH_RADIUS_SAMPLES + 1)
    if relaxed_end <= relaxed_start:
        return None
    strategy = "timestamp-anchor" if safe_low == safe_high == anchor else "timestamp-smooth-relaxed"
    return relaxed_start, relaxed_end, strategy


def _as_float_audio(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32)
