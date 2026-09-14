"""Timestamp-guided short-sentence phrase cutter with smooth boundaries."""

from __future__ import annotations

import numpy as np
from audiosig import find_smooth_cut_point

from pykokoro.constants import SAMPLE_RATE

from .energy_valley import find_energy_valley_cut_bounds
from .shared import BoundaryWindows, boundary_windows_from_metadata, record_cut_failure


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

    search_radius_samples = _metadata_samples(metadata, "search_radius_ms", 35.0)
    context_guard_samples = _metadata_samples(metadata, "context_guard_ms", 8.0, minimum=0)
    analysis_window_samples = _metadata_samples(metadata, "analysis_window_ms", 5.0)
    left_result = _find_left_cut(
        audio,
        windows,
        metadata,
        search_radius_samples=search_radius_samples,
        context_guard_samples=context_guard_samples,
        analysis_window_samples=analysis_window_samples,
    )
    right_result = _find_right_cut(
        audio,
        windows,
        metadata,
        search_radius_samples=search_radius_samples,
        context_guard_samples=context_guard_samples,
        analysis_window_samples=analysis_window_samples,
    )
    if left_result is None or right_result is None:
        return None
    left_cut, left_strategy = left_result
    right_cut, right_strategy = right_result

    if not 0 <= left_cut <= right_cut <= len(audio):
        record_cut_failure(metadata, "cut-order-invalid", "cut-validation")
        return None
    if windows.has_left_context:
        if windows.left_window is None:
            record_cut_failure(metadata, "left-search-invalid", "cut-search")
            return None
        previous_end, target_start = windows.left_window
        if not previous_end <= left_cut <= target_start:
            record_cut_failure(metadata, "left-cut-outside-safe-range", "cut-validation")
            return None
    elif left_cut != 0:
        record_cut_failure(metadata, "left-cut-outside-safe-range", "cut-validation")
        return None
    if windows.has_right_context:
        if windows.right_window is None:
            record_cut_failure(metadata, "right-search-invalid", "cut-search")
            return None
        target_end, next_start = windows.right_window
        if not target_end <= right_cut <= next_start:
            record_cut_failure(metadata, "right-cut-outside-safe-range", "cut-validation")
            return None
    elif right_cut != len(audio):
        record_cut_failure(metadata, "right-cut-outside-safe-range", "cut-validation")
        return None
    if right_cut <= left_cut:
        record_cut_failure(metadata, "empty-target-audio", "cut-validation")
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
    *,
    search_radius_samples: int,
    context_guard_samples: int,
    analysis_window_samples: int,
) -> tuple[int, str] | None:
    if not windows.has_left_context:
        return 0, "no-context"
    if windows.left_window is None:
        record_cut_failure(metadata, "left-search-invalid", "cut-search")
        return None
    interval = _search_interval(
        windows.left_window,
        anchor=windows.target_start,
        side="left",
        audio_length=len(audio),
        search_radius_samples=search_radius_samples,
        context_guard_samples=context_guard_samples,
    )
    if interval is None:
        record_cut_failure(metadata, "left-search-invalid", "cut-search")
        return None
    start, end, strategy = interval
    cut = find_smooth_cut_point(
        _as_float_audio(audio),
        start=start,
        end=end,
        anchor=windows.target_start,
        window_length=analysis_window_samples,
    )
    if cut is None:
        record_cut_failure(metadata, "left-search-invalid", "cut-search")
        return None
    return cut, strategy


def _find_right_cut(
    audio: np.ndarray,
    windows: BoundaryWindows,
    metadata: dict[str, object],
    *,
    search_radius_samples: int,
    context_guard_samples: int,
    analysis_window_samples: int,
) -> tuple[int, str] | None:
    if not windows.has_right_context:
        return len(audio), "no-context"
    if windows.right_window is None:
        record_cut_failure(metadata, "right-search-invalid", "cut-search")
        return None
    interval = _search_interval(
        windows.right_window,
        anchor=windows.target_end,
        side="right",
        audio_length=len(audio),
        search_radius_samples=search_radius_samples,
        context_guard_samples=context_guard_samples,
    )
    if interval is None:
        record_cut_failure(metadata, "right-search-invalid", "cut-search")
        return None
    start, end, strategy = interval
    cut = find_smooth_cut_point(
        _as_float_audio(audio),
        start=start,
        end=end,
        anchor=windows.target_end,
        window_length=analysis_window_samples,
    )
    if cut is None:
        record_cut_failure(metadata, "right-search-invalid", "cut-search")
        return None
    return cut, strategy


def _search_interval(
    window: tuple[int, int],
    *,
    anchor: int,
    side: str,
    audio_length: int,
    search_radius_samples: int,
    context_guard_samples: int,
) -> tuple[int, int, str] | None:
    safe_low, safe_high = window
    guard = min(context_guard_samples, (safe_high - safe_low) // 3)
    if side == "left":
        start = max(safe_low, anchor - search_radius_samples)
        end = min(safe_high, anchor - guard)
    else:
        start = max(safe_low, anchor + guard)
        end = min(safe_high, anchor + search_radius_samples)
    if end > start:
        return start, end, "timestamp-smooth"

    relaxed_start = max(safe_low, anchor - search_radius_samples)
    relaxed_end = min(audio_length, safe_high + 1, anchor + search_radius_samples + 1)
    if relaxed_end <= relaxed_start:
        return None
    strategy = "timestamp-anchor" if safe_low == safe_high == anchor else "timestamp-smooth-relaxed"
    return relaxed_start, relaxed_end, strategy


def _metadata_samples(
    metadata: dict[str, object],
    key: str,
    default_ms: float,
    *,
    minimum: int = 1,
) -> int:
    value = metadata.get(key, default_ms)
    return max(minimum, round(SAMPLE_RATE * float(value) / 1000.0))


def _as_float_audio(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32)
