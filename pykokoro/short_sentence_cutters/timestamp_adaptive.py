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
    """Cut phrase audio with strict energy selection followed by smooth boundaries."""
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

    left_cut = _find_left_cut(audio, windows)
    right_cut = _find_right_cut(audio, windows)
    if left_cut is None or right_cut is None:
        return None
    if not 0 <= left_cut < right_cut <= len(audio):
        return None
    if windows.left_window is not None and not (
        windows.left_window[0] <= left_cut < windows.left_window[1]
    ):
        return None
    if windows.right_window is not None and not (
        windows.right_window[0] <= right_cut <= windows.right_window[1]
    ):
        return None
    if left_cut >= windows.target_start or right_cut <= windows.target_end:
        return None

    metadata["cut_strategy"] = "timestamp-smooth"
    return left_cut, right_cut


def _find_left_cut(
    audio: np.ndarray,
    windows: BoundaryWindows,
) -> int | None:
    if not windows.has_left_context:
        return 0
    if windows.left_window is None:
        return None
    start, end = _search_interval(
        windows.left_window,
        anchor=windows.target_start,
        side="left",
    )
    if end <= start:
        return None
    return find_smooth_cut_point(
        _as_float_audio(audio),
        start=start,
        end=end,
        anchor=windows.target_start,
        window_length=_ANALYSIS_WINDOW_SAMPLES,
    )


def _find_right_cut(
    audio: np.ndarray,
    windows: BoundaryWindows,
) -> int | None:
    if not windows.has_right_context:
        return len(audio)
    if windows.right_window is None:
        return None
    start, end = _search_interval(
        windows.right_window,
        anchor=windows.target_end,
        side="right",
    )
    if end <= start:
        return None
    return find_smooth_cut_point(
        _as_float_audio(audio),
        start=start,
        end=end,
        anchor=windows.target_end,
        window_length=_ANALYSIS_WINDOW_SAMPLES,
    )


def _search_interval(
    window: tuple[int, int],
    *,
    anchor: int,
    side: str,
) -> tuple[int, int]:
    window_start, window_end = window
    legal_span = window_end - window_start
    if legal_span <= 0:
        return window_start, window_start
    guard = min(_CONTEXT_GUARD_SAMPLES, legal_span // 3)
    if side == "left":
        return (
            max(window_start, anchor - _SEARCH_RADIUS_SAMPLES),
            min(window_end, anchor - guard),
        )
    return (
        max(window_start, anchor + guard),
        min(window_end, anchor + _SEARCH_RADIUS_SAMPLES),
    )


def _as_float_audio(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32)
