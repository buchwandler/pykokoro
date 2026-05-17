"""VAD-run phrase cutter for short-sentence extraction."""

from __future__ import annotations

import numpy as np

from pykokoro.constants import SAMPLE_RATE
from pykokoro.trim import energy_based_vad

from .shared import BoundaryWindows, boundary_windows_from_metadata


def cut_with_vad(audio: np.ndarray, metadata: dict[str, object]) -> np.ndarray | None:
    """Cut phrase audio at quiet runs that overlap legal timestamp windows."""
    windows = boundary_windows_from_metadata(len(audio), metadata)
    if windows is None:
        return None

    runs = _quiet_runs(
        audio,
        frame_duration_ms=int(metadata.get("frame_duration_ms", 5)),
        energy_threshold=float(metadata.get("energy_threshold", 0.05)),
        min_silence_seconds=float(metadata.get("min_silence_seconds", 0.02)),
    )
    left_cut = _left_cut(runs, windows) if windows.has_left_context else 0
    right_cut = _right_cut(runs, windows) if windows.has_right_context else len(audio)
    if left_cut is None or right_cut is None or right_cut <= left_cut:
        return None
    return audio[left_cut:right_cut]


def _quiet_runs(
    audio: np.ndarray,
    *,
    frame_duration_ms: int,
    energy_threshold: float,
    min_silence_seconds: float,
) -> list[tuple[int, int]]:
    speech_frames = energy_based_vad(
        audio,
        SAMPLE_RATE,
        frame_duration_ms=frame_duration_ms,
        energy_threshold=energy_threshold,
    )
    quiet_frames = ~speech_frames
    samples_per_frame = max(1, int(SAMPLE_RATE * frame_duration_ms / 1000))
    min_frames = max(1, int(min_silence_seconds * 1000 / frame_duration_ms))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_quiet in enumerate(quiet_frames):
        if is_quiet and start is None:
            start = index
        elif not is_quiet and start is not None:
            if index - start >= min_frames:
                runs.append((start * samples_per_frame, index * samples_per_frame))
            start = None
    if start is not None and len(quiet_frames) - start >= min_frames:
        runs.append((start * samples_per_frame, len(audio)))
    return runs


def _left_cut(runs: list[tuple[int, int]], windows: BoundaryWindows) -> int | None:
    assert windows.left_window is not None
    candidates = _overlapping_runs(runs, windows.left_window)
    if not candidates:
        return None
    run = max(candidates, key=lambda value: min(value[1], windows.left_window[1]))
    return min(run[1], windows.left_window[1])


def _right_cut(runs: list[tuple[int, int]], windows: BoundaryWindows) -> int | None:
    assert windows.right_window is not None
    candidates = _overlapping_runs(runs, windows.right_window)
    if not candidates:
        return None
    run = min(candidates, key=lambda value: max(value[0], windows.right_window[0]))
    return max(run[0], windows.right_window[0])


def _overlapping_runs(
    runs: list[tuple[int, int]],
    window: tuple[int, int],
) -> list[tuple[int, int]]:
    window_start, window_end = window
    return [
        run
        for run in runs
        if min(run[1], window_end) > max(run[0], window_start)
    ]
