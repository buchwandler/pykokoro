"""VAD-run phrase cutter for short-sentence extraction."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from audiosig import activity_to_intervals, normalized_energy_vad

from pykokoro.constants import SAMPLE_RATE

from .shared import BoundaryWindows, boundary_windows_from_metadata


def cut_with_vad(audio: np.ndarray, metadata: dict[str, object]) -> np.ndarray | None:
    """Cut phrase audio at quiet runs that overlap legal timestamp windows."""
    windows = boundary_windows_from_metadata(len(audio), metadata)
    if windows is None:
        return None

    runs = _quiet_runs(
        audio,
        frame_duration_ms=int(cast(Any, metadata.get("frame_duration_ms", 5))),
        energy_threshold=float(cast(Any, metadata.get("energy_threshold", 0.05))),
        min_silence_seconds=float(cast(Any, metadata.get("min_silence_seconds", 0.02))),
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
    speech_frames = normalized_energy_vad(
        audio,
        SAMPLE_RATE,
        frame_duration_ms=frame_duration_ms,
        energy_threshold=energy_threshold,
    )
    samples_per_frame = max(1, int(SAMPLE_RATE * frame_duration_ms / 1000))
    min_frames = max(1, int(min_silence_seconds * 1000 / frame_duration_ms))
    intervals = activity_to_intervals(
        ~speech_frames,
        hop_length=samples_per_frame,
        sample_count=len(audio),
        min_frames=min_frames,
    )
    runs = [(int(start), int(end)) for start, end in intervals]

    represented_samples = len(speech_frames) * samples_per_frame
    if (
        runs
        and speech_frames.size
        and not speech_frames[-1]
        and runs[-1][1] == represented_samples
    ):
        runs[-1] = (runs[-1][0], len(audio))
    return runs


def _left_cut(runs: list[tuple[int, int]], windows: BoundaryWindows) -> int | None:
    assert windows.left_window is not None
    candidates = _overlapping_runs(runs, windows.left_window)
    if not candidates:
        return None
    window = windows.left_window
    assert window is not None
    run = max(candidates, key=lambda value: min(value[1], window[1]))
    return min(run[1], window[1])


def _right_cut(runs: list[tuple[int, int]], windows: BoundaryWindows) -> int | None:
    assert windows.right_window is not None
    candidates = _overlapping_runs(runs, windows.right_window)
    if not candidates:
        return None
    window = windows.right_window
    assert window is not None
    run = min(candidates, key=lambda value: max(value[0], window[0]))
    return max(run[0], window[0])


def _overlapping_runs(
    runs: list[tuple[int, int]],
    window: tuple[int, int],
) -> list[tuple[int, int]]:
    window_start, window_end = window
    return [run for run in runs if min(run[1], window_end) > max(run[0], window_start)]
