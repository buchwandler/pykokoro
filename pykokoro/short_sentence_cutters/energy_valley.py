"""Energy-valley phrase cutter for short-sentence extraction."""

from __future__ import annotations

import numpy as np

from pykokoro.constants import SAMPLE_RATE

from .shared import BoundaryWindows, boundary_windows_from_metadata


def cut_with_energy_valley(
    audio: np.ndarray,
    metadata: dict[str, object],
) -> np.ndarray | None:
    """Cut phrase audio at stable low-energy valleys inside legal timestamp windows."""
    cut_bounds = find_energy_valley_cut_bounds(audio, metadata)
    if cut_bounds is None:
        return None
    left_cut, right_cut = cut_bounds
    return audio[left_cut:right_cut]


def find_energy_valley_cut_bounds(
    audio: np.ndarray,
    metadata: dict[str, object],
) -> tuple[int, int] | None:
    """Return production energy-valley cut bounds without slicing audio."""
    windows = boundary_windows_from_metadata(len(audio), metadata)
    if windows is None:
        return None

    frame_duration_ms = int(metadata.get("frame_duration_ms", 5))
    frame_length = max(1, int(SAMPLE_RATE * frame_duration_ms / 1000))
    energy = _normalized_frame_energy(audio, frame_length)
    if energy.size == 0:
        return None
    min_frames = max(
        1,
        int(float(metadata.get("min_silence_seconds", 0.02)) * 1000 / frame_duration_ms),
    )
    threshold = float(metadata.get("energy_threshold", 0.05))

    left_cut = (
        _valley_cut(
            energy,
            frame_length,
            windows.left_window,
            min_frames,
            threshold,
            "left",
        )
        if windows.has_left_context
        else 0
    )
    right_cut = (
        _valley_cut(
            energy,
            frame_length,
            windows.right_window,
            min_frames,
            threshold,
            "right",
        )
        if windows.has_right_context
        else len(audio)
    )
    if left_cut is None or right_cut is None or right_cut <= left_cut:
        return None
    return left_cut, right_cut


def _normalized_frame_energy(audio: np.ndarray, frame_length: int) -> np.ndarray:
    if audio.size == 0:
        return np.array([], dtype=np.float32)
    frame_count = max(1, int(np.ceil(len(audio) / frame_length)))
    padded = np.zeros(frame_count * frame_length, dtype=np.float32)
    padded[: len(audio)] = audio.astype(np.float32)
    frames = padded.reshape(frame_count, frame_length)
    energy = np.sqrt(np.mean(frames**2, axis=1))
    span = float(energy.max() - energy.min())
    if span <= 1e-8:
        return np.zeros_like(energy)
    return (energy - energy.min()) / span


def _valley_cut(
    energy: np.ndarray,
    frame_length: int,
    window: tuple[int, int] | None,
    min_frames: int,
    threshold: float,
    side: str,
) -> int | None:
    if window is None:
        return None
    window_start, window_end = window
    first_frame = max(0, window_start // frame_length)
    last_frame = min(len(energy) - 1, max(first_frame, (window_end - 1) // frame_length))
    if last_frame < first_frame:
        return None

    window_frame_count = last_frame - first_frame + 1
    if window_frame_count < min_frames:
        return None
    local_energy = energy[first_frame : last_frame + 1]
    kernel = np.ones(min_frames, dtype=np.float32) / min_frames
    rolling_mean = np.convolve(local_energy, kernel, mode="valid")
    qualifying_offsets = np.flatnonzero(rolling_mean <= threshold)
    if qualifying_offsets.size == 0:
        return None
    best_offset = int(qualifying_offsets[-1] if side == "left" else qualifying_offsets[0])

    quiet_start = first_frame + best_offset
    quiet_end = quiet_start + min_frames
    run_start = quiet_start * frame_length
    run_end = quiet_end * frame_length
    if min(run_end, window_end) <= max(run_start, window_start):
        return None
    if side == "left":
        return min(run_end, window_end)
    return max(run_start, window_start)
