"""Tests for the downstream trim, framing, and VAD contracts."""

import numpy as np
import pytest
from audiosig import InvalidParameterError, frame_rms, frame_signal, normalized_energy_vad, trim

from pykokoro.short_sentence_cutters.energy_valley import _normalized_frame_energy
from pykokoro.short_sentence_cutters.vad import _quiet_runs


@pytest.mark.parametrize(
    ("length", "expected_frames"),
    [(0, 0), (1, 1), (9, 1), (10, 1), (11, 1)],
)
def test_normalized_energy_vad_short_inputs(length, expected_frames):
    audio = np.ones(length, dtype=np.float32)
    voice_activity = normalized_energy_vad(
        audio,
        sample_rate=1000,
        frame_duration_ms=10,
        energy_threshold=0.5,
    )
    assert voice_activity.shape == (expected_frames,)


@pytest.mark.parametrize(
    ("length", "expected_frames"),
    [(0, 0), (1, 1), (9, 1), (10, 1), (11, 1)],
)
def test_frame_signal_short_inputs(length, expected_frames):
    audio = np.arange(length, dtype=np.float32)
    frames = frame_signal(
        audio,
        sample_rate=1000,
        frame_ms=10,
        hop_ms=5,
    )
    assert frames.shape == (expected_frames, 10)

    if length > 0:
        assert frames[0, 0] == audio[0]


def test_frame_signal_invalid_params():
    audio = np.ones(10, dtype=np.float32)
    with pytest.raises(InvalidParameterError, match="frame_ms"):
        frame_signal(audio, sample_rate=1000, frame_ms=0, hop_ms=10)
    with pytest.raises(InvalidParameterError, match="hop_ms"):
        frame_signal(audio, sample_rate=1000, frame_ms=10, hop_ms=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_trim_empty_and_active_inputs(dtype):
    empty = np.array([], dtype=dtype)
    trimmed, empty_bounds = trim(empty)
    assert trimmed.shape == empty.shape
    assert empty_bounds.shape == (2,)

    audio = np.concatenate(
        [np.zeros(20, dtype=dtype), np.ones(40, dtype=dtype), np.zeros(20, dtype=dtype)]
    )
    trimmed, bounds = trim(audio, frame_length=10, hop_length=5, top_db=20)
    assert bounds[0] >= 0
    assert bounds[1] <= len(audio)
    np.testing.assert_array_equal(trimmed, audio[bounds[0] : bounds[1]])


def test_trim_fixed_reference_all_zero_input_is_empty():
    audio = np.zeros(32, dtype=np.float32)
    trimmed, bounds = trim(audio, ref=1.0)
    assert trimmed.size == 0
    np.testing.assert_array_equal(bounds, np.array([0, 0]))


def test_frame_signal_non_final_axis():
    audio = np.arange(24, dtype=np.float32).reshape(2, 12)
    frames = frame_signal(audio, frame_length=4, hop_length=4, axis=1)
    assert frames.shape == (2, 3, 4)
    np.testing.assert_array_equal(frames[0, 0], audio[0, :4])


@pytest.mark.parametrize("length", [0, 1, 19, 20, 21, 61])
def test_energy_valley_rms_matches_audiosig_contract(length):
    source = np.linspace(-1.0, 1.0, length, dtype=np.float32)
    expected = frame_rms(
        source,
        frame_length=20,
        hop_length=20,
        center=False,
        pad_end=True,
        normalize=True,
        dtype=np.float32,
    )
    np.testing.assert_array_equal(_normalized_frame_energy(source, 20), expected)


def test_quiet_runs_handles_empty_and_trailing_partial_frames():
    assert (
        _quiet_runs(
            np.array([], dtype=np.float32),
            frame_duration_ms=5,
            energy_threshold=0.05,
            min_silence_seconds=0.005,
        )
        == []
    )
    assert _quiet_runs(
        np.zeros(1, dtype=np.float32),
        frame_duration_ms=5,
        energy_threshold=0.05,
        min_silence_seconds=0.005,
    ) == [(0, 1)]
    assert _quiet_runs(
        np.zeros(12, dtype=np.float32),
        frame_duration_ms=5,
        energy_threshold=0.05,
        min_silence_seconds=0.005,
    ) == [(0, 12)]


def test_quiet_runs_filters_short_middle_run():
    audio = np.concatenate(
        [np.ones(5, dtype=np.float32), np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32)]
    )
    assert (
        _quiet_runs(
            audio,
            frame_duration_ms=5,
            energy_threshold=0.05,
            min_silence_seconds=0.01,
        )
        == []
    )
