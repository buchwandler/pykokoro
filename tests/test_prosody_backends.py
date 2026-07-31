"""Tests for AudioSig prosody boundaries and forbidden backend imports."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
from audiosig import InvalidParameterError

from pykokoro import prosody


def test_prosody_import_has_no_heavy_dsp_backends() -> None:
    code = """
import json
import sys
import pykokoro.prosody

forbidden = {
    "librosa",
    "scipy",
    "sklearn",
    "audiomentations",
    "signalsmith_stretch",
    "python_stretch",
    "numba",
    "resampy",
    "soxr",
    "torchaudio",
}
print(json.dumps(sorted(forbidden.intersection(sys.modules))))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_rate_audiosig_failure_is_fail_open(monkeypatch, caplog) -> None:
    audio = np.arange(100, dtype=np.float32)

    def fail(*args, **kwargs):
        raise InvalidParameterError("forced failure")

    monkeypatch.setattr(prosody, "time_stretch", fail)

    result = prosody.apply_rate(audio, "fast")

    np.testing.assert_array_equal(result, audio)
    assert "Failed to apply rate" in caplog.text


def test_audio_signal_failure_is_fail_open_for_pitch(monkeypatch, caplog) -> None:
    audio = np.arange(100, dtype=np.float32)

    def fail(*args, **kwargs):
        raise InvalidParameterError("forced failure")

    monkeypatch.setattr(prosody, "pitch_shift", fail)

    result = prosody.apply_pitch(audio, "+2st", 24000)

    np.testing.assert_array_equal(result, audio)
    assert "Failed to apply pitch" in caplog.text


def test_backend_availability_constants_are_removed() -> None:
    assert not hasattr(prosody, "AUDIOMENTATIONS_AVAILABLE")
    assert not hasattr(prosody, "LIBROSA_AVAILABLE")
