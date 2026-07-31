"""Tests for AudioSig prosody boundaries and forbidden backend imports."""

from __future__ import annotations

import json
import logging
import subprocess
import sys

import numpy as np
import pytest
from audiosig import InvalidParameterError

from pykokoro import ProsodyConfig, prosody


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


def test_selected_backend_failure_is_fail_open(monkeypatch, caplog) -> None:
    audio = np.arange(100, dtype=np.float32)

    def fail(*args, **kwargs):
        raise InvalidParameterError("forced failure")

    monkeypatch.setattr(prosody, "apply_speech_effects", fail)

    result = prosody.apply_rate(audio, "fast")

    np.testing.assert_array_equal(result, audio)
    assert "AudioSig prosody method 'wsola' failed" in caplog.text


def test_audio_signal_failure_is_fail_open_for_pitch(monkeypatch, caplog) -> None:
    audio = np.arange(100, dtype=np.float32)

    def fail(*args, **kwargs):
        raise InvalidParameterError("forced failure")

    monkeypatch.setattr(prosody, "apply_speech_effects", fail)

    result = prosody.apply_pitch(audio, "+2st", 24000)

    np.testing.assert_array_equal(result, audio)
    assert "AudioSig prosody method 'wsola' failed" in caplog.text


def test_fallback_succeeds_and_logs_source_and_target(monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO)
    calls = []

    def fake(audio, **kwargs):
        calls.append(kwargs["method"])
        if kwargs["method"] == "esola":
            raise InvalidParameterError("forced failure")
        return np.array(audio, copy=True)

    monkeypatch.setattr(prosody, "apply_speech_effects", fake)
    source = np.arange(100, dtype=np.float32)

    result = prosody.apply_rate(
        source,
        "fast",
        config=ProsodyConfig(method="esola", fallback_methods=("wsola",)),
    )

    np.testing.assert_array_equal(result, source)
    assert calls == ["esola", "wsola"]
    assert "'esola' -> 'wsola'" in caplog.text


def test_duplicate_fallback_methods_are_skipped(monkeypatch) -> None:
    calls = []

    def fake(audio, **kwargs):
        calls.append(kwargs["method"])
        raise InvalidParameterError("forced failure")

    monkeypatch.setattr(prosody, "apply_speech_effects", fake)
    source = np.arange(100, dtype=np.float32)

    result = prosody.apply_rate(
        source,
        "fast",
        config=ProsodyConfig(
            method="psola",
            fallback_methods=("td_psola", "psola", "wsola", "wsola"),
        ),
    )

    np.testing.assert_array_equal(result, source)
    assert calls == ["td_psola", "wsola"]


def test_strict_mode_does_not_fallback(monkeypatch) -> None:
    calls = []

    def fail(audio, **kwargs):
        calls.append(kwargs["method"])
        raise InvalidParameterError("forced failure")

    monkeypatch.setattr(prosody, "apply_speech_effects", fail)

    with pytest.raises(InvalidParameterError, match="forced failure"):
        prosody.apply_rate(
            np.arange(100, dtype=np.float32),
            "fast",
            config=ProsodyConfig(method="esola", fallback_methods=("wsola",), strict=True),
        )

    assert calls == ["esola"]


def test_backend_availability_constants_are_removed() -> None:
    assert not hasattr(prosody, "AUDIOMENTATIONS_AVAILABLE")
    assert not hasattr(prosody, "LIBROSA_AVAILABLE")


def _speech_like_audio() -> np.ndarray:
    sample_rate = 24000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    voiced = 0.18 * np.sin(2 * np.pi * 180 * time)
    harmonic = 0.06 * np.sin(2 * np.pi * 360 * time)
    modulation = 0.7 + 0.3 * np.sin(2 * np.pi * 4 * time)
    return ((voiced + harmonic) * modulation).astype(np.float32)


@pytest.mark.parametrize(
    "method, rate, semitones",
    [
        ("wsola", 0.85, -2.0),
        ("wsola", 1.2, 2.0),
        ("esola", 0.85, -2.0),
        ("esola", 1.2, 2.0),
        ("td_psola", 0.8, -4.0),
        ("td_psola", 1.0, 0.0),
        ("td_psola", 1.25, 4.0),
        ("phase_vocoder", 0.75, -4.0),
        ("phase_vocoder", 1.25, 4.0),
    ],
)
def test_real_audiosig_backend_contract(method: str, rate: float, semitones: float) -> None:
    source = _speech_like_audio()
    original = source.copy()
    config = ProsodyConfig(method=method, fallback_methods=(), strict=True)

    result = prosody.apply_prosody(
        source,
        24000,
        rate=f"{rate * 100:g}%",
        pitch=f"{semitones:+g}st",
        config=config,
    )
    repeat = prosody.apply_prosody(
        source,
        24000,
        rate=f"{rate * 100:g}%",
        pitch=f"{semitones:+g}st",
        config=config,
    )

    assert len(result) == round(len(source) / rate)
    assert result.dtype == source.dtype
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) < 10.0
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(result, repeat)


@pytest.mark.parametrize("method", ["wsola", "esola", "td_psola", "phase_vocoder"])
def test_silence_remains_finite(method: str) -> None:
    source = np.zeros(24000, dtype=np.float32)

    result = prosody.apply_prosody(
        source,
        24000,
        rate="85%",
        pitch="+2st",
        config=ProsodyConfig(method=method, fallback_methods=(), strict=True),
    )

    assert result.dtype == source.dtype
    assert np.isfinite(result).all()
