from __future__ import annotations

import numpy as np
import pytest
from audiosig import measure_loudness

from pykokoro import LoudnessConfig
from pykokoro.output_loudness import LoudnessNormalizationError, apply_complete_output_loudness
from pykokoro.types import Trace
from pykokoro.voice_level import (
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelCalibration,
    apply_voice_level_calibration,
)

SAMPLE_RATE = 24_000


def _audio(amplitude: float = 0.2) -> np.ndarray:
    time = np.arange(SAMPLE_RATE * 2, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


def test_one_static_gain_preserves_relative_sections() -> None:
    audio = np.concatenate([_audio(0.1), _audio(0.2)])
    before = measure_loudness(audio, sample_rate=SAMPLE_RATE)
    result = apply_complete_output_loudness(audio, SAMPLE_RATE, LoudnessConfig(target_lufs=-20.0))
    expected_gain = 10 ** ((-20.0 - before.integrated_lufs) / 20.0)
    np.testing.assert_allclose(result, audio * expected_gain, rtol=1e-5, atol=1e-6)


def test_silence_is_unchanged() -> None:
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    trace = Trace()
    result = apply_complete_output_loudness(
        audio, SAMPLE_RATE, LoudnessConfig(target_lufs=-18.0), trace
    )
    np.testing.assert_array_equal(result, audio)
    assert trace.warnings


def test_reduce_gain_respects_true_peak_ceiling() -> None:
    result = apply_complete_output_loudness(
        _audio(0.2), SAMPLE_RATE, LoudnessConfig(target_lufs=0.0, true_peak_ceiling_dbtp=-10.0)
    )
    assert measure_loudness(result, sample_rate=SAMPLE_RATE).true_peak_dbtp <= -10.0 + 0.05


def test_error_policy_reports_peak_context() -> None:
    with pytest.raises(LoudnessNormalizationError, match="measured LUFS.*target LUFS.*true peak"):
        apply_complete_output_loudness(
            _audio(0.2),
            SAMPLE_RATE,
            LoudnessConfig(target_lufs=0.0, true_peak_ceiling_dbtp=-10.0, peak_policy="error"),
        )


def test_attenuation_is_not_blocked_by_peak_ceiling() -> None:
    audio = _audio(0.2)
    current = measure_loudness(audio, sample_rate=SAMPLE_RATE).integrated_lufs
    result = apply_complete_output_loudness(
        audio, SAMPLE_RATE, LoudnessConfig(target_lufs=current - 3.0, true_peak_ceiling_dbtp=-40.0)
    )
    assert measure_loudness(result, sample_rate=SAMPLE_RATE).integrated_lufs == pytest.approx(
        current - 3.0, abs=0.1
    )


def test_disabled_target_returns_same_object() -> None:
    audio = _audio()
    assert apply_complete_output_loudness(audio, SAMPLE_RATE, LoudnessConfig()) is audio


def test_static_voice_leveling_composes_with_complete_output_normalization() -> None:
    key = VoiceCalibrationKey("github", "v1.0", "fp32", "voice")
    catalog = VoiceCalibrationCatalog(
        schema=1,
        method="bs1770",
        corpus="test",
        reference_lufs=-24.0,
        generated_with={},
        voices={key: VoiceLevelCalibration(gain_db=-3.0, samples=3)},
    )
    statically_leveled = apply_voice_level_calibration(
        _audio(),
        LoudnessConfig(voice_leveling="calibrated"),
        key,
        catalog=catalog,
    )
    result = apply_complete_output_loudness(
        statically_leveled,
        SAMPLE_RATE,
        LoudnessConfig(target_lufs=-24.0),
    )
    assert measure_loudness(result, sample_rate=SAMPLE_RATE).integrated_lufs == pytest.approx(
        -24.0, abs=0.1
    )
