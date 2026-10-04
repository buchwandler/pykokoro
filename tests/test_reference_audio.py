from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import soxr

from pykokoro.exceptions import InvalidVoiceError
from pykokoro.reference_audio import prepare_reference_audio


def tone(sample_rate: int, duration: float = 3.5) -> np.ndarray:
    times = np.arange(round(sample_rate * duration), dtype=np.float64) / sample_rate
    return (0.15 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)


def test_prepare_reference_audio_reads_wav_and_downmixes_stereo(tmp_path: Path) -> None:
    mono = tone(24_000)
    stereo = np.column_stack((mono, mono * 0.5))
    path = tmp_path / "reference.wav"
    sf.write(path, stereo, 24_000, subtype="FLOAT")

    prepared = prepare_reference_audio(path)

    assert prepared.original_sample_rate == 24_000
    np.testing.assert_allclose(prepared.mono_audio, mono * 0.75, atol=1.0e-6)
    assert prepared.duration_seconds == pytest.approx(3.5, abs=1 / 24_000)
    np.testing.assert_allclose(prepared.audio_24k, mono * 0.75, atol=1.0e-6)
    assert prepared.audio_16k.shape == (round(prepared.audio_24k.size * 2 / 3),)
    assert prepared.audio_24k.dtype == np.float32
    assert prepared.audio_16k.dtype == np.float32
    assert len(prepared.audio_sha256) == 64


def test_prepare_reference_audio_resamples_to_24k_then_16k() -> None:
    source = tone(44_100)

    prepared = prepare_reference_audio(source, sample_rate=44_100)

    assert prepared.original_sample_rate == 44_100
    assert prepared.audio_24k.size == round(source.size * 24_000 / 44_100)
    assert prepared.audio_16k.size == round(prepared.audio_24k.size * 16_000 / 24_000)
    expected_16k = soxr.resample(prepared.audio_24k, 24_000, 16_000, quality="HQ")
    np.testing.assert_array_equal(prepared.audio_16k, expected_16k)
    assert abs(prepared.audio_24k.size / 24_000 - source.size / 44_100) < 1 / 24_000


def test_soxr_24k_to_16k_stays_close_to_upstream_torchaudio() -> None:
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    times = np.arange(24_000 * 3, dtype=np.float64) / 24_000
    audio_24k = (
        0.2 * np.sin(2.0 * np.pi * 220.0 * times) + 0.05 * np.sin(2.0 * np.pi * 713.0 * times)
    ).astype(np.float32)

    actual = soxr.resample(audio_24k, 24_000, 16_000, quality="HQ")
    expected = torchaudio.functional.resample(torch.from_numpy(audio_24k), 24_000, 16_000).numpy()
    error = actual - expected

    assert actual.shape == expected.shape
    assert np.sqrt(np.mean(np.square(error))) / np.sqrt(np.mean(np.square(expected))) < 5.0e-4
    assert np.max(np.abs(error)) < 3.0e-4


def test_reference_audio_hash_is_stable_and_content_sensitive() -> None:
    audio = tone(24_000)

    first = prepare_reference_audio(audio, sample_rate=24_000)
    second = prepare_reference_audio(audio.copy(), sample_rate=24_000)
    changed = audio.copy()
    changed[0] += 0.01
    third = prepare_reference_audio(changed, sample_rate=24_000)

    assert first.audio_sha256 == second.audio_sha256
    assert first.audio_sha256 != third.audio_sha256


def test_numpy_reference_audio_requires_explicit_sample_rate() -> None:
    with pytest.raises(InvalidVoiceError, match="sample_rate is required"):
        prepare_reference_audio(tone(24_000))
    with pytest.raises(InvalidVoiceError, match="only accepted with NumPy"):
        prepare_reference_audio("reference.wav", sample_rate=24_000)
    with pytest.raises(InvalidVoiceError, match="positive integer"):
        prepare_reference_audio(tone(24_000), sample_rate=True)


@pytest.mark.parametrize(
    ("audio", "sample_rate", "message"),
    [
        (np.zeros(24_000 * 4, dtype=np.float32), 24_000, "silent or too quiet"),
        (np.ones(24_000 * 4, dtype=np.float32), 24_000, "severely clipped"),
        (np.full((24_000 * 4,), np.nan, dtype=np.float32), 24_000, "finite samples"),
        (np.zeros((2, 3, 4), dtype=np.float32), 24_000, "mono or multi-channel"),
        (tone(24_000, 2.9), 24_000, "between 3 and 30 seconds"),
        (tone(24_000, 30.1), 24_000, "between 3 and 30 seconds"),
    ],
)
def test_prepare_reference_audio_rejects_invalid_input(
    audio: np.ndarray, sample_rate: int, message: str
) -> None:
    with pytest.raises(InvalidVoiceError, match=message):
        prepare_reference_audio(audio, sample_rate=sample_rate)
