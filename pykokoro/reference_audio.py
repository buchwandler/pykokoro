"""Reference speech loading, normalization, and resampling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .exceptions import InvalidVoiceError

_REFERENCE_SAMPLE_RATE = 24_000
_IDENTITY_SAMPLE_RATE = 16_000
_MIN_DURATION_SECONDS = 3.0
_MAX_DURATION_SECONDS = 30.0
_MIN_RMS = 1.0e-4
_CLIPPING_THRESHOLD = 0.999
_MAX_CLIPPED_FRACTION = 0.001


@dataclass(frozen=True, slots=True)
class PreparedReferenceAudio:
    """Canonical audio arrays and non-content metadata for voice enrollment."""

    audio_24k: np.ndarray
    audio_16k: np.ndarray
    original_sample_rate: int
    duration_seconds: float
    audio_sha256: str


def prepare_reference_audio(
    reference_audio: str | Path | np.ndarray,
    *,
    sample_rate: int | None = None,
) -> PreparedReferenceAudio:
    """Load reference speech and prepare mono float32 arrays at 24 kHz and 16 kHz."""
    if isinstance(reference_audio, (str, Path)):
        if sample_rate is not None:
            raise InvalidVoiceError("sample_rate is only accepted with NumPy reference audio")
        try:
            audio, original_sample_rate = sf.read(reference_audio, dtype="float32", always_2d=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise InvalidVoiceError(
                f"could not read reference audio {reference_audio!s}: {exc}"
            ) from exc
        mono = _to_mono_float32(audio)
    else:
        if sample_rate is None:
            raise InvalidVoiceError("sample_rate is required for NumPy reference audio")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise InvalidVoiceError("sample_rate must be a positive integer")
        original_sample_rate = sample_rate
        mono = _to_mono_float32(reference_audio)

    digest = hashlib.sha256()
    digest.update(str(original_sample_rate).encode("ascii"))
    digest.update(b"\0")
    digest.update(mono.tobytes(order="C"))

    audio_24k = _resample(mono, original_sample_rate, _REFERENCE_SAMPLE_RATE)
    duration_seconds = audio_24k.size / _REFERENCE_SAMPLE_RATE
    if not _MIN_DURATION_SECONDS <= duration_seconds <= _MAX_DURATION_SECONDS:
        raise InvalidVoiceError("reference audio duration must be between 3 and 30 seconds")
    audio_16k = _resample(audio_24k, _REFERENCE_SAMPLE_RATE, _IDENTITY_SAMPLE_RATE)

    for audio, label in ((audio_24k, "24 kHz"), (audio_16k, "16 kHz")):
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if rms < _MIN_RMS:
            raise InvalidVoiceError(
                f"reference audio is silent or too quiet after {label} resampling"
            )
        clipped_fraction = float(np.mean(np.abs(audio) >= _CLIPPING_THRESHOLD))
        if clipped_fraction > _MAX_CLIPPED_FRACTION:
            raise InvalidVoiceError(f"reference audio is severely clipped after {label} resampling")

    return PreparedReferenceAudio(
        audio_24k=audio_24k,
        audio_16k=audio_16k,
        original_sample_rate=original_sample_rate,
        duration_seconds=duration_seconds,
        audio_sha256=digest.hexdigest(),
    )


def _to_mono_float32(value: np.ndarray) -> np.ndarray:
    audio = np.asarray(value)
    if (
        audio.dtype.hasobject
        or not np.issubdtype(audio.dtype, np.number)
        or np.issubdtype(audio.dtype, np.complexfloating)
    ):
        raise InvalidVoiceError("reference audio must contain real numeric samples")
    if audio.ndim not in (1, 2) or audio.size == 0:
        raise InvalidVoiceError(
            "reference audio must be a non-empty mono or multi-channel waveform"
        )
    if audio.ndim == 2:
        if 0 in audio.shape:
            raise InvalidVoiceError("reference audio must have at least one sample and one channel")
        audio = np.mean(audio, axis=1, dtype=np.float64)
    mono = np.ascontiguousarray(audio, dtype=np.float32)
    if not np.isfinite(mono).all():
        raise InvalidVoiceError("reference audio must contain only finite samples")
    return mono


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.array(audio, dtype=np.float32, order="C", copy=True)
    try:
        result = soxr.resample(audio, source_rate, target_rate, quality="HQ")
    except (ValueError, RuntimeError) as exc:
        raise InvalidVoiceError(
            f"could not resample reference audio to {target_rate} Hz: {exc}"
        ) from exc
    result = np.ascontiguousarray(result, dtype=np.float32)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise InvalidVoiceError(
            f"resampling reference audio to {target_rate} Hz produced invalid samples"
        )
    return result
