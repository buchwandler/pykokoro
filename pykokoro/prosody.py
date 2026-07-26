"""Prosody audio processing for PyKokoro.

This module provides functions to apply volume, pitch, and rate modifications
to audio based on SSMD prosody metadata.

Audio Processing Libraries:
    - audiomentations: Preferred for pitch/rate (highest quality)
    - librosa: Fallback for pitch/rate
    - numpy: Volume control (no external dependency)

Supports both absolute values (e.g., 'loud', 'fast', 'high') and relative values
(e.g., '+6dB', '+20%', '+2st').
"""

import logging
import re

import numpy as np

from .constants import PITCH_ABSOLUTE_MAP, RATE_ABSOLUTE_MAP, VOLUME_ABSOLUTE_MAP

# Try importing audio processing libraries
try:
    from audiomentations import PitchShift, TimeStretch

    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    AUDIOMENTATIONS_AVAILABLE = False

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


def parse_volume(volume_str: str) -> float:
    """Parse volume value to decibels (dB).

    Args:
        volume_str: Volume specification, can be:
            - Absolute: 'silent', 'x-soft', 'soft', 'medium', 'loud', 'x-loud'
            - Relative dB: '+6dB', '-3dB', '6dB'
            - Relative percentage: '+20%', '-10%', '120%'

    Returns:
        Volume change in decibels (0.0 = no change)

    Raises:
        ValueError: If the volume string format is invalid
    """
    volume_str = volume_str.strip().lower()

    # Check absolute values
    if volume_str in VOLUME_ABSOLUTE_MAP:
        return VOLUME_ABSOLUTE_MAP[volume_str]

    # Check for dB notation (e.g., '+6dB', '-3dB', '6dB')
    db_match = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*db$", volume_str)
    if db_match:
        return float(db_match.group(1))

    # Check for percentage (e.g., '+20%', '-10%', '120%')
    pct_match = re.match(r"^([+-]?\d+(?:\.\d+)?)%$", volume_str)
    if pct_match:
        pct_value = float(pct_match.group(1))
        # If no sign, treat as absolute percentage (120% = +20%)
        if not volume_str.startswith(("+", "-")):
            pct_value = pct_value - 100
        # Convert percentage to dB: 20*log10(1 + pct/100)
        multiplier = 1.0 + (pct_value / 100.0)
        if multiplier <= 0:
            return -float("inf")
        return 20 * np.log10(multiplier)

    msg = f"Invalid volume format: '{volume_str}'"
    raise ValueError(msg)


def parse_rate(rate_str: str) -> float:
    """Parse rate value to speed multiplier.

    Args:
        rate_str: Rate specification, can be:
            - Absolute: 'x-slow', 'slow', 'medium', 'fast', 'x-fast'
            - Percentage: '+20%', '-10%', '120%', '80%'

    Returns:
        Speed multiplier (1.0 = normal speed, 2.0 = double speed, 0.5 = half speed)

    Raises:
        ValueError: If the rate string format is invalid
    """
    rate_str = rate_str.strip().lower()

    # Check absolute values
    if rate_str in RATE_ABSOLUTE_MAP:
        return RATE_ABSOLUTE_MAP[rate_str]

    # Check for percentage (e.g., '+20%', '-10%', '120%')
    pct_match = re.match(r"^([+-]?\d+(?:\.\d+)?)%$", rate_str)
    if pct_match:
        pct_value = float(pct_match.group(1))
        # If no sign, treat as absolute percentage (120% = 1.2x speed)
        if not rate_str.startswith(("+", "-")):
            return pct_value / 100.0
        # If signed, treat as relative ('+20%' = 1.2x speed)
        return 1.0 + (pct_value / 100.0)

    msg = f"Invalid rate format: '{rate_str}'"
    raise ValueError(msg)


def parse_pitch(pitch_str: str) -> float:
    """Parse pitch value to semitones.

    Args:
        pitch_str: Pitch specification, can be:
            - Absolute: 'x-low', 'low', 'medium', 'high', 'x-high'
            - Relative semitones: '+2st', '-1.5st', '2st'
            - Relative percentage: '+10%', '-5%'

    Returns:
        Pitch change in semitones (0.0 = no change)

    Raises:
        ValueError: If the pitch string format is invalid
    """
    pitch_str = pitch_str.strip().lower()

    # Check absolute values
    if pitch_str in PITCH_ABSOLUTE_MAP:
        return PITCH_ABSOLUTE_MAP[pitch_str]

    # Check for semitones (e.g., '+2st', '-1.5st', '2st')
    st_match = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*st$", pitch_str)
    if st_match:
        return float(st_match.group(1))

    # Check for percentage (e.g., '+10%', '-5%')
    # Convert percentage to semitones: 12*log2(1 + pct/100)
    pct_match = re.match(r"^([+-]?\d+(?:\.\d+)?)%$", pitch_str)
    if pct_match:
        pct_value = float(pct_match.group(1))
        # If no sign, treat as absolute percentage (110% = +10%)
        if not pitch_str.startswith(("+", "-")):
            pct_value = pct_value - 100
        multiplier = 1.0 + (pct_value / 100.0)
        if multiplier <= 0:
            return 0.0
        return 12 * np.log2(multiplier)

    msg = f"Invalid pitch format: '{pitch_str}'"
    raise ValueError(msg)


def apply_volume(audio: np.ndarray, volume: str) -> np.ndarray:
    """Apply volume change to audio.

    Args:
        audio: Input audio array
        volume: Volume specification (see parse_volume for formats)

    Returns:
        Audio with volume adjustment applied
    """
    try:
        db_change = parse_volume(volume)

        # Handle silent case
        if db_change == -float("inf"):
            return np.zeros_like(audio)

        # Convert dB to amplitude multiplier: 10^(dB/20)
        amplitude_multiplier = 10 ** (db_change / 20.0)

        return audio * amplitude_multiplier

    except ValueError as e:
        logger.warning(f"Failed to apply volume '{volume}': {e}")
        return audio


def apply_pitch(audio: np.ndarray, pitch: str, sample_rate: int) -> np.ndarray:
    """Apply pitch shift to audio using audiomentations or librosa.

    Prefers audiomentations for higher quality, falls back to librosa.

    Args:
        audio: Input audio array
        pitch: Pitch specification (see parse_pitch for formats)
        sample_rate: Audio sample rate in Hz

    Returns:
        Audio with pitch shift applied
    """
    if not AUDIOMENTATIONS_AVAILABLE and not LIBROSA_AVAILABLE:
        logger.warning(
            "audiomentations/librosa not available, pitch shifting disabled. "
            "Install with: pip install pykokoro[prosody]"
        )
        return audio

    try:
        semitones = parse_pitch(pitch)

        # No change needed
        if abs(semitones) < 0.01:
            return audio

        # Use audiomentations if available (higher quality)
        if AUDIOMENTATIONS_AVAILABLE:
            augmenter = PitchShift(
                min_semitones=semitones,
                max_semitones=semitones,
                p=1.0,
            )
            shifted = augmenter(samples=audio, sample_rate=sample_rate)
            return shifted.astype(audio.dtype)

        # Fall back to librosa
        if LIBROSA_AVAILABLE:
            shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)
            return shifted.astype(audio.dtype)

        return audio

    except ValueError as e:
        logger.warning(f"Failed to apply pitch '{pitch}': {e}")
        return audio
    except (RuntimeError, OSError) as e:
        logger.warning(f"Pitch shift failed for '{pitch}': {e}")
        return audio


def apply_rate(audio: np.ndarray, rate: str, sample_rate: int = 24000) -> np.ndarray:
    """Apply speed/rate change to audio using audiomentations or librosa.

    Prefers audiomentations with signalsmith_stretch for highest quality,
    falls back to librosa.

    Args:
        audio: Input audio array
        rate: Rate specification (see parse_rate for formats)
        sample_rate: Audio sample rate in Hz (default: 24000)

    Returns:
        Audio with speed adjustment applied (length will change)
    """
    if not AUDIOMENTATIONS_AVAILABLE and not LIBROSA_AVAILABLE:
        logger.warning(
            "audiomentations/librosa not available, rate adjustment disabled. "
            "Install with: pip install pykokoro[prosody]"
        )
        return audio

    try:
        speed_multiplier = parse_rate(rate)

        # No change needed
        if abs(speed_multiplier - 1.0) < 0.01:
            return audio

        # Use audiomentations with signalsmith_stretch (highest quality)
        if AUDIOMENTATIONS_AVAILABLE:
            try:
                augmenter = TimeStretch(
                    min_rate=speed_multiplier,
                    max_rate=speed_multiplier,
                    leave_length_unchanged=False,
                    p=1.0,
                )
                stretched = augmenter(samples=audio, sample_rate=sample_rate)
                return stretched.astype(audio.dtype)
            except (RuntimeError, OSError, ValueError) as e:
                logger.debug(f"Audiomentations TimeStretch failed, falling back to librosa: {e}")

        # Fall back to librosa
        if LIBROSA_AVAILABLE:
            stretched = librosa.effects.time_stretch(audio, rate=speed_multiplier)
            return stretched.astype(audio.dtype)

        return audio

    except ValueError as e:
        logger.warning(f"Failed to apply rate '{rate}': {e}")
        return audio
    except (RuntimeError, OSError) as e:
        logger.warning(f"Rate adjustment failed for '{rate}': {e}")
        return audio


def apply_prosody(
    audio: np.ndarray,
    sample_rate: int,
    volume: str | None = None,
    pitch: str | None = None,
    rate: str | None = None,
) -> np.ndarray:
    """Apply all prosody modifications to audio.

    Order of operations:
    1. Pitch shift (preserves duration)
    2. Rate change (changes duration)
    3. Volume adjustment

    Args:
        audio: Input audio array
        sample_rate: Audio sample rate in Hz
        volume: Optional volume specification
        pitch: Optional pitch specification
        rate: Optional rate specification

    Returns:
        Audio with all prosody modifications applied
    """
    result = audio

    # Apply pitch shift first (preserves duration)
    if pitch:
        result = apply_pitch(result, pitch, sample_rate)

    # Apply rate change (changes duration)
    if rate:
        result = apply_rate(result, rate, sample_rate)

    # Apply volume last (doesn't affect duration)
    if volume:
        result = apply_volume(result, volume)

    return result
