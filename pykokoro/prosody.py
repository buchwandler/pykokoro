"""Prosody audio processing for PyKokoro through AudioSig.

This module provides functions to apply volume, pitch, and rate modifications
to audio based on SSMD prosody metadata.

Supports both absolute values (e.g., 'loud', 'fast', 'high') and relative values
(e.g., '+6dB', '+20%', '+2st').
"""

from __future__ import annotations

import logging
import re

import numpy as np
from audiosig import AudioSignalError, apply_gain_db, pitch_shift, time_stretch

from .constants import PITCH_ABSOLUTE_MAP, RATE_ABSOLUTE_MAP, VOLUME_ABSOLUTE_MAP

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
    """Apply an SSMD volume value through AudioSig."""
    try:
        db_change = parse_volume(volume)
        if db_change == 0.0:
            return audio
        return apply_gain_db(audio, db_change)
    except (ValueError, AudioSignalError) as exc:
        logger.warning("Failed to apply volume '%s': %s", volume, exc)
        return audio


def apply_pitch(audio: np.ndarray, pitch: str, sample_rate: int) -> np.ndarray:
    """Apply an SSMD pitch value through AudioSig."""
    try:
        semitones = parse_pitch(pitch)
        if abs(semitones) < 0.01:
            return audio
        return pitch_shift(
            audio,
            sample_rate=sample_rate,
            semitones=semitones,
        )
    except (ValueError, AudioSignalError) as exc:
        logger.warning("Failed to apply pitch '%s': %s", pitch, exc)
        return audio


def apply_rate(audio: np.ndarray, rate: str, sample_rate: int = 24000) -> np.ndarray:
    """Apply a pitch-preserving SSMD rate value through AudioSig."""
    del sample_rate  # Kept for public API compatibility.
    try:
        speed_multiplier = parse_rate(rate)
        if abs(speed_multiplier - 1.0) < 0.01:
            return audio
        return time_stretch(audio, speed_multiplier)
    except (ValueError, AudioSignalError) as exc:
        logger.warning("Failed to apply rate '%s': %s", rate, exc)
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
