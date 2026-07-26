"""Voice management for PyKokoro."""

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

DEFAULT_VOICE_STYLE_LENGTH = 512
VOICE_STYLE_TAIL_SHAPE = (1, 256)
VOICE_STYLE_DTYPES = (np.float16, np.float32)


def normalize_voice_style(
    voice_style: np.ndarray,
    *,
    expected_length: int | None = DEFAULT_VOICE_STYLE_LENGTH,
    allow_broadcast: bool = False,
    require_dtype: bool = False,
    voice_name: str | None = None,
) -> np.ndarray:
    """Normalize voice style arrays to (N, 1, 256) with validation.

    Args:
        voice_style: Input voice array.
        expected_length: Optional expected length for the first dimension.
            Use None to accept any length.
        allow_broadcast: If True, a single (1, 256) or (256,) vector can be
            broadcast to expected_length.
        require_dtype: If True, only float16/float32 are accepted.
        voice_name: Optional label for error context.

    Returns:
        Normalized voice style array with shape (N, 1, 256).

    Raises:
        ConfigurationError: If the array is invalid or ambiguous.
    """
    label = f"Voice '{voice_name}'" if voice_name else "Voice style"
    array = np.asarray(voice_style)

    if array.dtype == object:
        raise ConfigurationError(
            f"{label} contains object dtype data. Re-download voices to fix the archive."
        )

    if require_dtype:
        if array.dtype not in VOICE_STYLE_DTYPES:
            raise ConfigurationError(f"{label} must be float16 or float32, got {array.dtype}.")
    else:
        if not np.issubdtype(array.dtype, np.floating):
            raise ConfigurationError(f"{label} must be a floating point array.")

    normalized: np.ndarray
    if array.ndim == 3 and array.shape[1:] == VOICE_STYLE_TAIL_SHAPE:
        normalized = array
    elif array.ndim == 2 and array.shape[1] == VOICE_STYLE_TAIL_SHAPE[-1]:
        normalized = array[:, None, :]
    elif array.ndim == 2 and array.shape == (1, VOICE_STYLE_TAIL_SHAPE[-1]):
        if not allow_broadcast:
            raise ConfigurationError(
                f"{label} has shape {array.shape}. Explicit broadcasting is required."
            )
        if expected_length is None:
            raise ConfigurationError(f"{label} broadcast requires expected_length to be set.")
        normalized = np.tile(array[:, None, :], (expected_length, 1, 1))
    elif array.ndim == 1 and array.shape == (VOICE_STYLE_TAIL_SHAPE[-1],):
        if not allow_broadcast:
            raise ConfigurationError(
                f"{label} has shape {array.shape}. Explicit broadcasting is required."
            )
        if expected_length is None:
            raise ConfigurationError(f"{label} broadcast requires expected_length to be set.")
        normalized = np.tile(array.reshape(1, 1, -1), (expected_length, 1, 1))
    elif array.ndim == 1 and expected_length is not None and array.shape == (expected_length,):
        raise ConfigurationError(
            f"{label} has ambiguous shape {array.shape}. Expected (N, 1, 256)."
        )
    else:
        raise ConfigurationError(
            f"{label} has unsupported shape {array.shape}. Expected (N, 1, 256) or (N, 256)."
        )

    if expected_length is not None and normalized.shape[0] != expected_length:
        raise ConfigurationError(
            f"{label} length {normalized.shape[0]} does not match expected {expected_length}."
        )

    if normalized.dtype not in VOICE_STYLE_DTYPES:
        normalized = normalized.astype(np.float32)

    return normalized


# Model source type
ModelSource = Literal["huggingface", "github"]


def slerp_voices(
    a: np.ndarray,
    b: np.ndarray,
    t: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Spherical linear interpolation (SLERP) between two voice embeddings.

    SLERP interpolates along the shortest arc on a hypersphere, which can
    produce smoother, more natural voice transitions than linear interpolation.
    This is particularly effective for voice blending as it preserves the
    "direction" of the embedding vectors.

    Based on the standard SLERP formula:
        slerp(a, b, t) = sin((1-t)θ)/sin(θ) * a + sin(tθ)/sin(θ) * b
    where θ is the angle between vectors a and b.

    Args:
        a: First voice embedding. Shape: (512, 1, 256) for Kokoro voice embeddings.
           Will be normalized internally along the last dimension.
        b: Second voice embedding (same shape as a).
           Will be normalized internally along the last dimension.
        t: Interpolation parameter in [0, 1].
           t=0 returns voice a, t=1 returns voice b.
        epsilon: Threshold for falling back to linear interpolation when
                 vectors are nearly parallel (sin(θ) ≈ 0).

    Returns:
        Interpolated voice embedding with same shape and dtype as inputs.
        The output magnitude is preserved (interpolates between input magnitudes).

    Raises:
        ValueError: If input arrays have different shapes.

    Example:
        >>> voice_a = voice_manager.get_voice_style("af_bella")
        >>> voice_b = voice_manager.get_voice_style("am_adam")
        >>> blended = slerp_voices(voice_a, voice_b, t=0.5)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(f"Arrays must have same shape: {a.shape} vs {b.shape}")

    # Store original magnitudes to preserve them in output
    mag_a = np.linalg.norm(a, axis=-1, keepdims=True)
    mag_b = np.linalg.norm(b, axis=-1, keepdims=True)

    # Normalize vectors for SLERP calculation
    a_norm = a / (mag_a + 1e-10)  # Add epsilon to avoid division by zero
    b_norm = b / (mag_b + 1e-10)

    # Compute dot product along the last dimension
    # (cosine of angle between normalized vectors)
    dot = np.sum(a_norm * b_norm, axis=-1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)  # Ensure within [-1, 1]

    # Compute the angle theta between vectors
    theta = np.arccos(dot)

    # Compute sine of theta
    sin_theta = np.sin(theta)

    # If sin_theta is near zero, vectors are nearly parallel
    # Fall back to linear interpolation
    if np.max(np.abs(sin_theta)) < epsilon:
        result = (1 - t) * a + t * b
    else:
        # Compute SLERP using the spherical interpolation formula
        sin_t_theta = np.sin(t * theta)
        sin_one_minus_t_theta = np.sin((1 - t) * theta)

        # SLERP on normalized vectors
        result_norm = (sin_one_minus_t_theta / sin_theta) * a_norm + (
            sin_t_theta / sin_theta
        ) * b_norm

        # Interpolate the magnitudes linearly and apply to result
        mag_interp = (1 - t) * mag_a + t * mag_b
        result = result_norm * mag_interp

    # Return with original dtype (float32)
    return result.astype(a.dtype if a.dtype == np.float32 else np.float32)


@dataclass
class VoiceBlend:
    """Configuration for blending multiple voices.

    Args:
        voices: List of (voice_name, weight) tuples
        interpolation: Interpolation method - "linear" (weighted average) or
                      "slerp" (spherical linear interpolation). SLERP requires
                      exactly 2 voices and produces smoother transitions.
    """

    voices: list[tuple[str, float]]
    interpolation: Literal["linear", "slerp"] = "linear"

    def __post_init__(self):
        """Validate voice blend configuration."""
        if not self.voices:
            raise ValueError("VoiceBlend must have at least one voice")

        total_weight = sum(weight for _, weight in self.voices)
        if not 0.99 <= total_weight <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Voice blend weights must sum to 1.0, got {total_weight}")

        # SLERP only supports exactly 2 voices
        if self.interpolation == "slerp" and len(self.voices) != 2:
            raise ValueError(
                f"SLERP interpolation requires exactly 2 voices, "
                f"got {len(self.voices)}. "
                f"Use interpolation='linear' for blending more than 2 voices."
            )

    @classmethod
    def parse(cls, blend_str: str) -> "VoiceBlend":
        """Parse a voice blend string.

        Format: "voice1:weight1,voice2:weight2" or "voice1:50,voice2:50"
        Weights should sum to 100 (percentages).
        Optionally append "@slerp" for spherical interpolation.

        Examples:
            "af_bella:50,am_adam:50"        # Linear interpolation (default)
            "af_bella:50,am_adam:50@slerp"  # Spherical interpolation
            "af_bella:30,af_nicole:70"      # 30/70 linear blend

        Args:
            blend_str: String representation of voice blend

        Returns:
            VoiceBlend instance
        """
        # Check for interpolation method suffix
        interpolation: Literal["linear", "slerp"] = "linear"
        if blend_str.endswith("@slerp"):
            interpolation = "slerp"
            blend_str = blend_str[:-6]  # Remove "@slerp" suffix

        voices = []
        for part in blend_str.split(","):
            part = part.strip()
            if ":" in part:
                voice_name, weight_str = part.split(":", 1)
                weight = float(weight_str) / 100.0  # Convert percentage to fraction
            else:
                voice_name = part
                weight = 1.0
            voices.append((voice_name.strip(), weight))

        # Normalize weights if they don't sum to 1
        total_weight = sum(w for _, w in voices)
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            voices = [(v, w / total_weight) for v, w in voices]

        return cls(voices=voices, interpolation=interpolation)


class VoiceManager:
    """Manages voice loading and blending for PyKokoro.

    Handles:
    - Loading voices from different formats (.npz for HuggingFace, .bin for GitHub)
    - Voice style retrieval by name
    - Voice blending (weighted combination of multiple voices)
    - Voice listing

    Args:
        model_source: Source of the model ('huggingface' or 'github')
    """

    def __init__(self, model_source: ModelSource = "huggingface"):
        """Initialize the voice manager."""
        self._model_source = model_source
        self._voices_data: dict[str, np.ndarray] | None = None
        self._voice_length: int | None = None

    def load_voices(self, voices_path: Path) -> None:
        """Load voices from file.

        Args:
            voices_path: Path to the voices file (.npz or .bin)
        """
        if voices_path.exists() and voices_path.stat().st_size == 0:
            raise ConfigurationError(
                f"Voice archive {voices_path} is empty. Re-download voices to fix it."
            )
        # Check if it's a GitHub .bin file (which is actually .npz format)
        # or a standard .npz file (for HuggingFace)
        if voices_path.suffix == ".bin" or self._model_source == "github":
            voices = self._load_voices_bin_github(voices_path)
        else:
            # HuggingFace format: .npz archive with named voice arrays
            try:
                voices = dict(np.load(str(voices_path), allow_pickle=False))
            except (ValueError, TypeError, EOFError) as exc:
                raise ConfigurationError(
                    "Voice archive contains unsupported data. "
                    "Re-download voices to fix the archive."
                ) from exc

        self._voices_data = self._normalize_loaded_voices(voices, voices_path)
        logger.info(f"Successfully loaded {len(self._voices_data)} voices from {voices_path}")
        logger.debug(f"Available voices: {', '.join(sorted(self._voices_data.keys()))}")

    def _load_voices_bin_github(self, voices_path: Path) -> dict[str, np.ndarray]:
        """Load voices from GitHub format .bin file.

        The GitHub voices.bin format is a NumPy archive file (.npz format)
        containing voice arrays with voice names as keys.

        Args:
            voices_path: Path to the voices.bin file

        Returns:
            Dictionary mapping voice names to numpy arrays
        """
        # Load the NumPy file - it's actually .npz format despite .bin extension
        try:
            voices_npz = np.load(str(voices_path), allow_pickle=False)
        except (ValueError, TypeError, EOFError) as exc:
            raise ConfigurationError(
                "Voice archive contains unsupported data. Re-download voices to fix the archive."
            ) from exc

        # Convert NpzFile to dictionary
        voices: dict[str, np.ndarray] = dict(voices_npz)

        logger.info(f"Successfully loaded {len(voices)} voices from {voices_path}")
        logger.debug(f"Available voices: {', '.join(sorted(voices.keys()))}")

        return voices

    def _normalize_loaded_voices(
        self, voices: dict[str, np.ndarray], voices_path: Path
    ) -> dict[str, np.ndarray]:
        normalized: dict[str, np.ndarray] = {}
        lengths: set[int] = set()

        for voice_name, voice_style in voices.items():
            try:
                voice_array = normalize_voice_style(
                    voice_style,
                    expected_length=None,
                    require_dtype=True,
                    voice_name=voice_name,
                )
                lengths.add(voice_array.shape[0])
                normalized[voice_name] = voice_array
            except ConfigurationError as exc:
                raise ConfigurationError(
                    f"Invalid voice '{voice_name}' in {voices_path}. "
                    "Re-download voices to fix the archive."
                ) from exc

        if not lengths:
            self._voice_length = None
            return normalized

        length_counts = Counter(lengths)
        most_common = length_counts.most_common()
        max_count = most_common[0][1]
        candidate_lengths = [length for length, count in most_common if count == max_count]
        target_length = min(candidate_lengths)

        if len(lengths) > 1:
            logger.debug(
                "Loaded voices with mixed lengths: %s. "
                "Normalizing voices to length %d for blending.",
                ", ".join(str(length) for length in sorted(lengths)),
                target_length,
            )

        for voice_name, voice_array in list(normalized.items()):
            current_length = voice_array.shape[0]
            if current_length == target_length:
                continue
            if current_length > target_length:
                normalized[voice_name] = voice_array[:target_length]
                logger.debug(
                    "Truncated voice '%s' from %d to %d.",
                    voice_name,
                    current_length,
                    target_length,
                )
                continue
            pad_amount = target_length - current_length
            padded = np.pad(
                voice_array,
                ((0, pad_amount), (0, 0), (0, 0)),
                mode="constant",
            )
            normalized[voice_name] = padded
            logger.debug(
                "Padded voice '%s' from %d to %d.",
                voice_name,
                current_length,
                target_length,
            )

        self._voice_length = target_length
        return normalized

    def get_voices(self) -> list[str]:
        """Get list of available voice names.

        Returns:
            Sorted list of voice names

        Raises:
            RuntimeError: If voices have not been loaded yet
        """
        if self._voices_data is None:
            raise RuntimeError("Voices not loaded. Call load_voices() first.")
        return sorted(self._voices_data.keys())

    def get_voice_style(self, voice_name: str) -> np.ndarray:
        """Get the style vector for a voice.

        Args:
            voice_name: Name of the voice

        Returns:
            Numpy array representing the voice style

        Raises:
            RuntimeError: If voices have not been loaded yet
            KeyError: If voice_name is not found
        """
        if self._voices_data is None:
            raise RuntimeError("Voices not loaded. Call load_voices() first.")

        if voice_name not in self._voices_data:
            available = ", ".join(sorted(self._voices_data.keys()))
            raise KeyError(f"Voice '{voice_name}' not found. Available voices: {available}")

        return self._voices_data[voice_name]

    def create_blended_voice(self, blend: VoiceBlend) -> np.ndarray:
        """Create a blended voice from multiple voices.

        Supports two interpolation methods:
        - linear: Weighted average of voice embeddings (works with any number of voices)
        - slerp: Spherical linear interpolation (requires exactly 2 voices)

        SLERP produces smoother, more natural voice transitions by interpolating
        along the surface of a hypersphere rather than through its interior.

        Args:
            blend: VoiceBlend object specifying voices, weights,
                and interpolation method

        Returns:
            Numpy array representing the blended voice style

        Raises:
            RuntimeError: If voices have not been loaded yet
            KeyError: If any voice in the blend is not found

        Example:
            >>> # Linear blend (default)
            >>> blend = VoiceBlend([("af_bella", 0.5), ("am_adam", 0.5)])
            >>> voice = vm.create_blended_voice(blend)
            >>>
            >>> # SLERP blend
            >>> blend = VoiceBlend(
            ...     [("af_bella", 0.5), ("am_adam", 0.5)],
            ...     interpolation="slerp"
            ... )
            >>> voice = vm.create_blended_voice(blend)
        """
        if self._voices_data is None:
            raise RuntimeError("Voices not loaded. Call load_voices() first.")

        return self._blend_voices(blend, self.get_voice_style)

    def _blend_voices(
        self,
        blend: VoiceBlend,
        resolver: Callable[[str], np.ndarray],
    ) -> np.ndarray:
        # Optimize: single voice doesn't need blending
        if len(blend.voices) == 1:
            voice_name, _ = blend.voices[0]
            return resolver(voice_name)

        # SLERP interpolation (exactly 2 voices)
        if blend.interpolation == "slerp":
            (voice_a_name, _), (voice_b_name, weight_b) = blend.voices
            style_a = resolver(voice_a_name)
            style_b = resolver(voice_b_name)
            # Use weight_b as t parameter: t=0 -> voice_a, t=1 -> voice_b
            return slerp_voices(style_a, style_b, t=weight_b)

        # Linear interpolation (weighted sum) - works with any number of voices
        blended: np.ndarray | None = None
        for voice_name, weight in blend.voices:
            style = resolver(voice_name)
            weighted = style * weight
            blended = weighted if blended is None else np.add(blended, weighted)

        # This should never be None if blend.voices is not empty
        assert blended is not None, "No voices in blend"
        return blended

    def resolve_voice(
        self,
        voice: str | np.ndarray | VoiceBlend,
        *,
        voice_db_lookup: Callable[[str], np.ndarray | None] | None = None,
        allow_broadcast: bool = False,
    ) -> np.ndarray:
        """Resolve a voice specification to a style vector.

        Convenience method that handles all voice input types:
        - str: looks up voice by name
        - VoiceBlend: creates blended voice
        - np.ndarray: returns as-is (already a style vector)

        Args:
            voice: Voice specification (name, blend, or vector)

        Returns:
            Numpy array representing the voice style

        Raises:
            RuntimeError: If voices have not been loaded yet
            KeyError: If voice name is not found
        """
        if isinstance(voice, VoiceBlend):
            return self._blend_voices(
                voice, lambda name: self._resolve_voice_name(name, voice_db_lookup)
            )
        if isinstance(voice, str):
            if ":" in voice or "," in voice:
                blend = VoiceBlend.parse(voice)
                return self._blend_voices(
                    blend,
                    lambda name: self._resolve_voice_name(name, voice_db_lookup),
                )
            return self._resolve_voice_name(voice, voice_db_lookup)

        # Already a style vector
        return normalize_voice_style(
            voice,
            expected_length=self._voice_length,
            allow_broadcast=allow_broadcast,
        )

    def _resolve_voice_name(
        self,
        voice_name: str,
        voice_db_lookup: Callable[[str], np.ndarray | None] | None,
    ) -> np.ndarray:
        if voice_db_lookup:
            db_voice = voice_db_lookup(voice_name)
            if db_voice is not None:
                return normalize_voice_style(
                    db_voice,
                    expected_length=self._voice_length,
                    voice_name=voice_name,
                )

        if self._voices_data is None:
            raise RuntimeError("Voices not loaded. Call load_voices() first.")

        if voice_name not in self._voices_data:
            available = ", ".join(sorted(self._voices_data.keys()))
            raise KeyError(f"Voice '{voice_name}' not found. Available voices: {available}")

        return self._voices_data[voice_name]

    def is_loaded(self) -> bool:
        """Check if voices have been loaded.

        Returns:
            True if voices are loaded, False otherwise
        """
        return self._voices_data is not None
