"""Configuration for audio generation in PyKokoro.

This module provides the GenerationConfig dataclass for configuring
audio generation parameters in the KokoroPipeline.
"""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for audio generation in the KokoroPipeline.

    Groups all generation-time parameters for easier reuse and documentation.
    Instances are immutable (frozen) to prevent accidental modification.

    This config groups generation parameters into a reusable configuration object
    for PipelineConfig. You can create a config once and reuse it across multiple
    runs, with the ability to override individual parameters using kwargs.

    Priority order when using both config and kwargs:
        1. kwargs (highest priority - explicit per-call overrides)
        2. config (medium priority - structured configuration)
        3. defaults (lowest priority - fallback values)

    Attributes:
        speed: Speech speed multiplier. 1.0 = normal speed, 0.5 = half speed,
            2.0 = double speed. Must be > 0.0. Default: 1.0
        lang: Default language code for text-to-phoneme conversion.
            Examples: 'en-us', 'en-gb', 'es', 'fr', 'de', 'it', 'pt', 'ja',
            'ko', 'zh', 'hi'. Can be overridden per-segment with SSMD
            [text](lang) syntax. Default: "en-us"
        is_phonemes: If True, treat input text as IPA phonemes instead of
            regular text, bypassing text-to-phoneme conversion. Default: False
        pause_mode: Pause handling strategy:
            - "tts" (default): TTS generates pauses naturally at sentence
              boundaries. SSMD pauses are preserved. Best for natural speech.
            - "manual": PyKokoro controls pauses with precision. Silence is
              trimmed from segment boundaries and SSMD pauses are preserved.
              Best for precise timing control.
            - "auto": PyKokoro automatically inserts pauses at sentence and
              paragraph boundaries, and adds clause pauses when long sentences
              are split. Silence is trimmed from segment boundaries.
            Default: "tts"
        pause_clause: Duration in seconds for SSMD ...c (comma) breaks and
            automatic clause boundary pauses when pause_mode="manual" or "auto".
            Must be >= 0.0. Default: 0.3
        pause_sentence: Duration in seconds for SSMD ...s (sentence) breaks and
            automatic sentence boundary pauses when pause_mode="manual" or "auto".
            Must be >= 0.0. Default: 0.6
        pause_paragraph: Duration in seconds for SSMD ...p (paragraph) breaks and
            automatic paragraph boundary pauses when pause_mode="manual" or "auto".
            Must be >= 0.0. Default: 1.0
        pause_variance: Standard deviation in seconds for Gaussian variance added
            to automatic pauses. Only applies when pause_mode="manual" or "auto".
            Default 0.05 (±100ms at 95% confidence). Set to 0.0 to disable
            variance. Must be >= 0.0. Default: 0.05
        random_seed: Optional random seed for reproducible pause variance.
            If None, pauses will vary between runs. If set to an integer,
            pause variance will be reproducible. Default: None
        enable_short_sentence: Override short sentence handling for this run.
            - None (default): Use config setting from PipelineConfig
            - True: Force enable short sentence handling (phoneme pretext)
            - False: Force disable short sentence handling
            Default: None

    Example:
        Basic usage with config:

        >>> from pykokoro import KokoroPipeline, PipelineConfig
        >>> config = GenerationConfig(speed=1.2, pause_mode="manual")
        >>> pipe = KokoroPipeline(PipelineConfig(voice="af_sarah", generation=config))
        >>> res = pipe.run("Hello world")

        Reuse config across multiple generations:

        >>> config = GenerationConfig(
        ...     speed=1.2,
        ...     pause_mode="manual",
        ...     pause_clause=0.25,
        ...     pause_sentence=0.5,
        ... )
        >>> res1 = pipe.run("First sentence.")
        >>> res2 = pipe.run("Second sentence.")

        Override specific parameters using kwargs:

        >>> res = pipe.run(
        ...     "Fast speech",
        ...     generation=GenerationConfig(speed=2.0, pause_mode="manual"),
        ... )
    """

    # Speed and language
    speed: float = 1.0
    lang: str = "en-us"

    # Processing modes
    is_phonemes: bool = False

    # Pause control
    pause_mode: Literal["tts", "manual", "auto"] = "tts"
    pause_clause: float = 0.3
    pause_sentence: float = 0.6
    pause_paragraph: float = 1.0
    pause_variance: float = 0.05
    random_seed: int | None = None

    # Short sentence handling override
    enable_short_sentence: bool | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        # Validate speed
        if self.speed <= 0.0:
            raise ValueError(
                f"speed must be > 0.0, got {self.speed}. "
                f"Use 0.5 for half speed, 1.0 for normal, 2.0 for double speed."
            )

        # Validate pause durations
        if self.pause_clause < 0.0:
            raise ValueError(
                f"pause_clause must be >= 0.0 seconds, got {self.pause_clause}"
            )
        if self.pause_sentence < 0.0:
            raise ValueError(
                f"pause_sentence must be >= 0.0 seconds, got {self.pause_sentence}"
            )
        if self.pause_paragraph < 0.0:
            raise ValueError(
                f"pause_paragraph must be >= 0.0 seconds, got {self.pause_paragraph}"
            )
        if self.pause_variance < 0.0:
            raise ValueError(
                f"pause_variance must be >= 0.0 seconds, got {self.pause_variance}"
            )

        # Validate pause_mode
        if self.pause_mode not in ("tts", "manual", "auto"):
            raise ValueError(
                "pause_mode must be 'tts', 'manual', or 'auto', "
                f"got '{self.pause_mode}'"
            )

        # Validate lang is non-empty
        if not self.lang or not isinstance(self.lang, str):
            raise ValueError(f"lang must be a non-empty string, got {self.lang!r}")

    def merge_with_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Merge config with kwargs, with kwargs taking priority.

        This is used internally by KokoroPipeline to merge the config
        object with individual parameter overrides. Only non-None kwargs
        will override config values.

        Args:
            **kwargs: Individual parameter overrides (None values are ignored)

        Returns:
            Dictionary with merged parameters (non-None kwargs override config)

        Example:
            >>> config = GenerationConfig(speed=1.5, lang="en-gb")
            >>> merged = config.merge_with_kwargs(speed=2.0, lang=None)
            >>> merged["speed"]
            2.0
            >>> merged["lang"]  # Not overridden because kwarg was None
            'en-gb'
        """
        # Start with config values
        result = {
            "speed": self.speed,
            "lang": self.lang,
            "is_phonemes": self.is_phonemes,
            "pause_mode": self.pause_mode,
            "pause_clause": self.pause_clause,
            "pause_sentence": self.pause_sentence,
            "pause_paragraph": self.pause_paragraph,
            "pause_variance": self.pause_variance,
            "random_seed": self.random_seed,
            "enable_short_sentence": self.enable_short_sentence,
        }

        # Override with kwargs (only non-None values override)
        for key in result.keys():
            if key in kwargs and kwargs[key] is not None:
                result[key] = kwargs[key]

        return result
