"""Constants for pykokoro - default configuration and program metadata."""

# Program metadata
PROGRAM_NAME = "pykokoro"

# Default configuration
# Structure: {"model_quality": "fp32", "use_gpu": False, "vocab_version": "v1.0"}
# Keys: model_quality (quantization), use_gpu (bool), vocab_version (str)
DEFAULT_CONFIG = {
    # Options: fp32, fp16, q8, q8f16, q4, q4f16, uint8, uint8f16
    "model_quality": "fp32",
    # Whether to use GPU acceleration
    "use_gpu": False,
    # Vocabulary version
    "vocab_version": "v1.0",
}

# Model constants
MAX_PHONEME_LENGTH = 510
SAMPLE_RATE = 24000

# Native languages for the default ``kokorog2p`` backend.
# Format: language code -> kokorog2p language code. Languages that are only
# available through an explicitly selected fallback backend are kept separate.
SUPPORTED_LANGUAGES = {
    "en-us": "en-us",
    "en-gb": "en-gb",
    "en": "en-us",  # Default English to US
    "es": "es",
    "fr-fr": "fr-fr",
    "fr": "fr-fr",  # Accept both fr and fr-fr
    "de": "de",
    "it": "it",
    "pt": "pt",
    "ko": "ko",
    "ja": "ja",
    "zh": "zh",  # Mandarin Chinese
    "cmn": "cmn",  # Accept both zh and cmn
}

# These languages require an explicit ``backend="espeak"`` or ``"goruut"``
# choice. Keeping them out of SUPPORTED_LANGUAGES prevents the native adapter
# from advertising a factory route that kokorog2p does not provide.
ESPEAK_ONLY_LANGUAGES = {
    "pl": "pl",
    "tr": "tr",
    "ru": "ru",
}

# SSML-based default mappings for absolute values
# Reference: https://www.w3.org/TR/speech-synthesis11/#S3.2.4
VOLUME_ABSOLUTE_MAP = {
    "silent": -float("inf"),  # Complete silence
    "x-soft": -12.0,  # dB
    "soft": -6.0,  # dB
    "medium": 0.0,  # dB (no change)
    "loud": 6.0,  # dB
    "x-loud": 12.0,  # dB
    "default": 0.0,  # dB (no change)
}

RATE_ABSOLUTE_MAP = {
    "x-slow": 0.5,  # 50% speed
    "slow": 0.75,  # 75% speed
    "medium": 1.0,  # Normal speed
    "fast": 1.25,  # 125% speed
    "x-fast": 1.5,  # 150% speed
    "default": 1.0,  # Normal speed
}

# Core-only SSMD emphasis approximation. These relative values are deterministic
# and avoid making emphasis behavior depend on optional rate/pitch backends.
EMPHASIS_GAIN_DB = {
    "strong": 6.0,
    "moderate": 3.0,
    "reduced": -3.0,
}

PITCH_ABSOLUTE_MAP = {
    "x-low": -4.0,  # semitones
    "low": -2.0,  # semitones
    "medium": 0.0,  # semitones (no change)
    "high": 2.0,  # semitones
    "x-high": 4.0,  # semitones
    "default": 0.0,  # semitones (no change)
}
