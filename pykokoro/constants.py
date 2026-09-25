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
    "pt-pt": "pt-pt",
    "ko": "ko",
    "ja": "ja",
    "zh": "zh",  # Mandarin Chinese
    "cmn": "cmn",  # Accept both zh and cmn
    "ar": "ar",
    "he": "he",
    "kk": "kk",
    "sv": "sv",
    "th": "th",
    "vi": "vi",
}

# These languages require an explicit ``backend="espeak"`` or ``"goruut"``
# choice. Keeping them out of SUPPORTED_LANGUAGES prevents the native adapter
# from advertising a factory route that kokorog2p does not provide.
ESPEAK_ONLY_LANGUAGES = {
    "pl": "pl",
    "tr": "tr",
    "ru": "ru",
    "hi": "hi",
}
