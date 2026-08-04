"""Tokenizer for pykokoro - converts text to phonemes and tokens.

This module provides text-to-phoneme and phoneme-to-token conversion using
kokorog2p (dictionary + espeak fallback) as the phonemizer backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import kokorog2p as _kokorog2p
from kokorog2p import phonemize
from kokorog2p.base import G2PBase

from .constants import MAX_PHONEME_LENGTH
from .mixed_language_handler import MixedLanguageHandler
from .phoneme_dictionary import PhonemeDictionary
from .spacy_models import SpacyModelSize, make_spacy_model_request

N_TOKENS = _kokorog2p.N_TOKENS
BackendType: TypeAlias = str
GToken: TypeAlias = Any
filter_for_kokoro = _kokorog2p.filter_for_kokoro
get_g2p = _kokorog2p.get_g2p
get_kokoro_vocab = _kokorog2p.get_kokoro_vocab
ids_to_phonemes = _kokorog2p.ids_to_phonemes
phonemes_to_ids = _kokorog2p.phonemes_to_ids
validate_for_kokoro = _kokorog2p.validate_for_kokoro


logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer.

    Attributes:
        use_espeak_fallback: Whether to use espeak for OOV words (default: True).
            Only applies when backend='espeak'.
        use_goruut_fallback: Whether to use goruut for OOV words (default: False).
            Requires pygoruut to be installed. Only applies when backend='goruut'
        use_spacy: Whether to use spaCy for POS tagging. ``False`` disables it,
            ``None`` selects the best compatible local model when available and
            otherwise falls back, and ``True`` requires a compatible local model.
            Only applies to English.
        spacy_model: Explicit spaCy model package, or None for automatic selection.
            ``"auto"`` remains accepted as a compatibility alias for None.
        spacy_model_size: Exact spaCy package tier, or None to select the highest
            installed compatible model. One of: "sm", "md", "lg", "trf".
        use_dictionary: DEPRECATED. Use load_gold and load_silver instead.
        use_mixed_language: Enable automatic language detection for mixed-language
            text (default: False). Requires mixed_language_allowed to be set.
        mixed_language_primary: Primary language code for mixed-language mode
            (e.g., 'de', 'en-us'). If None, uses the language passed to phonemize().
        mixed_language_allowed: List of language codes to detect and support in
            mixed-language mode (e.g., ['de', 'en-us', 'fr']). Required when
            use_mixed_language is True.
        mixed_language_confidence: Minimum confidence threshold (0.0-1.0) for
            accepting language detection results. Words below this threshold
            fall back to primary_language (default: 0.7).
        phoneme_dictionary_path: Path to custom phoneme dictionary JSON file.
            Format: {"word": "/phoneme/"} where phonemes are in IPA format.
        phoneme_dict_case_sensitive: Whether phoneme dictionary matching should
            be case-sensitive (default: False).
        backend: Phonemization backend: "kokorog2p" (default), "espeak") or "goruut".
            Requires pygoruut for goruut backend. Raises ImportError if unavailable.
        load_gold: Load gold-tier dictionary (~170k common words). Only applies
            to languages with dictionaries (English, French, German). Default: True.
        load_silver: Load silver-tier dictionary (~100k extra entries). Only applies
            to English. Saves ~22-31 MB memory if False. Default: True.
    """

    use_espeak_fallback: bool = True
    use_goruut_fallback: bool = False
    use_spacy: bool | None = None
    spacy_model: str | None = None
    spacy_model_size: SpacyModelSize | None = None
    use_dictionary: bool = True
    use_mixed_language: bool = False
    mixed_language_primary: str | None = None
    mixed_language_allowed: list[str] | None = None
    mixed_language_confidence: float = 0.7
    phoneme_dictionary_path: str | None = None
    phoneme_dict_case_sensitive: bool = False

    # Backend configuration
    backend: BackendType = "kokorog2p"
    load_gold: bool = True
    load_silver: bool = True

    def __post_init__(self) -> None:
        request = make_spacy_model_request(
            model=self.spacy_model,
            size=self.spacy_model_size,
        )
        self.spacy_model = request.model
        self.spacy_model_size = request.size


# Backward compatibility alias
@dataclass
class EspeakConfig:
    """Configuration for espeak-ng backend (deprecated, use TokenizerConfig).

    Kept for backward compatibility. The lib_path and data_path are now
    managed by kokorog2p internally.

    Attributes:
        lib_path: Path to the espeak-ng shared library (ignored)
        data_path: Path to the espeak-ng data directory (ignored)
    """

    lib_path: str | None = None
    data_path: str | None = None


@dataclass
class PhonemeResult:
    """Result of phonemization with quality metadata.

    Attributes:
        phonemes: The phoneme string
        tokens: List of GToken objects with per-word phonemes
        low_confidence_words: Words that used espeak fallback
    """

    phonemes: str
    tokens: list[GToken] = field(default_factory=list)
    low_confidence_words: list[str] = field(default_factory=list)


class Tokenizer:
    """Text-to-phoneme-to-token converter using kokorog2p.

    This class handles:
    1. Text normalization
    2. Text to phoneme conversion (via kokorog2p dictionary + espeak fallback)
    3. Phoneme to token conversion (via Kokoro vocabulary)
    4. Token to phoneme conversion (reverse lookup)
    5. Optional mixed-language support for automatic language detection

    Mixed-Language Support:
        Enable automatic language detection for text containing multiple languages
        by setting TokenizerConfig.use_mixed_language=True and specifying
        allowed_languages. This uses kokorog2p's preprocess_multilang to annotate
        text with language tags before phonemization.

    Args:
        espeak_config: Deprecated, kept for backward compatibility
        vocab_version: Ignored (uses kokorog2p's embedded vocabulary)
        vocab: Optional custom vocabulary dict (overrides default)
        config: Optional TokenizerConfig for phonemization settings

    Example:
        >>> # Single-language usage (default)
        >>> tokenizer = Tokenizer()
        >>> phonemes = tokenizer.phonemize("Hello world")
        >>> tokens = tokenizer.tokenize(phonemes)

        >>> # Mixed-language usage
        >>> config = TokenizerConfig(
        ...     use_mixed_language=True,
        ...     mixed_language_primary="de",
        ...     mixed_language_allowed=["de", "en-us"]
        ... )
        >>> tokenizer = Tokenizer(config=config)
        >>> phonemes = tokenizer.phonemize("Ich gehe zum Meeting")
    """

    def __init__(
        self,
        espeak_config: EspeakConfig | None = None,
        vocab_version: str = "v1.0",
        vocab: dict[str, int] | None = None,
        config: TokenizerConfig | None = None,
    ):
        """Initialize the tokenizer.

        Args:
            espeak_config: Deprecated, kept for backward compatibility
            vocab_version: Model variant/version (e.g., 'v1.0', 'v1.1-zh') for filtering
            vocab: Optional custom vocabulary (overrides default)
            config: Optional TokenizerConfig for phonemization settings
        """
        self.vocab_version = vocab_version
        # Determine kokorog2p model parameter from vocab_version
        # Map variant names to kokorog2p model names
        if vocab_version == "v1.1-zh":
            self._kokorog2p_model = "1.1"
        else:
            self._kokorog2p_model = "1.0"

        self.vocab = vocab if vocab is not None else get_kokoro_vocab(model=self._kokorog2p_model)
        self._reverse_vocab: dict[int, str] | None = None
        self.config = config or TokenizerConfig()

        # Check for deprecated use_dictionary
        if not self.config.use_dictionary:
            import warnings

            warnings.warn(
                "TokenizerConfig.use_dictionary is deprecated. "
                "Use load_gold=False and load_silver=False instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Apply deprecation behavior: disable dictionary loading
            self.config.load_gold = False
            self.config.load_silver = False

        # G2P instances cache (lazy loaded per language)
        self._g2p_cache: dict[str, G2PBase] = {}

        # Mixed-language handler for automatic language detection
        self._mixed_language_handler = MixedLanguageHandler(
            config=self.config, kokorog2p_model=self._kokorog2p_model
        )

        # Phoneme dictionary for custom word->phoneme mappings
        self._phoneme_dictionary_obj: PhonemeDictionary | None = None
        if self.config.phoneme_dictionary_path:
            try:
                self._phoneme_dictionary_obj = PhonemeDictionary(
                    dictionary_path=self.config.phoneme_dictionary_path,
                    case_sensitive=self.config.phoneme_dict_case_sensitive,
                )
            except (OSError, ValueError) as e:
                logger.warning(
                    f"Failed to load phoneme dictionary from "
                    f"'{self.config.phoneme_dictionary_path}': {e}. "
                    f"Continuing without custom phoneme dictionary."
                )

        # Log if espeak_config was provided (deprecated)
        if espeak_config is not None and (espeak_config.lib_path or espeak_config.data_path):
            logger.warning("EspeakConfig is deprecated. kokorog2p manages espeak internally.")

    def _validate_mixed_language_config(self) -> None:
        """Delegate to MixedLanguageHandler.validate_config (backward compatibility)."""
        self._mixed_language_handler.validate_config()

    def _get_g2p(self, lang: str) -> G2PBase:
        """Get or create a G2P instance for the given language.

        If mixed-language mode is enabled, preprocessing is applied in the
        phonemize method before calling this G2P instance.

        Args:
            lang: Language code (e.g., 'en-us', 'en-gb', 'de', 'fr-fr')

        Returns:
            G2P instance for the language

        Raises:
            ValueError: If mixed-language config is invalid
        """
        # Validate mixed-language configuration if enabled
        if self.config.use_mixed_language:
            self._validate_mixed_language_config()

        if lang not in self._g2p_cache:
            # Map language to kokorog2p format
            from .constants import SUPPORTED_LANGUAGES

            kokorog2p_lang = SUPPORTED_LANGUAGES.get(lang, lang)

            # All languages are now fully supported by kokorog2p
            # kokorog2p uses dictionary + espeak fallback for all languages
            self._g2p_cache[lang] = get_g2p(
                language=kokorog2p_lang,
                use_goruut_fallback=self.config.use_goruut_fallback,
                use_espeak_fallback=self.config.use_espeak_fallback,
                use_spacy=self.config.use_spacy,
                spacy_model=self.config.spacy_model,
                spacy_model_size=self.config.spacy_model_size,
                backend=self.config.backend,
                load_gold=self.config.load_gold,
                load_silver=self.config.load_silver,
                version=self._kokorog2p_model,
                phoneme_quotes="curly",
            )

        return self._g2p_cache[lang]

    def _load_phoneme_dictionary(self, path: str | Path) -> dict[str, str]:
        """Delegate to PhonemeDictionary.load (backward compatibility)."""
        phoneme_dict = PhonemeDictionary()
        return phoneme_dict.load(path)

    def _apply_phoneme_dictionary(self, text: str) -> str:
        """Delegate to PhonemeDictionary.apply (backward compatibility)."""
        if self._phoneme_dictionary_obj:
            return self._phoneme_dictionary_obj.apply(text)
        return text

    @property
    def reverse_vocab(self) -> dict[int, str]:
        """Get the reverse vocabulary (token ID -> phoneme).

        Lazily constructed on first access.
        """
        if self._reverse_vocab is None:
            self._reverse_vocab = {v: k for k, v in self.vocab.items()}
        return self._reverse_vocab

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text before phonemization.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        return text.strip()

    def phonemize(
        self,
        text: str,
        lang: str = "en-us",
        normalize: bool = True,
    ) -> str:
        """Convert text to phonemes.

        If a custom phoneme dictionary is configured, words in the dictionary
        will be replaced with their custom pronunciations before phonemization.

        Args:
            text: Input text
            lang: Language code (e.g., 'en-us', 'en-gb')
            normalize: Whether to normalize text first

        Returns:
            Phoneme string (Kokoro format)

        Raises:
            ValueError: If language is not supported
        """
        if normalize:
            text = self.normalize_text(text)

        if not text:
            return ""

        # Preprocess for mixed-language detection (before custom dictionary)
        if self.config.use_mixed_language:
            text = self._mixed_language_handler.preprocess_text(text, default_language=lang)
        # Apply custom phoneme dictionary first
        processed_text = self._apply_phoneme_dictionary(text)
        g2p = self._get_g2p(lang)
        result = phonemize(processed_text, language=lang, g2p=g2p)
        return result.phonemes

    def phonemize_detailed(
        self,
        text: str,
        lang: str = "en-us",
        normalize: bool = True,
    ) -> PhonemeResult:
        """Convert text to phonemes with detailed token information.

        Args:
            text: Input text
            lang: Language code (e.g., 'en-us', 'en-gb')
            normalize: Whether to normalize text first

        Returns:
            PhonemeResult with phonemes, tokens, and quality metadata
        """
        if normalize:
            text = self.normalize_text(text)

        if not text:
            return PhonemeResult(phonemes="", tokens=[], low_confidence_words=[])

        # Get G2P instance for language
        g2p = self._get_g2p(lang)

        # Get tokens with per-word phonemes
        tokens = g2p(text)

        # Build phoneme string and identify low-confidence words
        phoneme_parts = []
        low_confidence = []

        for token in tokens:
            if token.phonemes:
                phoneme_parts.append(token.phonemes)
                # Check rating (1 = espeak fallback, 3-4 = dictionary)
                rating = token.get("rating", 4)
                if rating is not None and rating < 2:
                    low_confidence.append(token.text)
            if token.whitespace:
                phoneme_parts.append(" ")

        phonemes = "".join(phoneme_parts)
        phonemes = filter_for_kokoro(phonemes, model=self._kokorog2p_model)

        return PhonemeResult(
            phonemes=phonemes.strip(),
            tokens=tokens,
            low_confidence_words=low_confidence,
        )

    def tokenize(self, phonemes: str) -> list[int]:
        """Convert phonemes to token IDs.

        Args:
            phonemes: Phoneme string (Kokoro format)

        Returns:
            List of token IDs

        Raises:
            ValueError: If phoneme string exceeds MAX_PHONEME_LENGTH
        """
        if len(phonemes) > MAX_PHONEME_LENGTH:
            raise ValueError(
                f"Phoneme string too long ({len(phonemes)} chars). "
                f"Maximum is {MAX_PHONEME_LENGTH} phonemes."
            )

        return phonemes_to_ids(phonemes, model=self._kokorog2p_model)

    def detokenize(self, tokens: list[int]) -> str:
        """Convert token IDs back to phonemes.

        Args:
            tokens: List of token IDs

        Returns:
            Phoneme string
        """
        return ids_to_phonemes(tokens, model=self._kokorog2p_model)

    def text_to_tokens(
        self,
        text: str,
        lang: str = "en-us",
        normalize: bool = True,
    ) -> list[int]:
        """Convert text directly to tokens.

        Convenience method combining phonemize() and tokenize().

        Args:
            text: Input text
            lang: Language code
            normalize: Whether to normalize text first

        Returns:
            List of token IDs
        """
        phonemes = self.phonemize(text, lang=lang, normalize=normalize)
        return self.tokenize(phonemes)

    def text_to_phonemes_with_words(
        self,
        text: str,
        lang: str = "en-us",
    ) -> list[tuple[str, str]]:
        """Convert text to phonemes, preserving word boundaries.

        Useful for creating readable phoneme exports.

        Args:
            text: Input text
            lang: Language code

        Returns:
            List of (word, phonemes) tuples
        """
        g2p = self._get_g2p(lang)
        tokens = g2p(text)

        result = []
        for token in tokens:
            if token.phonemes and token.text.strip():
                # Filter phonemes for Kokoro vocabulary
                filtered_phonemes = filter_for_kokoro(token.phonemes, model=self._kokorog2p_model)
                result.append((token.text, filtered_phonemes))

        return result

    def format_readable(
        self,
        text: str,
        lang: str = "en-us",
    ) -> str:
        """Format text with phonemes in a human-readable way.

        Args:
            text: Input text
            lang: Language code

        Returns:
            Formatted string like "Hello [həˈloʊ] world [wɜːld]"
        """
        word_phonemes = self.text_to_phonemes_with_words(text, lang=lang)
        return " ".join(f"{word} [{phonemes}]" for word, phonemes in word_phonemes)

    def get_vocab_info(self) -> dict:
        """Get information about the current vocabulary.

        Returns:
            Dictionary with vocabulary metadata
        """
        return {
            "version": self.vocab_version,
            "num_tokens": len(self.vocab),
            "max_token_id": max(self.vocab.values()) if self.vocab else 0,
            "max_phoneme_length": MAX_PHONEME_LENGTH,
            "n_tokens": N_TOKENS,
            "backend": "kokorog2p",
        }

    def validate_phonemes(self, phonemes: str) -> tuple[bool, list[str]]:
        """Validate that all characters are in the Kokoro vocabulary.

        Args:
            phonemes: Phoneme string to validate

        Returns:
            Tuple of (is_valid, list_of_invalid_chars)
        """
        return validate_for_kokoro(phonemes)


# Convenience function for simple usage
def create_tokenizer(
    use_espeak_fallback: bool = True,
    use_goruut_fallback: bool = False,
    use_spacy: bool | None = None,
    spacy_model: str | None = None,
    spacy_model_size: SpacyModelSize | None = None,
) -> Tokenizer:
    """Create a tokenizer with the specified configuration.

    Args:
        use_espeak_fallback: Whether to use espeak for OOV words
        use_goruut_fallback: Whether to use goruut for OOV words
        use_spacy: False to disable spaCy, None for local-only automatic selection,
            or True to require a compatible local model.
        spacy_model: Explicit spaCy model package, or None for automatic selection
        model selection.
        spacy_model_size: Exact spaCy package tier, or None for highest available.

    Returns:
        Configured Tokenizer instance
    """
    config = TokenizerConfig(
        use_espeak_fallback=use_espeak_fallback,
        use_goruut_fallback=use_goruut_fallback,
        use_spacy=use_spacy,
        spacy_model=spacy_model,
        spacy_model_size=spacy_model_size,
    )
    return Tokenizer(config=config)
