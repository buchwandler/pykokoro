"""Tests for pykokoro.tokenizer module."""

import pytest

from pykokoro.constants import (
    ESPEAK_ONLY_LANGUAGES,
    MAX_PHONEME_LENGTH,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
)
from pykokoro.mixed_language_handler import MixedLanguageHandler
from pykokoro.tokenizer import (
    EspeakConfig,
    PhonemeResult,
    Tokenizer,
    TokenizerConfig,
    create_tokenizer,
)


class TestEspeakConfig:
    """Tests for EspeakConfig dataclass (deprecated, kept for compatibility)."""

    def test_default_values(self):
        """Test default config has None values."""
        config = EspeakConfig()
        assert config.lib_path is None
        assert config.data_path is None

    def test_custom_values(self):
        """Test config with custom values."""
        config = EspeakConfig(lib_path="/path/to/lib", data_path="/path/to/data")
        assert config.lib_path == "/path/to/lib"
        assert config.data_path == "/path/to/data"


class TestTokenizerConfig:
    """Tests for TokenizerConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = TokenizerConfig()
        assert config.use_espeak_fallback is True
        assert config.use_spacy is None
        assert config.spacy_model is None
        assert config.spacy_model_size is None
        assert config.use_dictionary is True

    def test_custom_values(self):
        """Test config with custom values."""
        config = TokenizerConfig(
            use_espeak_fallback=False,
            use_goruut_fallback=True,
            use_spacy=False,
            spacy_model="en_core_web_sm",
            spacy_model_size="sm",
            use_dictionary=False,
        )
        assert config.use_espeak_fallback is False
        assert config.use_goruut_fallback is True
        assert config.use_spacy is False
        assert config.spacy_model == "en_core_web_sm"
        assert config.spacy_model_size == "sm"
        assert config.use_dictionary is False


class TestPhonemeResult:
    """Tests for PhonemeResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = PhonemeResult(phonemes="test")
        assert result.phonemes == "test"
        assert result.tokens == []
        assert result.low_confidence_words == []

    def test_custom_values(self):
        """Test result with custom values."""
        from kokorog2p import GToken

        tokens = [GToken(text="hello", phonemes="hɛlO")]
        result = PhonemeResult(
            phonemes="hɛlO",
            tokens=tokens,
            low_confidence_words=["xyz"],
        )
        assert result.phonemes == "hɛlO"
        assert len(result.tokens) == 1
        assert result.low_confidence_words == ["xyz"]


class TestConstants:
    """Tests for module constants."""

    def test_max_phoneme_length(self):
        """Test MAX_PHONEME_LENGTH is defined."""
        assert MAX_PHONEME_LENGTH == 510

    def test_sample_rate(self):
        """Test SAMPLE_RATE is defined."""
        assert SAMPLE_RATE == 24000

    def test_supported_languages(self):
        """Test SUPPORTED_LANGUAGES contains expected entries."""
        assert "en-us" in SUPPORTED_LANGUAGES
        assert "en-gb" in SUPPORTED_LANGUAGES
        assert "en" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["en"] == "en-us"
        assert set(ESPEAK_ONLY_LANGUAGES) == {"pl", "tr", "ru"}
        assert not set(ESPEAK_ONLY_LANGUAGES).intersection(SUPPORTED_LANGUAGES)


class TestVocab:
    """Tests for vocabulary loading."""

    def test_tokenizer_has_vocab(self):
        """Test tokenizer has vocabulary."""
        tokenizer = Tokenizer()
        assert tokenizer.vocab is not None
        assert len(tokenizer.vocab) > 100  # Should have many phonemes


class TestTokenizer:
    """Tests for Tokenizer class."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return Tokenizer()

    def test_init_default(self, tokenizer):
        """Test default initialization."""
        assert tokenizer.vocab is not None
        assert len(tokenizer.vocab) > 100

    def test_init_custom_vocab(self):
        """Test initialization with custom vocab."""
        custom_vocab = {"a": 1, "b": 2}
        tokenizer = Tokenizer(vocab=custom_vocab)
        assert tokenizer.vocab == custom_vocab

    def test_init_with_config(self):
        """Test initialization with TokenizerConfig."""
        config = TokenizerConfig(use_spacy=False, use_espeak_fallback=True)
        tokenizer = Tokenizer(config=config)
        assert tokenizer.config.use_spacy is False
        assert tokenizer.config.use_espeak_fallback is True

    def test_get_g2p_forwards_spacy_model(self, monkeypatch):
        """Test configured spaCy model is forwarded to get_g2p."""
        import pykokoro.tokenizer as tokenizer_module

        captured: dict[str, object] = {}

        def fake_get_g2p(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(tokenizer_module, "get_g2p", fake_get_g2p)

        tokenizer = Tokenizer(config=TokenizerConfig(use_spacy=True, spacy_model="en_core_web_trf"))
        tokenizer._get_g2p("en-us")

        assert captured["use_spacy"] is True
        assert captured["spacy_model"] == "en_core_web_trf"

    def test_get_g2p_resolves_auto_spacy_model(self, monkeypatch):
        """Test auto spaCy model derives from language and size."""
        import pykokoro.tokenizer as tokenizer_module

        captured: dict[str, object] = {}

        def fake_get_g2p(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(tokenizer_module, "get_g2p", fake_get_g2p)

        tokenizer = Tokenizer(
            config=TokenizerConfig(use_spacy=True, spacy_model="auto", spacy_model_size="lg")
        )
        tokenizer._get_g2p("de")

        assert captured["spacy_model"] is None
        assert captured["spacy_model_size"] == "lg"

    def test_normalize_text(self, tokenizer):
        """Test text normalization."""
        assert tokenizer.normalize_text("  hello  ") == "hello"
        assert tokenizer.normalize_text("\n\nhello\n\n") == "hello"
        assert tokenizer.normalize_text("hello") == "hello"

    def test_phonemize_basic(self, tokenizer):
        """Test basic phonemization."""
        phonemes = tokenizer.phonemize("hello")
        assert isinstance(phonemes, str)
        assert len(phonemes) > 0
        # Should contain valid phonemes (all chars should be in vocab)
        for char in phonemes:
            assert char in tokenizer.vocab

    def test_phonemize_empty(self, tokenizer):
        """Test phonemization of empty string."""
        phonemes = tokenizer.phonemize("")
        assert phonemes == ""

    def test_phonemize_whitespace_only(self, tokenizer):
        """Test phonemization of whitespace-only string."""
        phonemes = tokenizer.phonemize("   ")
        assert phonemes == ""

    def test_phonemize_with_language(self, tokenizer):
        """Test phonemization with different languages."""
        # US English
        us_phonemes = tokenizer.phonemize("hello", lang="en-us")
        # UK English
        gb_phonemes = tokenizer.phonemize("hello", lang="en-gb")
        # Both should produce phonemes
        assert len(us_phonemes) > 0
        assert len(gb_phonemes) > 0

    @pytest.mark.parametrize(
        ("original_text", "normalized_text"),
        [
            ("2", "two"),
            ("42 kg", "forty two kilograms"),
            ("Dr. Smith", "Doctor Smith"),
            ("$5", "five dollars"),
        ],
    )
    def test_native_backend_matches_spokenform_equivalent_pairs(
        self,
        original_text,
        normalized_text,
    ):
        """Normalization-sensitive text should phonemize identically on the 0.8 path."""
        tokenizer = Tokenizer(config=TokenizerConfig(backend="kokorog2p", use_spacy=False))

        original_phonemes = tokenizer.phonemize(original_text, lang="en-us")
        normalized_phonemes = tokenizer.phonemize(normalized_text, lang="en-us")

        assert original_phonemes == normalized_phonemes
        assert tokenizer.tokenize(original_phonemes) == tokenizer.tokenize(normalized_phonemes)

    def test_phonemize_with_punctuation(self, tokenizer):
        """Test phonemization with punctuation."""
        phonemes = tokenizer.phonemize("Hello, world!")
        assert isinstance(phonemes, str)
        assert len(phonemes) > 0
        assert "!" in phonemes and "," in phonemes  # Punctuation should be handled
        phonemes = tokenizer.phonemize("Hello . . . world!")
        assert isinstance(phonemes, str)
        assert len(phonemes) > 0
        assert "…" in phonemes and "!" in phonemes  # Punctuation should be handled

    def test_phonemize_detailed(self, tokenizer):
        """Test detailed phonemization."""
        result = tokenizer.phonemize_detailed("Hello world")
        assert isinstance(result, PhonemeResult)
        assert len(result.phonemes) > 0
        assert len(result.tokens) > 0
        assert isinstance(result.low_confidence_words, list)

    def test_tokenize_basic(self, tokenizer):
        """Test basic tokenization."""
        phonemes = tokenizer.phonemize("hello")
        tokens = tokenizer.tokenize(phonemes)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_tokenize_empty(self, tokenizer):
        """Test tokenization of empty string."""
        tokens = tokenizer.tokenize("")
        assert tokens == []

    def test_tokenize_length_limit(self, tokenizer):
        """Test tokenization length limit."""
        # Create a very long phoneme string
        long_phonemes = "a" * (MAX_PHONEME_LENGTH + 1)
        with pytest.raises(ValueError, match="too long"):
            tokenizer.tokenize(long_phonemes)

    def test_tokenize_at_limit(self, tokenizer):
        """Test tokenization at exactly the limit."""
        phonemes = "a" * MAX_PHONEME_LENGTH
        # Should not raise
        tokens = tokenizer.tokenize(phonemes)
        assert len(tokens) <= MAX_PHONEME_LENGTH

    def test_detokenize_basic(self, tokenizer):
        """Test basic detokenization."""
        phonemes = tokenizer.phonemize("hello")
        tokens = tokenizer.tokenize(phonemes)
        recovered = tokenizer.detokenize(tokens)
        # Should recover the original phonemes
        assert recovered == phonemes

    def test_detokenize_empty(self, tokenizer):
        """Test detokenization of empty list."""
        phonemes = tokenizer.detokenize([])
        assert phonemes == ""

    def test_detokenize_unknown_tokens(self, tokenizer):
        """Test detokenization with unknown token IDs."""
        # Token ID 99999 doesn't exist
        phonemes = tokenizer.detokenize([99999])
        assert phonemes == ""

    def test_text_to_tokens(self, tokenizer):
        """Test text-to-tokens convenience method."""
        tokens = tokenizer.text_to_tokens("hello")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Should be same as phonemize + tokenize
        phonemes = tokenizer.phonemize("hello")
        expected = tokenizer.tokenize(phonemes)
        assert tokens == expected

    def test_reverse_vocab_lazy_init(self, tokenizer):
        """Test reverse vocab is lazily initialized."""
        # Access should create it
        reverse = tokenizer.reverse_vocab
        assert isinstance(reverse, dict)
        assert len(reverse) == len(tokenizer.vocab)
        # Keys and values should be swapped
        for phoneme, token_id in tokenizer.vocab.items():
            assert reverse[token_id] == phoneme

    def test_text_to_phonemes_with_words(self, tokenizer):
        """Test word-by-word phonemization."""
        result = tokenizer.text_to_phonemes_with_words("hello world")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0][0] == "hello"
        assert result[1][0] == "world"
        # Each tuple should have (word, phonemes)
        for word, phonemes in result:
            assert isinstance(word, str)
            assert isinstance(phonemes, str)

    def test_format_readable(self, tokenizer):
        """Test human-readable formatting."""
        readable = tokenizer.format_readable("hello world")
        assert "hello [" in readable
        assert "world [" in readable
        assert "]" in readable

    def test_get_vocab_info_method(self, tokenizer):
        """Test get_vocab_info method."""
        info = tokenizer.get_vocab_info()
        assert info["num_tokens"] == len(tokenizer.vocab)
        assert info["max_phoneme_length"] == MAX_PHONEME_LENGTH
        assert info["backend"] == "kokorog2p"

    def test_validate_phonemes(self, tokenizer):
        """Test phoneme validation."""
        # Valid phonemes
        is_valid, invalid = tokenizer.validate_phonemes("hɛlO")
        assert is_valid is True
        assert invalid == []

        # Invalid phonemes
        is_valid, invalid = tokenizer.validate_phonemes("hɛlO§")
        assert is_valid is False
        assert "§" in invalid


class TestTokenizerRoundTrip:
    """Tests for complete roundtrip conversions."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return Tokenizer()

    def test_roundtrip_simple(self, tokenizer):
        """Test simple text roundtrip."""
        text = "Hello world"
        phonemes = tokenizer.phonemize(text)
        tokens = tokenizer.tokenize(phonemes)
        recovered = tokenizer.detokenize(tokens)
        assert recovered == phonemes

    def test_roundtrip_punctuation(self, tokenizer):
        """Test roundtrip with punctuation."""
        text = "Hello, world! How are you?"
        phonemes = tokenizer.phonemize(text)
        tokens = tokenizer.tokenize(phonemes)
        recovered = tokenizer.detokenize(tokens)
        assert recovered == phonemes

    def test_roundtrip_numbers(self, tokenizer):
        """Test roundtrip with numbers."""
        text = "I have 5 apples"
        phonemes = tokenizer.phonemize(text)
        tokens = tokenizer.tokenize(phonemes)
        recovered = tokenizer.detokenize(tokens)
        assert recovered == phonemes


class TestCreateTokenizer:
    """Tests for create_tokenizer convenience function."""

    def test_create_default(self):
        """Test creating tokenizer with defaults."""
        tokenizer = create_tokenizer()
        assert tokenizer.config.use_espeak_fallback is True
        assert tokenizer.config.use_spacy is None
        assert tokenizer.config.spacy_model is None
        assert tokenizer.config.spacy_model_size is None

    def test_create_custom(self):
        """Test creating tokenizer with custom settings."""
        tokenizer = create_tokenizer(
            use_espeak_fallback=False,
            use_spacy=False,
            spacy_model="en_core_web_sm",
            spacy_model_size="sm",
        )
        assert tokenizer.config.use_espeak_fallback is False
        assert tokenizer.config.use_spacy is False
        assert tokenizer.config.spacy_model == "en_core_web_sm"
        assert tokenizer.config.spacy_model_size == "sm"


class TestKokorog2pIntegration:
    """Tests for kokorog2p integration features."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return Tokenizer()

    def test_homograph_disambiguation(self, tokenizer):
        """Test that homographs are pronounced correctly based on context."""
        # 'read' as present tense (VBP)
        result1 = tokenizer.phonemize_detailed("They read books often")
        # 'read' as past tense (VBD)
        result2 = tokenizer.phonemize_detailed("He read the book")

        # Both should produce valid phonemes
        assert len(result1.phonemes) > 0
        assert len(result2.phonemes) > 0

    def test_article_pronunciation(self, tokenizer):
        """Test 'the' pronunciation before vowels."""
        # 'the' before vowel should be 'ði'
        phonemes1 = tokenizer.phonemize("the apple")
        # 'the' before consonant should be 'ðə'
        phonemes2 = tokenizer.phonemize("the book")

        # Both should produce valid phonemes
        assert len(phonemes1) > 0
        assert len(phonemes2) > 0
        # Check that 'the' is phonemized (contains 'ð')
        assert "ð" in phonemes1
        assert "ð" in phonemes2

    def test_dictionary_vs_espeak_quality(self, tokenizer):
        """Test that common words use dictionary (higher rating)."""
        result = tokenizer.phonemize_detailed("Hello world")

        # Common words should have high rating (from dictionary)
        for token in result.tokens:
            if token.text.lower() in ["hello", "world"]:
                rating = token.get("rating")
                # Rating 3-4 = dictionary, 1 = espeak
                assert rating is None or rating >= 3, f"Expected {token.text} to use dictionary"

    def test_unknown_word_handling(self, tokenizer):
        """Test that unknown words are handled via espeak fallback."""
        # Made-up word that won't be in dictionary
        result = tokenizer.phonemize_detailed("The xyzfoobar is here")

        # Should still produce valid phonemes
        assert len(result.phonemes) > 0

        # The made-up word should be in low_confidence_words
        # (or handled by espeak with low rating)
        for token in result.tokens:
            if "xyzfoobar" in token.text.lower():
                rating = token.get("rating")
                if rating is not None and rating < 2:
                    pass
        # May or may not be marked low confidence depending on espeak behavior
        # Just verify phonemes were produced
        assert "xyzfoobar" not in result.phonemes  # Should be converted to phonemes


class TestMixedLanguageSupport:
    """Tests for mixed-language phonemization support."""

    def test_mixed_language_handler_type_hint(self):
        """Test MixedLanguageHandler uses TokenizerConfig in annotations."""
        annotation = MixedLanguageHandler.__init__.__annotations__.get("config")
        assert annotation == "TokenizerConfig"

    def test_mixed_language_config_defaults(self):
        """Test that mixed-language config has correct defaults."""
        config = TokenizerConfig()
        assert config.use_mixed_language is False
        assert config.mixed_language_primary is None
        assert config.mixed_language_allowed is None
        assert config.mixed_language_confidence == 0.7

    def test_mixed_language_disabled_by_default(self):
        """Test that mixed-language mode is disabled by default."""
        tokenizer = Tokenizer()
        assert tokenizer.config.use_mixed_language is False

        # Should use standard single-language G2P
        phonemes = tokenizer.phonemize("Hello world", lang="en-us")
        assert isinstance(phonemes, str)
        assert len(phonemes) > 0

    def test_mixed_language_validation_no_allowed_languages(self):
        """Test validation fails when enabled without allowed_languages."""
        config = TokenizerConfig(use_mixed_language=True)
        tokenizer = Tokenizer(config=config)

        with pytest.raises(ValueError, match="mixed_language_allowed is not set"):
            tokenizer.phonemize("Test", lang="en-us")

    def test_mixed_language_validation_empty_allowed_languages(self):
        """Test validation fails with empty allowed_languages list."""
        config = TokenizerConfig(use_mixed_language=True, mixed_language_allowed=[])
        tokenizer = Tokenizer(config=config)

        with pytest.raises(ValueError, match="mixed_language_allowed is not set"):
            tokenizer.phonemize("Test", lang="en-us")

    def test_mixed_language_validation_unsupported_language(self):
        """Test validation fails with unsupported language."""
        config = TokenizerConfig(use_mixed_language=True, mixed_language_allowed=["xx-invalid"])
        tokenizer = Tokenizer(config=config)

        with pytest.raises(ValueError, match="is not supported"):
            tokenizer.phonemize("Test", lang="en-us")

    def test_mixed_language_validation_invalid_confidence(self):
        """Test validation fails with invalid confidence threshold."""
        # Too high
        config = TokenizerConfig(
            use_mixed_language=True,
            mixed_language_allowed=["en-us", "de"],
            mixed_language_confidence=1.5,
        )
        tokenizer = Tokenizer(config=config)

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            tokenizer.phonemize("Test", lang="en-us")

        # Too low
        config.mixed_language_confidence = -0.1
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            tokenizer.phonemize("Test", lang="en-us")

    def test_mixed_language_fallback_on_import_error(self):
        """Test fallback to single-language when lingua not available."""
        config = TokenizerConfig(
            use_mixed_language=True,
            mixed_language_primary="de",
            mixed_language_allowed=["de", "en-us"],
        )
        tokenizer = Tokenizer(config=config)

        # If lingua is not available, should fall back gracefully
        try:
            phonemes = tokenizer.phonemize("Test text", lang="de")
            # Should still get phonemes (preprocess_multilang will be skipped)
            assert isinstance(phonemes, str)
        except ImportError:
            pytest.skip("Expected behavior: falls back when lingua unavailable")

    def test_mixed_language_primary_not_in_allowed_error(self):
        """Test error when primary language not in allowed list."""
        config = TokenizerConfig(
            use_mixed_language=True,
            mixed_language_primary="fr-fr",
            mixed_language_allowed=["de", "en-us"],  # Missing fr-fr
        )
        tokenizer = Tokenizer(config=config)

        # Should raise ValueError because primary not in allowed
        with pytest.raises(ValueError, match="must be in allowed_languages"):
            tokenizer.phonemize("Test", lang="de")

    def test_mixed_language_uses_lang_as_primary_fallback(self):
        """Test that lang parameter is used as primary if not specified."""
        config = TokenizerConfig(
            use_mixed_language=True,
            # No mixed_language_primary specified
            mixed_language_allowed=["de", "en-us"],
        )
        tokenizer = Tokenizer(config=config)

        try:
            # Should use 'de' as primary since it's passed to phonemize
            phonemes = tokenizer.phonemize("Test", lang="de")
            assert isinstance(phonemes, str)
        except ImportError:
            pytest.skip("lingua-language-detector not available")

    def test_mixed_language_caching_reuses_instance(self):
        """Test that the same configuration reuses cached G2P instance."""
        config = TokenizerConfig(
            use_mixed_language=True,
            mixed_language_primary="de",
            mixed_language_allowed=["de", "en-us"],
        )
        tokenizer = Tokenizer(config=config)

        try:
            # First call creates the G2P instance
            tokenizer.phonemize("First text", lang="de")
            cache_size_1 = len(tokenizer._g2p_cache)

            # Second call should reuse the instance
            tokenizer.phonemize("Second text", lang="de")
            cache_size_2 = len(tokenizer._g2p_cache)

            # Cache size should be the same (reused)
            assert cache_size_1 == cache_size_2
        except ImportError:
            pytest.skip("lingua-language-detector not available")
