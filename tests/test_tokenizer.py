from __future__ import annotations

import pytest

from pykokoro.constants import MAX_PHONEME_LENGTH
from pykokoro.tokenizer import Tokenizer, TokenizerConfig


def test_tokenizer_encodes_and_decodes_one_kokoro_vocabulary() -> None:
    tokenizer = Tokenizer(vocab_version="v1.0", vocab={"h": 1, "i": 2, " ": 0})
    tokens = tokenizer.tokenize("hi ?")
    assert tokens == [1, 2]
    assert tokenizer.detokenize(tokens) == "hi"
    assert tokenizer.get_vocab_info()["version"] == "v1.0"
    assert tokenizer.validate_phonemes("hi") == (True, [])
    assert tokenizer.validate_phonemes("x") == (False, ["x"])


def test_tokenizer_rejects_unbounded_phoneme_input() -> None:
    tokenizer = Tokenizer(vocab_version="v1.0", vocab={"a": 1})
    with pytest.raises(ValueError, match="too long"):
        tokenizer.tokenize("a" * (MAX_PHONEME_LENGTH + 1))


def test_tokenizer_config_contains_only_kokorog2p_frontend_settings() -> None:
    config = TokenizerConfig(fallback="goruut", lexicons=("custom",), use_spacy=False)
    assert config.fallback == "goruut"
    assert config.lexicons == ("custom",)
    assert config.use_spacy is False
    assert not hasattr(config, "phoneme_dictionary_path")
    assert not hasattr(config, "use_dictionary")
