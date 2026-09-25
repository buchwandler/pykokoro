from __future__ import annotations

from pykokoro.spacy_models import make_spacy_model_request
from pykokoro.tokenizer import TokenizerConfig


def test_default_request_is_highest_available_and_auto_is_unset() -> None:
    request = make_spacy_model_request(model=None, size=None)
    assert request.model is None
    assert request.size is None


def test_explicit_model_wins_over_size_in_request() -> None:
    request = make_spacy_model_request(model="de_core_news_lg", size="sm")
    assert request.model == "de_core_news_lg"
    assert request.size == "sm"


def test_tokenizer_config_normalizes_spacy_model_selection() -> None:
    config = TokenizerConfig(spacy_model_size="lg")
    assert config.spacy_model is None
    assert config.spacy_model_size == "lg"
