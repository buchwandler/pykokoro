from __future__ import annotations

from pykokoro import PipelineConfig, with_spacy_model
from pykokoro.pipeline import KokoroPipeline
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


def test_with_spacy_model_updates_pipeline_configuration() -> None:
    config = with_spacy_model(size="lg")(PipelineConfig())
    assert isinstance(config.tokenizer_config, TokenizerConfig)
    assert config.tokenizer_config.spacy_model_size == "lg"
    assert isinstance(KokoroPipeline(config), KokoroPipeline)
