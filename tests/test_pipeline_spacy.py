from pykokoro.pipeline import with_spacy_model, with_spacy_model_size
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.tokenizer import TokenizerConfig


def test_with_spacy_model_size_creates_tokenizer_config_when_missing():
    cfg = with_spacy_model_size(PipelineConfig())

    assert cfg.tokenizer_config is not None
    assert cfg.tokenizer_config.spacy_model is None
    assert cfg.tokenizer_config.spacy_model_size is None


def test_with_spacy_model_size_overrides_existing_tokenizer_config():
    cfg = PipelineConfig(tokenizer_config=TokenizerConfig(spacy_model="en_core_web_trf"))
    updated = with_spacy_model_size(cfg, size="sm")

    assert updated.tokenizer_config is not None
    assert updated.tokenizer_config.spacy_model is None
    assert updated.tokenizer_config.spacy_model_size == "sm"


def test_with_spacy_model_returns_highest_available_transform():
    cfg = with_spacy_model()(PipelineConfig())

    assert cfg.tokenizer_config is not None
    assert cfg.tokenizer_config.spacy_model is None
    assert cfg.tokenizer_config.spacy_model_size is None


def test_with_spacy_model_accepts_explicit_model_and_size():
    cfg = with_spacy_model("en_core_web_sm", size="lg")(PipelineConfig())

    assert cfg.tokenizer_config is not None
    assert cfg.tokenizer_config.spacy_model == "en_core_web_sm"
    assert cfg.tokenizer_config.spacy_model_size == "lg"
