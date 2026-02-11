from pykokoro.pipeline import with_spacy_model_size
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.tokenizer import TokenizerConfig


def test_with_spacy_model_size_creates_tokenizer_config_when_missing():
    cfg = with_spacy_model_size(PipelineConfig())

    assert cfg.tokenizer_config is not None
    assert cfg.tokenizer_config.spacy_model == "auto"
    assert cfg.tokenizer_config.spacy_model_size == "md"


def test_with_spacy_model_size_overrides_existing_tokenizer_config():
    cfg = PipelineConfig(
        tokenizer_config=TokenizerConfig(spacy_model="en_core_web_trf")
    )
    updated = with_spacy_model_size(cfg, size="sm")

    assert updated.tokenizer_config is not None
    assert updated.tokenizer_config.spacy_model == "auto"
    assert updated.tokenizer_config.spacy_model_size == "sm"
