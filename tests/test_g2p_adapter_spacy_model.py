from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.tokenizer import TokenizerConfig


def test_kokorog2p_adapter_forwards_spacy_model(monkeypatch):
    captured: dict[str, object] = {}

    class FakeG2PModule:
        @staticmethod
        def get_g2p(**kwargs):
            captured.update(kwargs)
            return object()

    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())

    cfg = PipelineConfig(
        tokenizer_config=TokenizerConfig(
            use_spacy=True,
            spacy_model="en_core_web_trf",
        )
    )

    adapter._get_g2p_instance("en-us", cfg)

    assert captured["language"] == "en-us"
    assert captured["use_spacy"] is True
    assert captured["spacy_model"] == "en_core_web_trf"


def test_kokorog2p_adapter_resolves_auto_spacy_model(monkeypatch):
    captured: dict[str, object] = {}

    class FakeG2PModule:
        @staticmethod
        def get_g2p(**kwargs):
            captured.update(kwargs)
            return object()

    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())

    cfg = PipelineConfig(
        tokenizer_config=TokenizerConfig(
            use_spacy=True,
            spacy_model="auto",
            spacy_model_size="sm",
        )
    )

    adapter._get_g2p_instance("fr-fr", cfg)

    assert captured["language"] == "fr-fr"
    assert captured["spacy_model"] == "fr_core_news_sm"
