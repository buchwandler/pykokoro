import pytest

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
    assert captured["spacy_model"] is None
    assert captured["spacy_model_size"] == "sm"


def test_kokorog2p_adapter_forwards_unset_spacy_as_auto(monkeypatch):
    captured: dict[str, object] = {}

    class FakeG2PModule:
        @staticmethod
        def get_g2p(**kwargs):
            captured.update(kwargs)
            return object()

    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())
    adapter._get_g2p_instance("en-us", PipelineConfig())

    assert captured["use_spacy"] is None
    assert captured["spacy_model"] is None
    assert captured["spacy_model_size"] is None


@pytest.mark.parametrize("variant", ["de-crane", "de-thorsten"])
def test_german_profiles_use_native_kokorog2p_backend(monkeypatch, variant):
    captured: dict[str, object] = {}

    class FakeG2PModule:
        @staticmethod
        def get_g2p(**kwargs):
            captured.update(kwargs)
            return object()

    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())
    cfg = PipelineConfig(
        model_source="github",
        model_variant=variant,
        allow_experimental_frontend=True,
    )

    adapter._get_g2p_instance("de", cfg)

    assert captured["backend"] == "kokorog2p"


def test_explicit_espeak_profile_backend_is_preserved(monkeypatch):
    captured: dict[str, object] = {}

    class FakeG2PModule:
        @staticmethod
        def get_g2p(**kwargs):
            captured.update(kwargs)
            return object()

    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())
    cfg = PipelineConfig(
        model_source="github",
        model_variant="he-hebrew-nc",
        allow_experimental_frontend=True,
    )

    adapter._get_g2p_instance("he", cfg)

    assert captured["backend"] == "espeak"
