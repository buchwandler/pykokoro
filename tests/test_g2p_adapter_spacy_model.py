from dataclasses import replace

import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.tokenizer import TokenizerConfig, _effective_lexicons


def test_legacy_dictionary_flags_map_to_current_lexicon_selection() -> None:
    assert _effective_lexicons(TokenizerConfig(load_gold=False, load_silver=False)) == ()
    assert _effective_lexicons(TokenizerConfig(load_gold=True, load_silver=False)) is None
    assert _effective_lexicons(TokenizerConfig(lexicons=("gold",), load_gold=False)) == ("gold",)


def test_legacy_silver_only_selection_is_actionable() -> None:
    with pytest.raises(ValueError, match="no KokoroG2P 0.9.4 equivalent"):
        _effective_lexicons(TokenizerConfig(load_gold=False, load_silver=True))


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


def test_hindi_fallback_uses_explicit_espeak_backend(monkeypatch):
    captured: dict[str, object] = {}

    class FakeG2PModule:
        @staticmethod
        def get_g2p(**kwargs):
            captured.update(kwargs)
            return object()

    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())

    adapter._get_g2p_instance(
        "hi", PipelineConfig(tokenizer_config=TokenizerConfig(backend="espeak"))
    )

    assert captured["language"] == "hi"
    assert captured["backend"] == "espeak"


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


def test_kokorog2p_adapter_forwards_named_lexicons(monkeypatch):
    captured: dict[str, object] = {}

    class FakeG2PModule:
        @staticmethod
        def get_g2p(**kwargs):
            captured.update(kwargs)
            return object()

    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())
    cfg = PipelineConfig(tokenizer_config=TokenizerConfig(lexicons=("crane",), fallback="goruut"))
    adapter._get_g2p_instance("de", cfg)

    assert captured["lexicons"] == ("crane",)
    assert captured["use_espeak_fallback"] is False
    assert captured["use_goruut_fallback"] is True


def test_g2p_instance_cache_distinguishes_named_lexicons(monkeypatch):
    created: list[dict[str, object]] = []

    class FakeG2PModule:
        @staticmethod
        def get_g2p(**kwargs):
            created.append(kwargs)
            return object()

    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())
    gold_cfg = PipelineConfig(tokenizer_config=TokenizerConfig(lexicons=("gold",), fallback="none"))
    crane_cfg = PipelineConfig(tokenizer_config=TokenizerConfig(lexicons=("crane",)))
    espeak_cfg = PipelineConfig(
        tokenizer_config=TokenizerConfig(lexicons=("gold",), fallback="espeak")
    )

    installed_only_cfg = PipelineConfig(
        tokenizer_config=replace(gold_cfg.tokenizer_config, lexicon_data_policy="installed-only")
    )
    gold = adapter._get_g2p_instance("de", gold_cfg)
    espeak = adapter._get_g2p_instance("de", espeak_cfg)
    crane = adapter._get_g2p_instance("de", crane_cfg)
    installed_only = adapter._get_g2p_instance("de", installed_only_cfg)
    gold_again = adapter._get_g2p_instance("de", gold_cfg)

    assert gold is not crane
    assert gold is not espeak
    assert gold is gold_again
    assert gold is not installed_only
    assert [entry["lexicons"] for entry in created] == [
        ("gold",),
        ("gold",),
        ("crane",),
        ("gold",),
    ]
    assert created[0]["use_espeak_fallback"] is False
    assert created[0]["use_goruut_fallback"] is False
    assert created[1]["use_espeak_fallback"] is True
    assert created[1]["use_goruut_fallback"] is False


def test_adapter_retries_after_lexphon_provisioning(monkeypatch):
    from lexphon import LexiconNotInstalledError

    calls = 0
    installed: list[tuple[str, object]] = []

    class FakeG2PModule:
        @staticmethod
        def get_g2p(**kwargs):
            nonlocal calls
            calls += 1
            assert "load_gold" not in kwargs
            assert "load_silver" not in kwargs
            if calls == 1:
                raise LexiconNotInstalledError("de-de:gold")
            return object()

    monkeypatch.setattr(
        "pykokoro.lexicon_data.install_missing_lexphon_data",
        lambda language, lexicons: installed.append((language, lexicons)) or ("de-de:gold",),
    )
    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())

    result = adapter._get_g2p_instance(
        "de", PipelineConfig(tokenizer_config=TokenizerConfig(lexicons=("gold",)))
    )

    assert result is adapter._g2p_instances[next(iter(adapter._g2p_instances))]
    assert calls == 2
    assert installed == [("de", ("gold",)), ("de", ("gold",))]


def test_tokenizer_forwards_named_lexicons(monkeypatch):
    captured: dict[str, object] = {}

    def fake_get_g2p(**kwargs):
        captured.update(kwargs)
        return object()

    import pykokoro.tokenizer as tokenizer_module

    monkeypatch.setattr(tokenizer_module, "get_g2p", fake_get_g2p)
    tokenizer = tokenizer_module.Tokenizer(
        vocab={}, config=TokenizerConfig(lexicons=("gold",), fallback="none")
    )

    tokenizer._get_g2p("de")

    assert captured["lexicons"] == ("gold",)

    assert captured["use_espeak_fallback"] is False
    assert captured["use_goruut_fallback"] is False


def test_legacy_tokenizer_retries_after_lexphon_provisioning(monkeypatch):
    from lexphon import LexiconNotInstalledError

    import pykokoro.tokenizer as tokenizer_module

    calls = 0
    installed: list[tuple[str, object]] = []

    def fake_get_g2p(**kwargs):
        nonlocal calls
        calls += 1
        assert "load_gold" not in kwargs
        assert "load_silver" not in kwargs
        if calls == 1:
            raise LexiconNotInstalledError("de-de:gold")
        return object()

    monkeypatch.setattr(tokenizer_module, "get_g2p", fake_get_g2p)
    monkeypatch.setattr(
        "pykokoro.lexicon_data.install_missing_lexphon_data",
        lambda language, lexicons: installed.append((language, lexicons)) or ("de-de:gold",),
    )
    tokenizer = tokenizer_module.Tokenizer(vocab={}, config=TokenizerConfig(lexicons=("gold",)))

    result = tokenizer._get_g2p("de")

    assert result is tokenizer._g2p_cache["de"]
    assert calls == 2
    assert installed == [("de", ("gold",)), ("de", ("gold",))]


def test_routed_language_uses_effective_default_lexicons(monkeypatch):
    captured: dict[str, object] = {}
    provisioned: list[tuple[str, object]] = []

    class FakeG2PModule:
        @staticmethod
        def get_g2p(**kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(
        "pykokoro.lexicon_data.install_missing_lexphon_data",
        lambda language, lexicons: provisioned.append((language, lexicons)) or (),
    )
    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())

    cfg = PipelineConfig(
        generation=GenerationConfig(lang="de"),
        tokenizer_config=TokenizerConfig(lexicons=("crane",)),
    )
    adapter._get_g2p_instance("en-us", cfg)

    assert captured["language"] == "en-us"
    assert captured["lexicons"] is None
    assert provisioned == [("en-us", None)]
