from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from examples import german, german2, german3


def test_german_model_configurations() -> None:
    martin = german.make_config()
    crane = german2.make_config()
    thorsten = german3.make_config()

    assert martin.model_variant is None
    assert martin.generation.lang == "de"
    assert martin.generation.speed == 1.125
    assert german.OUTPUT_FILE == "german_martin_v1_2.wav"

    assert crane.model_source == "github"
    assert crane.model_variant == "de-crane"
    assert crane.model_quality == "fp32"
    assert crane.voice == "default"
    assert crane.generation.lang == "de"
    assert crane.generation.speed == 1.0
    assert crane.allow_experimental_frontend is True
    assert german2.OUTPUT_FILE == "german_kerstin_crane.wav"

    assert thorsten.model_source == "github"
    assert thorsten.model_variant == "de-thorsten"
    assert thorsten.model_quality == "fp32"
    assert thorsten.voice == "thorsten"
    assert thorsten.generation.lang == "de"
    assert thorsten.generation.speed == 1.0
    assert thorsten.allow_experimental_frontend is False
    assert german3.OUTPUT_FILE == "german_thorsten.wav"
    from pykokoro.model_profiles import get_model_profile

    assert get_model_profile("de-crane", "github").g2p_backend == "kokorog2p"
    assert get_model_profile("de-thorsten", "github").g2p_backend == "kokorog2p"
    assert thorsten.short_sentence_config is not None
    assert thorsten.short_sentence_config.enabled is True
    no_short = german3.make_config(short_sentence=False)
    assert no_short.short_sentence_config is not None
    assert no_short.short_sentence_config.enabled is False


def test_german_examples_share_normalization_comparison_text() -> None:
    assert german.TEXT == german2.TEXT == german3.TEXT
    for case in ("14.05.2026", "18:20", "1,5 kg", "Prof.", "Min.", "12,80 EUR"):
        assert case in german.TEXT


def test_german_comparison_layout_lists_all_lexicons() -> None:
    assert [label for label, _lexicons in german.LEXICON_SOURCES] == [
        "gold",
        "crane",
        "espeak",
        "olaph",
    ]
    assert german.format_lexicon_layout() == (
        "gold -> 1.0 s silence -> crane -> 1.0 s silence -> espeak -> 1.0 s silence -> olaph"
    )


@pytest.mark.parametrize("module", [german, german2, german3])
def test_german_configs_change_only_lexicon(module) -> None:
    gold = module.make_config(lexicons=("gold",))
    crane = module.make_config(lexicons=("crane",))

    assert gold.tokenizer_config is not None
    assert crane.tokenizer_config is not None
    assert gold.tokenizer_config.lexicons == ("gold",)
    assert crane.tokenizer_config.lexicons == ("crane",)
    assert gold.generation == crane.generation
    assert gold.voice == crane.voice
    assert gold.model_source == crane.model_source
    assert gold.model_variant == crane.model_variant
    assert gold.model_quality == crane.model_quality
    assert gold.allow_experimental_frontend == crane.allow_experimental_frontend
    assert gold.short_sentence_config == crane.short_sentence_config


@pytest.mark.parametrize("module", [german, german2, german3])
def test_german_audio_combination(module) -> None:
    gold = SimpleNamespace(audio=np.ones(2, dtype=np.float32), sample_rate=4)
    crane = SimpleNamespace(audio=np.full(3, 2.0, dtype=np.float32), sample_rate=4)

    combined = module.combine_lexicon_audio(gold, crane)

    assert combined.shape == (9,)
    np.testing.assert_array_equal(combined[:2], [1.0, 1.0])
    np.testing.assert_array_equal(combined[2:6], np.zeros(4))
    np.testing.assert_array_equal(combined[6:], [2.0, 2.0, 2.0])


def test_german_audio_combination_rejects_mismatched_sample_rates() -> None:
    gold = SimpleNamespace(audio=np.ones(2, dtype=np.float32), sample_rate=4)
    crane = SimpleNamespace(audio=np.ones(2, dtype=np.float32), sample_rate=8)

    with pytest.raises(RuntimeError, match="different sample rates"):
        german.combine_lexicon_audio(gold, crane)


def test_crane_registry_voice_alias_resolves_archive_voice(monkeypatch) -> None:
    from pykokoro.onnx_backend import Kokoro

    class FakeVoiceManager:
        def __init__(self) -> None:
            self.resolved: list[str] = []

        def resolve_voice(self, voice, *, voice_db_lookup):
            _ = voice_db_lookup
            self.resolved.append(voice)
            return object()

    kokoro = Kokoro.__new__(Kokoro)
    kokoro._runtime = None
    kokoro._model_variant = "de-crane"
    manager = FakeVoiceManager()
    kokoro._voice_manager = manager
    monkeypatch.setattr(kokoro, "_init_kokoro", lambda: None)

    kokoro.resolve_voice_style("default")

    assert manager.resolved == ["df_kerstin"]
