from __future__ import annotations

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


def test_german_examples_share_normalization_comparison_text() -> None:
    assert german.TEXT == german2.TEXT == german3.TEXT
    for case in ("14.05.2026", "18:20", "1,5 kg", "Prof.", "Min.", "12,80 EUR"):
        assert case in german.TEXT


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
