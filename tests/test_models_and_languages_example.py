from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples import models_and_languages as showcase
from pykokoro import ModelCapabilities, ModelDiscoveryResult, VoiceCapabilities


def _model(
    model_id: str = "v1.0",
    *,
    languages: tuple[str, ...] = ("de-de",),
    voices: tuple[str, ...] = ("voice-a",),
    qualities: tuple[str, ...] = ("fp32", "q8"),
    status: str = "ready",
    experimental: bool = False,
) -> ModelCapabilities:
    return ModelCapabilities(
        model_id=model_id,
        source="github",
        languages=languages,
        voices=voices,
        default_voice=voices[0],
        qualities=qualities,
        g2p_backend="kokorog2p",
        lexicons=None,
        frontend="kokorog2p",
        status=status,
        experimental=experimental,
        runtime_available=True,
        redistribution_allowed=True,
        provider="github-release",
        sample_rate=24_000,
        max_tokens=510,
        voice_details=(
            VoiceCapabilities(voices[0], "female", languages[0], languages[0], "German"),
        ),
    )


def _inventory(*models: ModelCapabilities) -> ModelDiscoveryResult:
    return ModelDiscoveryResult(tuple(models), "fixture", False)


def test_listing_displays_public_model_capabilities(capsys) -> None:
    showcase.list_models(_inventory(_model()))
    output = capsys.readouterr().out

    for value in ("v1.0", "de-de", "voice-a", "fp32", "github-release", "24000", "510"):
        assert value in output


def test_public_selection_validates_language_voice_and_quality() -> None:
    model = _model()
    assert showcase._choose_language(model, "de") == "de-de"
    assert showcase._choose_voice(model, None) == "voice-a"
    assert showcase._choose_quality(model, None) == "fp32"

    with pytest.raises(ValueError, match="Language"):
        showcase._choose_language(model, "fr")
    with pytest.raises(ValueError, match="Voice"):
        showcase._choose_voice(model, "missing")
    with pytest.raises(ValueError, match="Quality"):
        showcase._choose_quality(model, "q4")


def test_synthesis_uses_discovered_public_metadata_and_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _model()

    class FakeSynthesizer:
        config = None

        def __init__(self, config) -> None:
            type(self).config = config

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def synthesize_text(self, text: str, *, language: str, voice: str):
            assert text == showcase.SAMPLE_TEXTS["de"]
            assert language == "de-de"
            assert voice == "voice-a"
            return SimpleNamespace(audio=np.zeros(4), sample_rate=24_000)

    writes: list[Path] = []
    monkeypatch.setattr(showcase, "KokoroSynthesizer", FakeSynthesizer)
    monkeypatch.setattr(showcase.sf, "write", lambda path, *args: writes.append(path))

    output = showcase.synthesize(
        _inventory(model),
        model_id=model.model_id,
        language="de",
        voice=None,
        quality="q8",
        include_experimental=False,
        output_dir=tmp_path,
    )

    assert output == tmp_path / "v1.0_de-de_voice-a.wav"
    assert writes == [output]
    assert FakeSynthesizer.config.model_source == model.source
    assert FakeSynthesizer.config.model_variant == model.model_id
    assert FakeSynthesizer.config.model_quality == "q8"
    assert FakeSynthesizer.config.voice == model.default_voice
    assert FakeSynthesizer.config.generation.lang == "de-de"


def test_experimental_model_requires_opt_in(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="experimental frontend"):
        showcase.synthesize(
            _inventory(_model(status="experimental", experimental=True)),
            model_id="v1.0",
            language=None,
            voice=None,
            quality=None,
            include_experimental=False,
            output_dir=tmp_path,
        )
