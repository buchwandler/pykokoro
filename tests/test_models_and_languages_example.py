from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples import models_and_languages as showcase


class FakeDistribution:
    def __init__(self, *qualities: str, provider: str = "github-release") -> None:
        self.provider = provider
        self.qualities = qualities


class FakeModel:
    def __init__(
        self,
        model_id: str,
        *,
        languages: tuple[str, ...] = ("en",),
        voices: tuple[str, ...] = ("default",),
        default_voice: str = "default",
        distributions: tuple[FakeDistribution, ...] = (FakeDistribution("fp32"),),
        frontend: str = "pykokoro-native-v1",
        layout: str = "single-onnx-v1",
        runtime_available: bool = True,
        redistribution_allowed: bool = True,
    ) -> None:
        self.model_id = model_id
        self.language_codes = languages
        self.voices = voices
        self.default_voice = default_voice
        self.distributions = distributions
        self.frontend = frontend
        self.layout = layout
        self.runtime_available = runtime_available
        self.redistribution_allowed = redistribution_allowed

    def distribution(self) -> FakeDistribution:
        return self.distributions[0]


class FakeRegistry:
    def __init__(self, *models: FakeModel) -> None:
        self.models = {model.model_id: model for model in models}

    def model(self, model_id: str) -> FakeModel:
        return self.models[model_id]


def _profile(model_id: str, *, experimental: bool = False):
    return SimpleNamespace(
        source="github",
        variant=model_id,
        frontend_experimental=experimental,
    )


def test_import_does_not_load_registry(monkeypatch) -> None:
    monkeypatch.setattr(showcase, "load_registry", lambda: pytest.fail("registry loaded at import"))
    assert Path("model_language_outputs") == showcase.OUTPUT_DIR


def test_listing_uses_all_injected_models_and_languages(capsys, monkeypatch) -> None:
    registry = FakeRegistry(
        FakeModel("first", languages=("en", "fr"), voices=("a", "b"), default_voice="a"),
        FakeModel("second", languages=("de-de",), frontend="german-ipa-v1"),
        FakeModel("unavailable", runtime_available=False),
        FakeModel("restricted", redistribution_allowed=False),
    )
    monkeypatch.setattr(
        showcase,
        "get_registry_model_profile",
        lambda model_id, *, registry: _profile(model_id, experimental=model_id == "second"),
    )

    showcase.list_models(registry)
    output = capsys.readouterr().out

    for value in ("first", "second", "unavailable", "restricted", "en", "fr", "de-de"):
        assert value in output
    assert "Provider: github-release" in output
    assert "Default voice: a" in output
    assert "Voices (2): a, b" in output
    assert "Qualities: fp32" in output
    assert "first [ready]" in output
    assert "second [experimental]" in output
    assert "unavailable [registry-unavailable]" in output
    assert "restricted [restricted]" in output


def test_unavailable_and_restricted_models_are_not_synthesized(monkeypatch, tmp_path) -> None:
    for status, model_id in (("registry-unavailable", "unavailable"), ("restricted", "restricted")):
        registry = FakeRegistry(FakeModel(model_id))
        monkeypatch.setattr(
            showcase,
            "_status_for_model",
            lambda model, registry, status=status: status,
        )
        with pytest.raises(ValueError, match=status):
            showcase.synthesize(
                registry,
                model_id=model_id,
                language=None,
                voice=None,
                quality=None,
                include_experimental=False,
                output_dir=tmp_path,
            )


def test_experimental_model_requires_opt_in(monkeypatch, tmp_path) -> None:
    registry = FakeRegistry(FakeModel("experimental"))
    monkeypatch.setattr(showcase, "_status_for_model", lambda model, registry: "experimental")

    with pytest.raises(ValueError, match="include-experimental"):
        showcase.synthesize(
            registry,
            model_id="experimental",
            language=None,
            voice=None,
            quality=None,
            include_experimental=False,
            output_dir=tmp_path,
        )


def test_selection_validates_language_voice_and_quality(monkeypatch, tmp_path) -> None:
    model = FakeModel(
        "selected",
        languages=("de", "de-at"),
        voices=("one", "two"),
        default_voice="one",
        distributions=(FakeDistribution("fp32", "q8"),),
    )
    registry = FakeRegistry(model)
    monkeypatch.setattr(showcase, "_status_for_model", lambda model, registry: "ready")

    for kwargs, message in (
        ({"language": "fr"}, "Language"),
        ({"voice": "missing"}, "Voice"),
        ({"quality": "q4"}, "Quality"),
    ):
        with pytest.raises(ValueError, match=message):
            showcase.synthesize(
                registry,
                model_id="selected",
                language=kwargs.get("language"),
                voice=kwargs.get("voice"),
                quality=kwargs.get("quality"),
                include_experimental=False,
                output_dir=tmp_path,
            )


def test_synthesis_uses_registry_profile_and_selected_output(monkeypatch, tmp_path) -> None:
    model = FakeModel(
        "selected",
        languages=("de",),
        voices=("one", "two"),
        default_voice="one",
        distributions=(FakeDistribution("fp32", "q8"),),
    )
    registry = FakeRegistry(model)
    fake_profile = _profile("selected")
    monkeypatch.setattr(showcase, "_status_for_model", lambda model, registry: "ready")
    monkeypatch.setattr(
        showcase,
        "get_registry_model_profile",
        lambda model_id, *, registry: fake_profile,
    )

    class FakePipeline:
        config = None

        def __init__(self, config) -> None:
            type(self).config = config

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def run(self, text: str):
            assert text == showcase.SAMPLE_TEXTS["de"]
            return SimpleNamespace(audio=np.zeros(4), sample_rate=24000)

    writes: list[Path] = []
    monkeypatch.setattr(showcase, "KokoroPipeline", FakePipeline)
    monkeypatch.setattr(showcase.sf, "write", lambda path, audio, sample_rate: writes.append(path))

    output = showcase.synthesize(
        registry,
        model_id="selected",
        language="de",
        voice="two",
        quality="q8",
        include_experimental=False,
        output_dir=tmp_path,
    )

    assert output == tmp_path / "selected_de_two.wav"
    assert writes == [output]
    assert FakePipeline.config.model_source == "github"
    assert FakePipeline.config.model_variant == "selected"
    assert FakePipeline.config.model_quality == "q8"
    assert FakePipeline.config.voice == "two"
    assert FakePipeline.config.generation.lang == "de"
