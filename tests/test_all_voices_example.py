from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import examples.all_voices as all_voices
from pykokoro.discovery import ModelCapabilities, ModelDiscoveryResult, VoiceCapabilities
from pykokoro.short_sentence_handler import ShortSentenceConfig


def _model(
    model_id: str,
    voices: tuple[str, ...],
    *,
    status: str = "ready",
    source: str = "github",
    qualities: tuple[str, ...] = ("fp32",),
    language: str = "en",
    locale: str = "en-US",
    label: str = "American English",
    gender: str = "female",
    experimental: bool = False,
) -> ModelCapabilities:
    details = tuple(VoiceCapabilities(voice, gender, language, locale, label) for voice in voices)
    return ModelCapabilities(
        model_id=model_id,
        source=source,
        languages=(language,),
        voices=voices,
        default_voice=voices[0],
        qualities=qualities,
        g2p_backend=None,
        lexicons=None,
        frontend="test-frontend",
        status=status,
        experimental=experimental,
        runtime_available=True,
        redistribution_allowed=True,
        sample_rate=24000,
        voice_details=details,
    )


def _discovery(*models: ModelCapabilities) -> ModelDiscoveryResult:
    return ModelDiscoveryResult(tuple(models), "fixture", False)


def test_catalog_ordering_preserves_model_priority_and_voice_roster_order() -> None:
    catalog = all_voices.build_catalog(
        _discovery(
            _model("ordinary", ("voice_c", "voice_a")),
            _model("v1.1-zh", ("zf_002",)),
            _model("v1.0", ("af_alloy", "af_aoede")),
            _model("restricted", ("blocked",), status="restricted"),
        )
    )

    assert [entry.voice for entry in catalog.entries] == [
        "af_alloy",
        "af_aoede",
        "zf_002",
        "voice_c",
        "voice_a",
    ]
    assert [entry.number for entry in catalog.entries] == list(range(1, 6))
    assert catalog.skipped == (all_voices.SkippedModel("restricted", "restricted", 1),)


def test_russian_github_models_are_scheduled_with_all_voices() -> None:
    catalog = all_voices.build_catalog(
        _discovery(
            _model(
                "ru-zaakirio-base",
                ("sveta", "masha"),
                language="ru",
                locale="ru",
                label="Russian",
            ),
            _model(
                "ru-zaakirio-dima",
                ("dima",),
                language="ru",
                locale="ru",
                label="Russian",
            ),
        )
    )

    assert [(entry.model_id, entry.model_source, entry.voice) for entry in catalog.entries] == [
        ("ru-zaakirio-base", "github", "sveta"),
        ("ru-zaakirio-base", "github", "masha"),
        ("ru-zaakirio-dima", "github", "dima"),
    ]
    assert {entry.quality for entry in catalog.entries} == {"fp32"}


def test_missing_metadata_fails_closed() -> None:
    model = replace(_model("v1.0", ("af_alloy",)), voice_details=())

    with pytest.raises(all_voices.ShowcaseError, match="missing voice metadata"):
        all_voices.build_catalog(_discovery(model))


def test_quality_selection_prefers_fp32_and_rejects_empty() -> None:
    assert (
        all_voices.choose_quality(_model("quality", ("voice",), qualities=("q8", "fp16"))) == "q8"
    )
    assert (
        all_voices.choose_quality(_model("quality", ("voice",), qualities=("fp32", "q8"))) == "fp32"
    )
    with pytest.raises(all_voices.ShowcaseError, match="no registry-declared"):
        all_voices.choose_quality(_model("quality", ("voice",), qualities=()))


def test_announcements_use_each_supported_locale() -> None:
    for index, locale in enumerate(sorted(all_voices.ANNOUNCEMENT_BUILDERS), 1):
        entry = all_voices.VoiceShowcaseEntry(
            index,
            "model",
            "github",
            "fp32",
            "af_alloy",
            locale.split("-")[0],
            locale,
            "Language",
            "female",
            False,
            "ready",
            24000,
        )
        sentence = all_voices.announcement_for(entry)
        assert str(index) in sentence
        assert "A F alloy" in sentence
        if locale not in {"en-US", "en-GB"}:
            assert "This is" not in sentence


def test_spoken_voice_name_spells_code_prefix_and_numeric_suffix() -> None:
    assert all_voices.spoken_voice_name("af_alloy") == "A F alloy"
    assert all_voices.spoken_voice_name("zf_001") == "Z F 0 0 1"


def test_synthesize_streams_audio_and_separators(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entries = (
        replace_entry(1, "voice_a"),
        replace(
            replace_entry(2, "af_maple"),
            model_id="v1.1-zh",
            experimental=True,
        ),
        replace(
            replace_entry(3, "zf_001"),
            model_id="v1.1-zh",
            language="zh",
            locale="zh",
            experimental=True,
        ),
    )
    catalog = all_voices.ShowcaseCatalog(entries, (), "fixture")
    calls: list[dict[str, object]] = []

    pipeline_configs: list[object] = []

    class FakePipeline:
        def __init__(self, config: object) -> None:
            self.config = config
            pipeline_configs.append(config)

        def __enter__(self) -> FakePipeline:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def run(self, text: str, **overrides: object) -> object:
            calls.append({"text": text, **overrides})
            return SimpleNamespace(audio=np.array([0.1, 0.2], dtype=np.float32), sample_rate=24000)

    writes: list[np.ndarray] = []

    class FakeSoundFile:
        def __init__(self, path: Path, **kwargs: object) -> None:
            Path(path).touch()

        def __enter__(self) -> FakeSoundFile:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def write(self, audio: np.ndarray) -> None:
            writes.append(np.asarray(audio))

    import pykokoro

    monkeypatch.setattr(pykokoro, "KokoroPipeline", FakePipeline)
    monkeypatch.setattr(all_voices.sf, "SoundFile", FakeSoundFile)
    output = tmp_path / "all_voices.wav"

    duration = all_voices.synthesize_catalog(catalog, output=output, pause=0.25)

    assert output.is_file()
    assert len(calls) == 3
    assert [call["voice"] for call in calls] == ["voice_a", "af_maple", "zf_001"]
    assert [call["model_source"] for call in calls] == ["github"] * 3
    assert [call["model_variant"] for call in calls] == ["v1.0", "v1.1-zh", "v1.1-zh"]
    assert [call["model_quality"] for call in calls] == ["fp32"] * 3
    assert [call["lang"] for call in calls] == ["en-US", "en-US", "zh"]
    assert [call["allow_experimental_frontend"] for call in calls] == [False, True, True]
    assert [write.size for write in writes] == [2, 6000, 2, 6000, 2]
    assert duration == pytest.approx((2 * 3 + 6000 * 2) / 24000)
    assert len(pipeline_configs) == 1
    short_sentence_config = pipeline_configs[0].short_sentence_config
    assert isinstance(short_sentence_config, ShortSentenceConfig)
    assert short_sentence_config.resolve_mode == "wrap"


def test_list_only_does_not_create_pipeline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        all_voices, "discover_models", lambda **_: _discovery(_model("v1.0", ("voice",)))
    )

    def fail_pipeline(*args: object, **kwargs: object) -> None:
        raise AssertionError("pipeline must not be created")

    import pykokoro

    monkeypatch.setattr(pykokoro, "KokoroPipeline", fail_pipeline)
    assert all_voices.main(["--list-only"]) == 0
    assert "Voices scheduled: 1" in capsys.readouterr().out


def replace_entry(number: int, voice: str) -> all_voices.VoiceShowcaseEntry:
    return all_voices.VoiceShowcaseEntry(
        number,
        "v1.0",
        "github",
        "fp32",
        voice,
        "en",
        "en-US",
        "American English",
        "female",
        False,
        "ready",
        24000,
    )
