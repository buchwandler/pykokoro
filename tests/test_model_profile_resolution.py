from __future__ import annotations

from pathlib import Path

import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.model_profiles import (
    GERMAN_MARTIN_V1_2,
    IMPLEMENTED_FRONTENDS,
    get_model_profile,
    profile_for_language,
    profile_for_voice,
)
from pykokoro.synthesis_config import SynthesisConfig, resolve_synthesis_config


def _resolve(config: SynthesisConfig, language: str | None = None) -> SynthesisConfig:
    return resolve_synthesis_config(
        config,
        language=language or config.generation.lang or "en-us",
        voice=config.voice,
    )


def test_model_profiles_expose_runtime_metadata_without_release_artifacts() -> None:
    profile = get_model_profile("v1.2-de-martin", "github")
    assert profile == GERMAN_MARTIN_V1_2
    assert profile.default_voice == "martin"
    assert profile.frontend in IMPLEMENTED_FRONTENDS
    assert not hasattr(profile, "release_tag")
    assert not hasattr(profile, "model_sha256")


def test_language_resolution_selects_german_chinese_and_english_profiles() -> None:
    german = _resolve(SynthesisConfig(generation=GenerationConfig(lang="de-DE")))
    chinese = _resolve(SynthesisConfig(generation=GenerationConfig(lang="zh")))
    english = _resolve(SynthesisConfig(generation=GenerationConfig(lang="en-us")))

    assert (german.model_source, german.model_variant, german.voice) == (
        "github",
        "v1.2-de-martin",
        "martin",
    )
    assert (chinese.model_source, chinese.model_variant, chinese.voice) == (
        "github",
        "v1.1-zh",
        "af_maple",
    )
    assert (english.model_source, english.model_variant, english.voice) == (
        "github",
        "v1.0",
        "af_heart",
    )


def test_voice_profiles_select_model_when_language_is_generic() -> None:
    for voice, expected in (
        ("sveta", "ru-zaakirio-base"),
        ("dima", "ru-zaakirio-dima"),
        ("ngoc_huyen", "vi-ngoc-huyen"),
    ):
        profile = profile_for_voice(voice)
        assert profile is not None
        resolved = _resolve(SynthesisConfig(voice=voice), language=profile.language_codes[0])
        assert resolved.model_variant == expected
        assert resolved.voice == voice


def test_explicit_model_settings_are_retained() -> None:
    model_path = Path("custom-model.onnx")
    resolved = _resolve(
        SynthesisConfig(
            model_source="github",
            model_variant="vi-ngoc-huyen",
            model_quality="fp32",
            model_path=model_path,
            voice="ngoc_huyen",
            allow_experimental_frontend=True,
        ),
        language="vi",
    )
    assert resolved.model_path == model_path
    assert resolved.model_variant == "vi-ngoc-huyen"
    assert resolved.model_quality == "fp32"
    assert resolved.allow_experimental_frontend is True


def test_unavailable_frontend_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="no runtime-ready distribution"):
        _resolve(
            SynthesisConfig(
                model_source="github",
                model_variant="pl-mateusz",
                generation=GenerationConfig(lang="pl"),
            )
        )


def test_custom_voice_archive_may_supply_its_own_voice_name(tmp_path: Path) -> None:
    voice_archive = tmp_path / "voices.npz"
    resolved = _resolve(
        SynthesisConfig(
            model_source="github",
            model_variant="v1.2-de-martin",
            model_quality="fp32",
            voice="custom-german",
            voices_path=voice_archive,
            generation=GenerationConfig(lang="de"),
        )
    )
    assert resolved.voice == "custom-german"
    assert resolved.voices_path == voice_archive


def test_runtime_profile_registry_is_explicitly_selectable() -> None:
    profile = get_model_profile("en-oddadmix-7m-distill", "github")
    assert profile.runtime_available is True
    assert profile.default_voice == "af_msa"
    assert profile_for_language("en") is None
    assert profile_for_voice("af_msa") is None


def test_explicit_runtime_profile_resolution() -> None:
    resolved = _resolve(
        SynthesisConfig(
            model_source="github",
            model_variant="en-oddadmix-7m-distill",
            generation=GenerationConfig(lang="en-us"),
        )
    )
    assert resolved.model_variant == "en-oddadmix-7m-distill"
    assert resolved.voice == "af_msa"
