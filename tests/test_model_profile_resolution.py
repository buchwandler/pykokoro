from __future__ import annotations

import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.model_profiles import GERMAN_MARTIN_V1_2, get_model_profile
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig, resolve_model_defaults


def test_martin_profile_contains_runtime_metadata_only():
    profile = get_model_profile("v1.2-de-martin", "github")
    assert profile == GERMAN_MARTIN_V1_2
    assert profile.frontend == "german-ipa-v1"
    assert profile.default_voice == "martin"
    assert profile.quality_files == {}
    assert not hasattr(profile, "release_tag")
    assert not hasattr(profile, "model_sha256")


@pytest.mark.parametrize("lang", ["de", "de-DE", "de_at", "de-ch"])
def test_german_defaults_resolve_to_martin(lang):
    resolved = resolve_model_defaults(PipelineConfig(generation=GenerationConfig(lang=lang)))
    assert resolved.model_source == "github"
    assert resolved.model_variant == "v1.2-de-martin"
    assert resolved.model_quality == "fp32"
    assert resolved.voice == "martin"
    assert resolved.generation.lang == lang.lower().replace("_", "-")


def test_explicit_voice_validation_is_deferred_to_release_metadata():
    resolved = resolve_model_defaults(
        PipelineConfig(
            voice="df_eva",
            model_variant="v1.2-de-martin",
            generation=GenerationConfig(lang="de"),
        )
    )
    assert resolved.voice == "df_eva"


def test_martin_voice_alone_infers_german_profile():
    resolved = resolve_model_defaults(PipelineConfig(voice="martin"))
    assert resolved.generation.lang == "de"
    assert resolved.model_variant == "v1.2-de-martin"


def test_custom_voice_archive_can_define_its_own_voice_name(tmp_path):
    resolved = resolve_model_defaults(
        PipelineConfig(
            voice="custom-german",
            voices_path=tmp_path / "voices.bin",
            generation=GenerationConfig(lang="de"),
            model_variant="v1.2-de-martin",
        )
    )
    assert resolved.voice == "custom-german"


def test_resolved_profile_is_in_cache_key():
    pipeline = KokoroPipeline(PipelineConfig())
    resolved = resolve_model_defaults(PipelineConfig(generation=GenerationConfig(lang="de")))
    key = pipeline._kokoro_key(resolved)
    assert "v1.2-de-martin" in key
    assert "github" in key


@pytest.mark.parametrize(
    ("variant", "voice"),
    [
        ("ru-zaakirio-base", "sveta"),
        ("ru-zaakirio-dima", "dima"),
    ],
)
def test_registry_huggingface_profiles_defer_quality_validation(variant, voice):
    cfg = PipelineConfig(
        model_source="huggingface",
        model_variant=variant,
        model_quality="fp32",
        generation=GenerationConfig(lang="ru"),
    )

    resolved = resolve_model_defaults(cfg)

    assert resolved.model_source == "huggingface"
    assert resolved.model_variant == variant
    assert resolved.model_quality == "fp32"
    assert resolved.voice == voice


def test_dima_voice_selects_dima_registry_profile():
    resolved = resolve_model_defaults(PipelineConfig(voice="dima"))

    assert resolved.model_source == "huggingface"
    assert resolved.model_variant == "ru-zaakirio-dima"
    assert resolved.generation.lang == "ru"
    assert resolved.model_quality == "fp32"
