from __future__ import annotations

import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.model_profiles import (
    GERMAN_MARTIN_V1_2,
    IMPLEMENTED_FRONTENDS,
    get_model_profile,
    get_registry_model_profile,
    profile_for_language,
    profile_for_voice,
)
from pykokoro.model_registry import ModelRegistry
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


def test_vi_anphunl_explicit_profile_is_runtime_available() -> None:
    profile = get_model_profile("vi-anphunl", "github")

    assert profile.default_voice == "diem_trinh"
    assert profile.runtime_available is True
    assert profile.support_status == "ready"


def test_vi_ngoc_huyen_explicit_profile_resolves_for_all_voices() -> None:
    profile = get_model_profile("vi-ngoc-huyen", "github")

    assert profile.language_codes == ("vi",)
    assert profile.default_voice == "ngoc_huyen"
    assert profile.frontend == "vig2p-v1"
    assert profile.frontend_experimental is True
    assert profile.g2p_backend == "espeak"
    assert profile.onnx_inputs == {
        "tokens": "int64",
        "style": "float32",
        "speed": "float32",
    }

    resolved = resolve_model_defaults(
        PipelineConfig(
            model_source="github",
            model_variant="vi-ngoc-huyen",
            model_quality="fp32",
            voice="ngoc_huyen",
            allow_experimental_frontend=True,
            generation=GenerationConfig(lang="vi"),
        )
    )

    assert resolved.model_source == "github"
    assert resolved.model_variant == "vi-ngoc-huyen"
    assert resolved.model_quality == "fp32"
    assert resolved.voice == "ngoc_huyen"
    assert resolved.allow_experimental_frontend is True


def test_software_mansion_anna_is_ready() -> None:
    profile = get_model_profile("de-anna", "github")

    assert profile.language_codes == ("de",)
    assert profile.default_voice == "df_anna"
    assert profile.frontend == "german-ipa-v1"
    assert profile.frontend_experimental is False
    assert profile.g2p_backend == "kokorog2p"
    assert profile.runtime_available is True
    assert profile.support_status == "ready"
    assert profile.onnx_inputs == {
        "tokens": "int64",
        "style": "float32",
        "speed": "float32",
    }
    assert profile.frontend in IMPLEMENTED_FRONTENDS


def test_software_mansion_mateusz_is_known_but_staged() -> None:
    profile = get_model_profile("pl-mateusz", "github")

    assert profile.language_codes == ("pl",)
    assert profile.default_voice == "pm_mateusz"
    assert profile.frontend == "phonemis-pl-v1"
    assert profile.runtime_available is False
    assert profile.support_status == "registry-unavailable"
    assert profile.frontend not in IMPLEMENTED_FRONTENDS


def test_automatic_language_selection_keeps_martin_and_excludes_mateusz() -> None:
    assert profile_for_language("de").variant == "v1.2-de-martin"
    assert profile_for_language("pl") is None


def test_oddadmix_profile_is_explicitly_selectable() -> None:
    profile = get_model_profile("en-oddadmix-7m-distill", "github")

    assert profile.default_voice == "af_msa"
    assert profile.frontend == "pykokoro-native-v1"
    assert profile.vocabulary_source == "downloaded-config"
    assert profile.sample_rate == 24_000
    assert profile.runtime_available is True
    assert profile.voice_names == ("af_msa",)

    resolved = resolve_model_defaults(
        PipelineConfig(
            model_source="github",
            model_variant="en-oddadmix-7m-distill",
            generation=GenerationConfig(lang="en-us"),
        )
    )
    assert resolved.model_variant == "en-oddadmix-7m-distill"
    assert resolved.voice == "af_msa"


def test_oddadmix_does_not_change_automatic_english_or_voice_selection() -> None:
    assert profile_for_language("en") is None
    assert profile_for_voice("af_msa") is None


def test_mateusz_reports_missing_runtime_distribution() -> None:
    with pytest.raises(ValueError, match="present but has no runtime-ready distribution"):
        resolve_model_defaults(
            PipelineConfig(
                model_source="github",
                model_variant="pl-mateusz",
                generation=GenerationConfig(lang="pl"),
            )
        )


def test_anna_resolves_explicit_local_release_assets(tmp_path) -> None:
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(
        '{"assets": ['
        '{"name": "model.onnx", "role": "model"},'
        '{"name": "voices.npz", "role": "voices", "format": "numpy-npz"},'
        '{"name": "config.json", "role": "config"}'
        "]}",
        encoding="utf-8",
    )

    resolved = resolve_model_defaults(
        PipelineConfig(
            release_manifest_path=manifest,
            model_source="github",
            model_variant="de-anna",
            generation=GenerationConfig(lang="de"),
        )
    )

    assert resolved.model_variant == "de-anna"
    assert resolved.voice == "df_anna"
    assert resolved.model_path == tmp_path / "model.onnx"
    assert resolved.voices_path == tmp_path / "voices.npz"
    assert resolved.model_config_path == tmp_path / "config.json"


def test_anna_resolves_as_a_normal_github_model() -> None:
    resolved = resolve_model_defaults(
        PipelineConfig(
            model_source="github",
            model_variant="de-anna",
            generation=GenerationConfig(lang="de"),
        )
    )

    assert resolved.model_source == "github"
    assert resolved.model_variant == "de-anna"
    assert resolved.model_quality == "fp32"
    assert resolved.voice == "df_anna"
    assert resolved.allow_experimental_frontend is False


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


def test_martin_voice_requires_explicit_language() -> None:
    resolved = resolve_model_defaults(
        PipelineConfig(voice="martin", generation=GenerationConfig(lang="de"))
    )
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
    pipeline = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us")))
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
def test_registry_github_profiles_defer_quality_validation(variant, voice):
    cfg = PipelineConfig(
        model_source="github",
        model_variant=variant,
        model_quality="fp32",
        voice=voice,
        generation=GenerationConfig(lang="ru"),
    )

    resolved = resolve_model_defaults(cfg)

    assert resolved.model_source == "github"
    assert resolved.model_variant == variant
    assert resolved.model_quality == "fp32"
    assert resolved.voice == voice


@pytest.mark.parametrize(
    ("voice", "variant"),
    [
        ("sveta", "ru-zaakirio-base"),
        ("masha", "ru-zaakirio-base"),
        ("dima", "ru-zaakirio-dima"),
    ],
)
def test_russian_voice_selection_uses_github_profile(voice, variant) -> None:
    resolved = resolve_model_defaults(
        PipelineConfig(voice=voice, generation=GenerationConfig(lang="ru"))
    )

    assert resolved.model_source == "github"
    assert resolved.model_variant == variant
    assert resolved.generation.lang == "ru"
    assert resolved.model_quality == "fp32"


def test_russian_language_defaults_to_github_base() -> None:
    resolved = resolve_model_defaults(PipelineConfig(generation=GenerationConfig(lang="ru")))

    assert resolved.model_source == "github"
    assert resolved.model_variant == "ru-zaakirio-base"
    assert resolved.voice == "sveta"
    assert resolved.model_quality == "fp32"


@pytest.mark.parametrize("variant", ["ru-zaakirio-base", "ru-zaakirio-dima"])
def test_russian_github_profiles_are_registered(variant) -> None:
    profile = get_model_profile(variant, "github")

    assert profile.source == "github"
    assert profile.variant == variant


def _russian_registry(variant: str, voice: str) -> ModelRegistry:
    return ModelRegistry(
        {
            "schema": 1,
            "runtime_contract": 1,
            "models": {
                variant: {
                    "runtime_available": True,
                    "language_codes": ["ru"],
                    "frontend": "kokorog2p-ru-v1",
                    "runtime": {
                        "layout": "single-onnx-v1",
                        "default_voice": voice,
                        "voices": [voice],
                    },
                    "distributions": [
                        {
                            "id": f"{variant}-github",
                            "provider": "github-release",
                            "transport": "https",
                            "runtime_ready": True,
                            "artifacts": [
                                {
                                    "id": f"{variant}-model",
                                    "role": "model",
                                    "format": "onnx",
                                    "url": "https://fixture.test/model.onnx",
                                    "local_name": "model.onnx",
                                    "size": 1,
                                    "sha256": "0" * 64,
                                    "quality": "fp32",
                                },
                                {
                                    "id": f"{variant}-voices",
                                    "role": "voices",
                                    "format": "numpy-npz",
                                    "url": "https://fixture.test/voices.npz",
                                    "local_name": "voices.npz",
                                    "size": 1,
                                    "sha256": "0" * 64,
                                },
                            ],
                        }
                    ],
                }
            },
        },
        "russian-profile-fixture",
    )


@pytest.mark.parametrize(
    ("variant", "voice"),
    [
        ("ru-zaakirio-base", "sveta"),
        ("ru-zaakirio-dima", "dima"),
    ],
)
def test_registry_github_source_resolves_through_local_profile(variant, voice) -> None:
    registry = _russian_registry(variant, voice)
    registry_profile = get_registry_model_profile(variant, registry=registry)

    assert registry_profile.source == "github"
    assert registry_profile.variant == variant

    resolved = resolve_model_defaults(
        PipelineConfig(
            model_source=registry_profile.source,
            model_variant=registry_profile.variant,
            voice=voice,
            generation=GenerationConfig(lang="ru"),
        )
    )

    assert resolved.model_source == "github"
    assert resolved.model_variant == variant
    assert resolved.voice == voice


@pytest.mark.parametrize(
    ("source", "variant", "voice", "vocab_version"),
    [
        ("github", "v1.0", "af_heart", "1.0"),
        ("github", "v1.1-zh", "af_maple", "1.1"),
        ("huggingface", "v1.0", "af_heart", "1.0"),
        ("huggingface", "v1.1-zh", "af_maple", "1.1"),
    ],
)
def test_legacy_runtime_profiles_do_not_duplicate_asset_inventory(
    source, variant, voice, vocab_version
):
    profile = get_model_profile(variant, source)

    assert profile.default_voice == voice
    assert profile.tokenizer_vocab_version == vocab_version
    assert profile.quality_files == {}
    assert profile.voice_names == ()


def test_default_english_resolves_without_asset_constants():
    resolved = resolve_model_defaults(
        PipelineConfig(
            voice="af_heart",
            generation=GenerationConfig(lang="en-us"),
        )
    )

    assert resolved.model_source == "github"
    assert resolved.model_variant == "v1.0"
    assert resolved.model_quality == "fp32"
    assert resolved.voice == "af_heart"


def test_default_chinese_still_selects_v1_1_zh():
    resolved = resolve_model_defaults(PipelineConfig(generation=GenerationConfig(lang="zh")))

    assert resolved.model_source == "github"
    assert resolved.model_variant == "v1.1-zh"
    assert resolved.voice == "af_maple"


def test_legacy_quality_validation_is_deferred_to_runtime_registry():
    resolved = resolve_model_defaults(
        PipelineConfig(
            model_source="github",
            model_variant="v1.0",
            model_quality="q8",
            generation=GenerationConfig(lang="en-us"),
        )
    )

    assert resolved.model_quality == "q8"
