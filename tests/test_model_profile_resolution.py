from __future__ import annotations

import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.model_profiles import GERMAN_MARTIN_V1_2, get_model_profile
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig, resolve_model_defaults


def test_martin_profile_metadata_is_pinned():
    profile = get_model_profile("v1.2-de-martin", "github")
    assert profile == GERMAN_MARTIN_V1_2
    assert profile.release_tag == "model-files-german-martin-v1.2"
    assert profile.release_revision == "670bf630bb02428ad323f78195f9583f52c5c604"
    assert profile.model_sha256 is not None
    assert profile.voices_sha256 == "5b9c8553398d7abf67498ce500c186cefaa7b68fed3e3d415da5380670105acd"
    assert profile.model_sizes == {"kokoro-german-martin-v1.2.onnx": 325_512_630}
    assert profile.voices_size == 522_506
    assert profile.suggested_speed == 1.125


@pytest.mark.parametrize("lang", ["de", "de-DE", "de_at", "de-ch"])
def test_german_defaults_resolve_to_martin(lang):
    resolved = resolve_model_defaults(PipelineConfig(generation=GenerationConfig(lang=lang)))
    assert resolved.model_source == "github"
    assert resolved.model_variant == "v1.2-de-martin"
    assert resolved.model_quality == "fp32"
    assert resolved.voice == "martin"
    assert resolved.generation.lang == lang.lower().replace("_", "-")


def test_explicit_legacy_german_configuration_is_preserved():
    resolved = resolve_model_defaults(
        PipelineConfig(
            voice="dm_bernd",
            model_source="github",
            model_variant="v1.1-de",
            generation=GenerationConfig(lang="de"),
        )
    )
    assert resolved.model_variant == "v1.1-de"
    assert resolved.voice == "dm_bernd"


def test_incompatible_explicit_martin_voice_is_rejected():
    with pytest.raises(ValueError, match="Voice 'df_eva'.*martin"):
        resolve_model_defaults(
            PipelineConfig(
                voice="df_eva",
                model_variant="v1.2-de-martin",
                generation=GenerationConfig(lang="de"),
            )
        )


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
