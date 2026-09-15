import json

import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.model_profiles import get_model_profile
from pykokoro.pipeline_config import PipelineConfig, resolve_model_defaults


def test_runtime_profiles_do_not_duplicate_published_inventory():
    expected_backends = {
        "vi-contextbox": "espeak",
        "vi-anphunl": "espeak",
        "ar-nabra": None,
        "vi-ngoc-huyen": "espeak",
        "de-crane": "kokorog2p",
        "he-hebrew-nc": "espeak",
    }
    for variant in (
        "vi-contextbox",
        "vi-anphunl",
        "vi-ngoc-huyen",
        "ar-nabra",
        "de-crane",
        "he-hebrew-nc",
    ):
        profile = get_model_profile(variant, "github")
        assert profile.quality_files == {}
        assert profile.voice_names == ()
        assert profile.onnx_inputs["speed"] == "float32"
        assert profile.frontend_experimental is (variant not in {"ar-nabra", "de-crane"})
        assert profile.g2p_backend == expected_backends[variant]
    nabra = get_model_profile("ar-nabra", "github")
    assert nabra.vocabulary_source == "downloaded-release"
    assert not hasattr(nabra, "vocabulary_filename")
    assert nabra.onnx_inputs == {
        "input_ids": "int64",
        "ref_s": "float32",
        "speed": "float32",
    }
    assert nabra.max_tokens == 510


def test_vi_contextbox_uses_published_named_default_voice():
    profile = get_model_profile("vi-contextbox", "github")

    assert profile.default_voice == "diem_trinh"


def test_vi_ngoc_huyen_uses_published_named_default_voice():
    profile = get_model_profile("vi-ngoc-huyen", "github")

    assert profile.default_voice == "ngoc_huyen"


def test_publication_policy_is_remote_manifest_metadata():
    assert not hasattr(get_model_profile("he-hebrew-nc", "github"), "publication_enabled")


def test_release_manifest_resolves_explicit_assets(tmp_path):
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {"name": "model.onnx", "role": "model", "quality": "fp32"},
                    {"name": "voices.npz", "role": "voices", "format": "numpy-npz"},
                    {"name": "config.json", "role": "config", "format": "json"},
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_defaults(
        PipelineConfig(
            release_manifest_path=manifest,
            model_variant="vi-contextbox",
            generation=GenerationConfig(lang="vi"),
        )
    )

    assert resolved.model_path == tmp_path / "model.onnx"
    assert resolved.voices_path == tmp_path / "voices.npz"
    assert resolved.model_config_path == tmp_path / "config.json"


def test_release_manifest_selects_requested_model_quality(tmp_path):
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 2,
                "assets": [
                    {"name": "model-fp32.onnx", "role": "model", "quality": "fp32"},
                    {"name": "model-q8.onnx", "role": "model", "quality": "q8"},
                    {"name": "voices.npz", "role": "voices", "format": "numpy-npz"},
                ],
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_defaults(
        PipelineConfig(
            release_manifest_path=manifest,
            model_source="github",
            model_variant="v1.0",
            model_quality="q8",
            generation=GenerationConfig(lang="en"),
        )
    )

    assert resolved.model_quality == "q8"
    assert resolved.model_path == tmp_path / "model-q8.onnx"


def test_release_manifest_automatic_quality_ignores_asset_order(tmp_path):
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 2,
                "assets": [
                    {"name": "model-q8.onnx", "role": "model", "quality": "q8"},
                    {"name": "model-fp32.onnx", "role": "model", "quality": "fp32"},
                    {"name": "voices.npz", "role": "voices", "format": "numpy-npz"},
                ],
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_defaults(
        PipelineConfig(
            release_manifest_path=manifest,
            model_source="github",
            model_variant="v1.0",
            generation=GenerationConfig(lang="en"),
        )
    )

    assert resolved.model_quality == "fp32"
    assert resolved.model_path == tmp_path / "model-fp32.onnx"


def test_release_manifest_rejects_unavailable_quality(tmp_path):
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 2,
                "assets": [
                    {"name": "model-fp32.onnx", "role": "model", "quality": "fp32"},
                    {"name": "voices.npz", "role": "voices", "format": "numpy-npz"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"no model asset for quality 'q8'.*Available: fp32",
    ):
        resolve_model_defaults(
            PipelineConfig(
                release_manifest_path=manifest,
                model_source="github",
                model_variant="v1.0",
                model_quality="q8",
                generation=GenerationConfig(lang="en"),
            )
        )


def test_release_manifest_rejects_duplicate_quality(tmp_path):
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 2,
                "assets": [
                    {"name": "model-q8-a.onnx", "role": "model", "quality": "q8"},
                    {"name": "model-q8-b.onnx", "role": "model", "quality": "q8"},
                    {"name": "voices.npz", "role": "voices", "format": "numpy-npz"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"multiple model assets for quality 'q8'"):
        resolve_model_defaults(
            PipelineConfig(
                release_manifest_path=manifest,
                model_source="github",
                model_variant="v1.0",
                model_quality="q8",
                generation=GenerationConfig(lang="en"),
            )
        )


def test_release_manifest_preserves_single_unqualified_model(tmp_path):
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {"name": "model.onnx", "role": "model"},
                    {"name": "voices.npz", "role": "voices", "format": "numpy-npz"},
                ],
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_defaults(
        PipelineConfig(
            release_manifest_path=manifest,
            model_source="github",
            model_variant="v1.0",
            generation=GenerationConfig(lang="en"),
        )
    )

    assert resolved.model_quality == "fp32"
    assert resolved.model_path == tmp_path / "model.onnx"


def test_release_manifest_preserves_explicit_model_path_and_other_assets(tmp_path):
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 2,
                "assets": [
                    {"name": "model-q8.onnx", "role": "model", "quality": "q8"},
                    {"name": "voices.npz", "role": "voices", "format": "numpy-npz"},
                    {"name": "config.json", "role": "config"},
                ],
            }
        ),
        encoding="utf-8",
    )
    custom_model = tmp_path / "custom.onnx"

    resolved = resolve_model_defaults(
        PipelineConfig(
            release_manifest_path=manifest,
            model_source="github",
            model_variant="v1.0",
            model_quality="q8",
            model_path=custom_model,
            generation=GenerationConfig(lang="en"),
        )
    )

    assert resolved.model_path == custom_model
    assert resolved.voices_path == tmp_path / "voices.npz"
    assert resolved.model_config_path == tmp_path / "config.json"


def test_nabra_loads_direct_release_vocabulary(tmp_path):
    from pykokoro.onnx_backend import load_vocab_from_config

    vocab_path = tmp_path / "vocab-arabic-nabra-v0.1.json"
    vocab_path.write_text('{"ʕ": 7, "ħ": 8, "a": 43}', encoding="utf-8")

    vocab = load_vocab_from_config("ar-nabra", vocab_path)

    assert vocab == {"ʕ": 7, "ħ": 8, "a": 43}
