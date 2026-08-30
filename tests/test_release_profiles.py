import json

from pykokoro.generation_config import GenerationConfig
from pykokoro.model_profiles import get_model_profile
from pykokoro.pipeline_config import PipelineConfig, resolve_model_defaults


def test_runtime_profiles_do_not_duplicate_published_inventory():
    expected_backends = {
        "vi-contextbox": "espeak",
        "vi-anphunl": "espeak",
        "ar-nabra": None,
        "de-crane": "kokorog2p",
        "he-hebrew-nc": "espeak",
    }
    for variant in ("vi-contextbox", "vi-anphunl", "ar-nabra", "de-crane", "he-hebrew-nc"):
        profile = get_model_profile(variant, "github")
        assert profile.quality_files == {}
        assert profile.voice_names == ()
        assert profile.onnx_inputs["speed"] == "float32"
        assert profile.frontend_experimental is (variant != "ar-nabra")
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


def test_nabra_loads_direct_release_vocabulary(tmp_path):
    from pykokoro.onnx_backend import load_vocab_from_config

    vocab_path = tmp_path / "vocab-arabic-nabra-v0.1.json"
    vocab_path.write_text('{"ʕ": 7, "ħ": 8, "a": 43}', encoding="utf-8")

    vocab = load_vocab_from_config("ar-nabra", vocab_path)

    assert vocab == {"ʕ": 7, "ħ": 8, "a": 43}
