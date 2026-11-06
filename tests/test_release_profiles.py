import json
from pathlib import Path

import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.model_profiles import get_model_profile
from pykokoro.synthesis_config import SynthesisConfig, resolve_synthesis_config


def _resolve(config: SynthesisConfig) -> SynthesisConfig:
    return resolve_synthesis_config(
        config, language=config.generation.lang or "en-us", voice=config.voice
    )


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

    resolved = _resolve(
        SynthesisConfig(
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

    resolved = _resolve(
        SynthesisConfig(
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

    resolved = _resolve(
        SynthesisConfig(
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
        _resolve(
            SynthesisConfig(
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
        _resolve(
            SynthesisConfig(
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

    resolved = _resolve(
        SynthesisConfig(
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

    resolved = _resolve(
        SynthesisConfig(
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


_ROUTER_LANGUAGE_MODELS = {"en": "model_en", "de": "model_de"}


def _write_language_router_manifest(
    tmp_path: Path,
    *,
    assets: list[dict[str, object]] | None = None,
    runtime: dict[str, object] | None = None,
) -> Path:
    manifest_assets = assets or [
        {
            "name": "model_en.onnx",
            "role": "model",
            "component": "model_en",
            "quality": "fp32",
        },
        {
            "name": "model_de.onnx",
            "role": "model",
            "component": "model_de",
            "quality": "fp32",
        },
        {"name": "voices.npz", "role": "voices", "format": "numpy-npz"},
    ]
    for asset in manifest_assets:
        name = asset.get("name")
        if isinstance(name, str):
            (tmp_path / name).write_bytes(b"asset")
    manifest = tmp_path / "language-router-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 2,
                "runtime": runtime
                or {
                    "layout": "language-router-onnx-v1",
                    "language_models": _ROUTER_LANGUAGE_MODELS,
                },
                "assets": manifest_assets,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _resolve_language_router_manifest(manifest: Path) -> SynthesisConfig:
    return _resolve(
        SynthesisConfig(
            release_manifest_path=manifest,
            model_source="github",
            model_variant="akinvox-multispeaker-v1",
            model_quality="fp32",
            voice="AV_cori",
            generation=GenerationConfig(lang="en"),
        )
    )


def test_release_manifest_resolves_language_router_components_by_quality(tmp_path: Path) -> None:
    assets = [
        {
            "name": f"{component}-{quality}.onnx",
            "role": "model",
            "component": component,
            "quality": quality,
        }
        for component in ("model_en", "model_de")
        for quality in ("q8", "fp32")
    ]
    assets.append({"name": "voices.npz", "role": "voices", "format": "numpy-npz"})
    manifest = _write_language_router_manifest(tmp_path, assets=assets)

    resolved = _resolve_language_router_manifest(manifest)

    assert resolved.model_path is None
    assert resolved.model_artifacts == {
        "model_en": tmp_path / "model_en-fp32.onnx",
        "model_de": tmp_path / "model_de-fp32.onnx",
    }
    assert resolved.voices_path == tmp_path / "voices.npz"


def test_release_manifest_rejects_missing_language_router_component(tmp_path: Path) -> None:
    assets = [item for item in _write_router_assets() if item.get("component") != "model_de"]
    manifest = _write_language_router_manifest(tmp_path, assets=assets)

    with pytest.raises(ValueError, match="exactly one model asset.*model_de.*fp32"):
        _resolve_language_router_manifest(manifest)


def test_release_manifest_rejects_duplicate_language_router_component(tmp_path: Path) -> None:
    assets = _write_router_assets()
    assets.append(
        {
            "name": "model_en-copy.onnx",
            "role": "model",
            "component": "model_en",
            "quality": "fp32",
        }
    )
    manifest = _write_language_router_manifest(tmp_path, assets=assets)

    with pytest.raises(ValueError, match="exactly one model asset.*model_en.*fp32"):
        _resolve_language_router_manifest(manifest)


def test_release_manifest_rejects_wrong_language_router_quality(tmp_path: Path) -> None:
    assets = _write_router_assets()
    assets[1]["quality"] = "q8"
    manifest = _write_language_router_manifest(tmp_path, assets=assets)

    with pytest.raises(ValueError, match="exactly one model asset.*model_de.*fp32"):
        _resolve_language_router_manifest(manifest)


@pytest.mark.parametrize(
    ("runtime", "message"),
    [
        ({"layout": "single-onnx-v1", "language_models": _ROUTER_LANGUAGE_MODELS}, "layout"),
        (
            {"layout": "language-router-onnx-v1", "language_models": {"en": "model_de"}},
            "language_models",
        ),
    ],
)
def test_release_manifest_rejects_invalid_language_router_metadata(
    tmp_path: Path, runtime: dict[str, object], message: str
) -> None:
    manifest = _write_language_router_manifest(tmp_path, runtime=runtime)

    with pytest.raises(ValueError, match=message):
        _resolve_language_router_manifest(manifest)


def test_release_manifest_requires_existing_language_router_voices_npz(tmp_path: Path) -> None:
    manifest = _write_language_router_manifest(tmp_path)
    (tmp_path / "voices.npz").unlink()

    with pytest.raises(ValueError, match="voices NPZ does not exist"):
        _resolve_language_router_manifest(manifest)


def _write_router_assets() -> list[dict[str, object]]:
    return [
        {
            "name": "model_en.onnx",
            "role": "model",
            "component": "model_en",
            "quality": "fp32",
        },
        {
            "name": "model_de.onnx",
            "role": "model",
            "component": "model_de",
            "quality": "fp32",
        },
        {"name": "voices.npz", "role": "voices", "format": "numpy-npz"},
    ]
