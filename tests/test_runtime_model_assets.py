from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from pykokoro.model_profiles import get_registry_model_profile
from pykokoro.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    RuntimeArtifact,
    RuntimeDistribution,
    RuntimeModel,
)
from pykokoro.runtime.model_assets import ResolvedRuntimeAssets, resolve_runtime_assets


def _artifact(artifact_id: str, url: str, local_name: str) -> dict[str, object]:
    payload = artifact_id.encode()
    return {
        "id": artifact_id,
        "role": "model" if artifact_id.endswith("model") else "voices",
        "quality": "fp32" if artifact_id.endswith("model") else None,
        "format": "onnx" if artifact_id.endswith("model") else "numpy-npz",
        "url": url,
        "local_name": local_name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _registry() -> ModelRegistry:
    return ModelRegistry(
        {
            "schema": 1,
            "runtime_contract": 1,
            "models": {
                "test-model": {
                    "runtime_available": True,
                    "language_codes": ["en"],
                    "frontend": "test-frontend",
                    "runtime": {
                        "layout": "single-onnx-v1",
                        "default_voice": "default",
                        "voices": ["default"],
                    },
                    "distributions": [
                        {
                            "id": "github-dist",
                            "provider": "github-release",
                            "transport": "https",
                            "runtime_ready": True,
                            "artifacts": [
                                _artifact("github-model", "https://github/model", "model.onnx"),
                                _artifact("github-voices", "https://github/voices", "voices.npz"),
                            ],
                        },
                        {
                            "id": "hf-dist",
                            "provider": "huggingface",
                            "transport": "https",
                            "runtime_ready": True,
                            "artifacts": [
                                _artifact("hf-model", "https://hf/model", "model.onnx"),
                                _artifact("hf-voices", "https://hf/voices", "voices.npz"),
                            ],
                        },
                    ],
                },
                "unavailable": {
                    "runtime_available": False,
                    "runtime": {
                        "layout": "single-onnx-v1",
                        "default_voice": "default",
                        "voices": ["default"],
                    },
                    "distributions": [],
                },
            },
        },
        "test",
    )


def test_resolver_materializes_one_atomic_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def materialize(artifact, target):
        calls.append(artifact.id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(artifact.id.encode())
        return target

    monkeypatch.setattr("pykokoro.runtime.model_assets.download_artifact", materialize)

    resolved = resolve_runtime_assets(
        model_id="test-model", quality="fp32", registry=_registry(), cache_dir=tmp_path
    )

    assert isinstance(resolved, ResolvedRuntimeAssets)
    assert resolved.distribution_id == "github-dist"
    assert set(calls) == {"github-model", "github-voices"}
    assert all("github-dist" in str(path) for path in resolved.artifacts.values())


def test_resolver_rejects_registry_unavailable_model_without_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pykokoro.runtime.model_assets.download_artifact",
        lambda *args: pytest.fail("unavailable models must not download"),
    )

    with pytest.raises(ModelRegistryError, match="no runtime-ready distribution"):
        resolve_runtime_assets(model_id="unavailable", registry=_registry())


def test_registry_profile_uses_canonical_runtime_metadata() -> None:
    profile = get_registry_model_profile("test-model", registry=_registry())

    assert profile.default_voice == "default"
    assert profile.voice_names == ("default",)
    assert profile.frontend == "test-frontend"
    assert profile.layout == "single-onnx-v1"
    assert profile.support_status == "unsupported-frontend"


def test_raw_voice_materialization_preserves_shape_and_provenance(tmp_path: Path) -> None:
    raw_path = tmp_path / "sveta.bin"
    raw_path.write_bytes(np.arange(6, dtype="<f4").tobytes())
    artifact = RuntimeArtifact(
        "voice-sveta",
        "voice",
        "raw-float32-le",
        "https://example/sveta",
        raw_path.name,
        raw_path.stat().st_size,
        hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        voice="sveta",
        handling={"dtype": "float32", "shape": [2, 3], "endianness": "little"},
    )
    distribution = RuntimeDistribution(
        "upstream", "huggingface", "https", True, (artifact,), revision="pinned"
    )
    model = RuntimeModel(
        "ru",
        {"runtime": {"layout": "single-onnx-v1", "default_voice": "sveta", "voices": ["sveta"]}},
        (distribution,),
    )
    assets = ResolvedRuntimeAssets(
        "ru",
        "upstream",
        "huggingface",
        "single-onnx-v1",
        {artifact.id: raw_path},
        model,
        distribution,
    )

    materialized = assets.materialize_raw_voices()
    with np.load(materialized, allow_pickle=False) as archive:
        assert archive["sveta"].shape == (2, 1, 3)
        np.testing.assert_array_equal(archive["sveta"].reshape(-1), np.arange(6, dtype=np.float32))
    assert materialized.with_suffix(".json").is_file()
