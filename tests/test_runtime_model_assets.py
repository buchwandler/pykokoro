from __future__ import annotations

import copy
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
    verify_artifact,
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


def _anna_artifact(
    artifact_id: str,
    role: str,
    format: str,
    local_name: str,
    quality: str | None = None,
) -> dict[str, object]:
    payload = artifact_id.encode()
    artifact: dict[str, object] = {
        "id": artifact_id,
        "role": role,
        "format": format,
        "url": f"https://github.test/{local_name}",
        "local_name": local_name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if quality is not None:
        artifact["quality"] = quality
    return artifact


def _quality_registry() -> ModelRegistry:
    data = copy.deepcopy(_registry().data)
    data["models"]["test-model"]["distributions"][0]["artifacts"] = [
        _anna_artifact("fp32-model", "model", "onnx", "model-fp32.onnx", "fp32"),
        _anna_artifact("fp16-model", "model", "onnx", "model-fp16.onnx", "fp16"),
        _anna_artifact("q8-model", "model", "onnx", "model-q8.onnx", "q8"),
        _anna_artifact("quality-voices", "voices", "numpy-npz", "voices.npz"),
        _anna_artifact("quality-config", "config", "json", "config.json"),
        _anna_artifact("quality-metadata", "metadata", "json", "metadata.json"),
    ]
    return ModelRegistry(data, "quality-fixture")


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


def _anna_registry() -> ModelRegistry:
    return ModelRegistry(
        {
            "schema": 1,
            "runtime_contract": 1,
            "models": {
                "de-anna": {
                    "model_version": "1",
                    "runtime_available": True,
                    "language_codes": ["de"],
                    "frontend": "german-ipa-v1",
                    "sample_rate": 24000,
                    "runtime": {
                        "layout": "single-onnx-v1",
                        "max_tokens": 510,
                        "default_voice": "df_anna",
                        "voices": ["df_anna"],
                    },
                    "distributions": [
                        {
                            "id": "anna-github",
                            "provider": "github-release",
                            "transport": "https",
                            "runtime_ready": True,
                            "release_key": "de-anna",
                            "release_tag": "model-files-german-software-mansion-anna-v1",
                            "artifacts": [
                                _anna_artifact("anna-model", "model", "onnx", "model.onnx", "fp32"),
                                _anna_artifact("anna-voices", "voices", "numpy-npz", "voices.npz"),
                                _anna_artifact("anna-config", "config", "json", "config.json"),
                                _anna_artifact("anna-bundle", "bundle", "json", "bundle.json"),
                            ],
                        }
                    ],
                }
            },
        },
        "anna-fixture",
    )


def _portuguese_registry() -> ModelRegistry:
    return ModelRegistry(
        {
            "schema": 1,
            "runtime_contract": 1,
            "models": {
                "pt-eu-logus2k": {
                    "model_version": "1.0",
                    "runtime_available": True,
                    "language_codes": ["pt"],
                    "frontend": "tts-eu-pt-v1",
                    "sample_rate": 24000,
                    "runtime": {
                        "layout": "single-onnx-v1",
                        "max_tokens": 510,
                        "default_voice": "pt_eu",
                        "voices": ["pt_eu"],
                    },
                    "distributions": [
                        {
                            "id": "portuguese-github",
                            "provider": "github-release",
                            "transport": "https",
                            "runtime_ready": True,
                            "release_key": "pt-eu-logus2k",
                            "release_tag": "model-files-portuguese-eu-pt-v1.0",
                            "artifacts": [
                                _anna_artifact(
                                    "portuguese-model",
                                    "model",
                                    "onnx",
                                    "kokoro-portuguese-eu-pt-v1.0.onnx",
                                    "fp32",
                                ),
                                _anna_artifact(
                                    "portuguese-voices",
                                    "voices",
                                    "numpy-npz",
                                    "voices-portuguese-eu-pt-v1.0.npz",
                                ),
                                _anna_artifact(
                                    "portuguese-config",
                                    "config",
                                    "json",
                                    "config-portuguese-eu-pt-v1.0.json",
                                ),
                                _anna_artifact(
                                    "portuguese-bundle", "bundle", "json", "bundle.json"
                                ),
                            ],
                        }
                    ],
                }
            },
        },
        "portuguese-fixture",
    )


def test_resolver_materializes_only_selected_model_quality(
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
        model_id="test-model", quality="fp32", registry=_quality_registry(), cache_dir=tmp_path
    )

    assert set(calls) == {
        "fp32-model",
        "quality-voices",
        "quality-config",
        "quality-metadata",
    }
    assert "fp16-model" not in resolved.artifacts
    assert "q8-model" not in resolved.artifacts


def test_resolver_warm_cache_verifies_only_selected_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _quality_registry()
    distribution = registry.model("test-model").distribution()
    selected = [
        artifact
        for artifact in distribution.artifacts
        if artifact.role != "model" or artifact.quality == "fp32"
    ]
    for artifact in selected:
        target = tmp_path / "test-model" / distribution.id / artifact.local_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(artifact.id.encode())

    verified: list[str] = []
    original_verify = verify_artifact

    def record_verify(path, artifact):
        verified.append(artifact.id)
        original_verify(path, artifact)

    monkeypatch.setattr("pykokoro.runtime.model_assets.verify_artifact", record_verify)
    resolve_runtime_assets(
        model_id="test-model", quality="fp32", registry=registry, cache_dir=tmp_path
    )

    assert set(verified) == {artifact.id for artifact in selected}
    assert "fp16-model" not in verified
    assert "q8-model" not in verified


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


def test_anna_registry_profile_and_runtime_assets_are_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def materialize(artifact, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(artifact.id.encode())
        return target

    monkeypatch.setattr("pykokoro.runtime.model_assets.download_artifact", materialize)

    profile = get_registry_model_profile("de-anna", registry=_anna_registry())
    assert profile.source == "github"
    assert profile.variant == "de-anna"
    assert profile.frontend == "german-ipa-v1"
    assert profile.frontend_experimental is False
    assert profile.g2p_backend == "kokorog2p"
    assert profile.runtime_available is True
    assert profile.support_status == "ready"
    assert profile.default_voice == "df_anna"
    assert profile.available_qualities == ("fp32",)

    assets = resolve_runtime_assets(
        model_id="de-anna",
        quality="fp32",
        registry=_anna_registry(),
        cache_dir=tmp_path,
    )
    assert assets.distribution_id == "anna-github"
    assert assets.provider == "github-release"
    assert assets.artifact_for_role("model", quality="fp32").name == "model.onnx"
    assert assets.artifact_for_role("voices").name == "voices.npz"
    assert assets.artifact_for_role("config").name == "config.json"
    assert assets.artifact_for_role("bundle").name == "bundle.json"
    assert assets.artifact_for_role("voices").suffix == ".npz"


def test_portuguese_registry_profile_and_runtime_assets_are_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def materialize(artifact, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(artifact.id.encode())
        return target

    monkeypatch.setattr("pykokoro.runtime.model_assets.download_artifact", materialize)

    registry = _portuguese_registry()
    profile = get_registry_model_profile("pt-eu-logus2k", registry=registry)
    assert profile.source == "github"
    assert profile.variant == "pt-eu-logus2k"
    assert profile.frontend == "tts-eu-pt-v1"
    assert profile.g2p_backend == "kokorog2p"
    assert profile.language_codes == ("pt-pt",)
    assert profile.default_voice == "pt_eu"
    assert profile.available_qualities == ("fp32",)
    assert profile.runtime_available is True
    assert profile.redistribution_allowed is True
    assert profile.support_status == "ready"

    assets = resolve_runtime_assets(
        model_id="pt-eu-logus2k",
        quality="fp32",
        registry=registry,
        cache_dir=tmp_path,
    )
    assert assets.distribution_id == "portuguese-github"
    assert assets.provider == "github-release"
    assert assets.artifact_for_role("model", quality="fp32").name == (
        "kokoro-portuguese-eu-pt-v1.0.onnx"
    )
    assert assets.artifact_for_role("voices").name == "voices-portuguese-eu-pt-v1.0.npz"
    assert assets.artifact_for_role("config").name == "config-portuguese-eu-pt-v1.0.json"
    assert assets.artifact_for_role("bundle").name == "bundle.json"


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


def _registry_with_model_bytes(payload: bytes, source: str) -> ModelRegistry:
    data = copy.deepcopy(_registry().data)
    artifact = data["models"]["test-model"]["distributions"][0]["artifacts"][0]
    artifact["size"] = len(payload)
    artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    return ModelRegistry(data, source)


class RefreshingClient:
    def __init__(self, initial: ModelRegistry, refreshed: ModelRegistry) -> None:
        self.initial = initial
        self.refreshed = refreshed
        self.normal_loads = 0
        self.refresh_loads = 0

    def load(
        self, *, offline: bool = False, refresh: bool = False, allow_cache_fallback: bool = True
    ):
        if refresh:
            self.refresh_loads += 1
            return self.refreshed
        self.normal_loads += 1
        return self.initial


def test_resolver_refreshes_once_when_cached_registry_lacks_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial_data = copy.deepcopy(_registry().data)
    del initial_data["models"]["test-model"]
    client = RefreshingClient(ModelRegistry(initial_data, "cached"), _registry())

    def download(artifact, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(artifact.id.encode())
        verify_artifact(path, artifact)
        return path

    monkeypatch.setattr("pykokoro.runtime.model_assets.download_artifact", download)
    resolved = resolve_runtime_assets(
        model_id="test-model", registry_client=client, cache_dir=tmp_path
    )

    assert resolved.model_id == "test-model"
    assert client.normal_loads == 1
    assert client.refresh_loads == 1


def test_resolver_refreshes_once_when_cached_registry_lacks_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial_data = copy.deepcopy(_registry().data)
    initial_data["models"]["test-model"]["distributions"][0]["artifacts"][0]["quality"] = "q8"
    client = RefreshingClient(ModelRegistry(initial_data, "cached"), _registry())

    def download(artifact, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(artifact.id.encode())
        verify_artifact(path, artifact)
        return path

    monkeypatch.setattr("pykokoro.runtime.model_assets.download_artifact", download)
    resolved = resolve_runtime_assets(
        model_id="test-model", quality="fp32", registry_client=client, cache_dir=tmp_path
    )

    assert resolved.artifact("github-model").is_file()
    assert client.normal_loads == 1
    assert client.refresh_loads == 1


def test_resolver_does_not_loop_after_resolution_refresh_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial_data = copy.deepcopy(_registry().data)
    del initial_data["models"]["test-model"]
    client = RefreshingClient(
        ModelRegistry(initial_data, "cached"), ModelRegistry(initial_data, "still-cached")
    )
    monkeypatch.setattr(
        "pykokoro.runtime.model_assets.download_artifact",
        lambda *args, **kwargs: pytest.fail("resolution failure must not download"),
    )

    with pytest.raises(ModelRegistryError, match="after one registry refresh"):
        resolve_runtime_assets(model_id="test-model", registry_client=client, cache_dir=tmp_path)

    assert client.normal_loads == 1
    assert client.refresh_loads == 1


def _download_payload(artifact: RuntimeArtifact) -> bytes:
    if artifact.id == "github-model":
        return b"new-model"
    return artifact.id.encode()


def test_resolver_replaces_stale_cached_artifact_without_directory_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry()
    target = tmp_path / "test-model" / "github-dist" / "model.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old-model")
    sentinel = target.parent / "keep-me"
    sentinel.write_text("keep", encoding="utf-8")

    def download(artifact, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(artifact.id.encode())
        verify_artifact(path, artifact)
        return path

    monkeypatch.setattr("pykokoro.runtime.model_assets.download_artifact", download)
    resolved = resolve_runtime_assets(model_id="test-model", registry=registry, cache_dir=tmp_path)

    assert resolved.artifact("github-model").read_bytes() == b"github-model"
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert target.parent.is_dir()


def test_resolver_refreshes_registry_once_after_download_integrity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_payload = b"old-model"
    new_payload = b"new-model"
    client = RefreshingClient(
        _registry_with_model_bytes(old_payload, "cached"),
        _registry_with_model_bytes(new_payload, "fresh"),
    )

    def download(artifact, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_download_payload(artifact))
        verify_artifact(path, artifact)
        return path

    monkeypatch.setattr("pykokoro.runtime.model_assets.download_artifact", download)
    resolved = resolve_runtime_assets(
        model_id="test-model", registry_client=client, cache_dir=tmp_path
    )

    assert resolved.artifact("github-model").read_bytes() == new_payload
    assert client.normal_loads == 1
    assert client.refresh_loads == 1


def test_resolver_does_not_loop_when_refreshed_registry_still_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = RefreshingClient(
        _registry_with_model_bytes(b"old-model", "cached"),
        _registry_with_model_bytes(b"other-model", "fresh"),
    )

    def download(artifact, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wrong-data")
        verify_artifact(path, artifact)
        return path

    monkeypatch.setattr("pykokoro.runtime.model_assets.download_artifact", download)
    with pytest.raises(ModelRegistryError, match="published release and catalog are inconsistent"):
        resolve_runtime_assets(model_id="test-model", registry_client=client, cache_dir=tmp_path)

    assert client.normal_loads == 1
    assert client.refresh_loads == 1


def test_resolver_offline_integrity_failure_never_refreshes_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry()
    client = RefreshingClient(registry, registry)
    target = tmp_path / "test-model" / "github-dist" / "model.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bad")

    with pytest.raises(ModelRegistryError, match="Offline mode"):
        resolve_runtime_assets(
            model_id="test-model", registry_client=client, offline=True, cache_dir=tmp_path
        )

    assert client.normal_loads == 1
    assert client.refresh_loads == 0


def test_resolver_does_not_replace_explicit_registry_after_integrity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry_with_model_bytes(b"expected", "explicit")
    client = RefreshingClient(registry, _registry_with_model_bytes(b"fresh", "unexpected"))

    def download(artifact, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wrong")
        verify_artifact(path, artifact)
        return path

    monkeypatch.setattr("pykokoro.runtime.model_assets.download_artifact", download)
    with pytest.raises(ModelRegistryError, match="SHA-256 mismatch|Size mismatch"):
        resolve_runtime_assets(
            model_id="test-model", registry=registry, registry_client=client, cache_dir=tmp_path
        )

    assert client.normal_loads == 0
    assert client.refresh_loads == 0


def test_thorsten_style_stale_registry_refreshes_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_registry = _registry_with_model_bytes(b"old!!", "models-cache")
    new_registry = _registry_with_model_bytes(b"new-data", "models-fresh")
    client = RefreshingClient(old_registry, new_registry)

    def download(artifact, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"new-data" if artifact.id == "github-model" else artifact.id.encode())
        verify_artifact(path, artifact)
        return path

    monkeypatch.setattr("pykokoro.runtime.model_assets.download_artifact", download)
    resolved = resolve_runtime_assets(
        model_id="test-model", registry_client=client, cache_dir=tmp_path
    )

    assert resolved.artifact("github-model").read_bytes() == b"new-data"
    assert client.refresh_loads == 1


def _download_with_progress(artifact, target, *, progress_callback=None, phase_callback=None):
    payload = artifact.id.encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    if progress_callback is not None:
        progress_callback(artifact, len(payload), artifact.size)
    if phase_callback is not None:
        phase_callback("verify")
    verify_artifact(target, artifact)
    return target


def test_resolver_reports_download_event_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "pykokoro.runtime.model_assets.download_artifact",
        _download_with_progress,
    )
    events = []

    resolve_runtime_assets(
        model_id="test-model",
        quality="fp32",
        registry=_registry(),
        cache_dir=tmp_path,
        progress_callback=events.append,
    )

    assert [event.phase for event in events] == [
        "download-start",
        "download-progress",
        "verify-start",
        "download-complete",
        "download-start",
        "download-progress",
        "verify-start",
        "download-complete",
    ]
    assert all(event.target for event in events)


def test_resolver_does_not_report_valid_cache_hits(tmp_path: Path) -> None:
    registry = _registry()
    distribution = registry.model("test-model").distribution()
    for artifact in distribution.artifacts:
        target = tmp_path / "test-model" / distribution.id / artifact.local_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(artifact.id.encode())

    events = []
    resolve_runtime_assets(
        model_id="test-model",
        quality="fp32",
        registry=registry,
        cache_dir=tmp_path,
        progress_callback=events.append,
    )

    assert events == []


def test_resolver_reports_invalid_cache_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "test-model" / "github-dist" / "model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"invalid")
    monkeypatch.setattr(
        "pykokoro.runtime.model_assets.download_artifact",
        _download_with_progress,
    )
    events = []

    resolve_runtime_assets(
        model_id="test-model",
        quality="fp32",
        registry=_registry(),
        cache_dir=tmp_path,
        progress_callback=events.append,
    )

    assert events[0].phase == "download-start"
    assert events[-1].phase == "download-complete"
    assert target.read_bytes() == b"github-model"


def test_resolver_offline_does_not_report_download_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "pykokoro.runtime.model_assets.download_artifact",
        lambda *args, **kwargs: pytest.fail("offline resolution must not download"),
    )
    events = []

    with pytest.raises(ModelRegistryError, match="Offline mode"):
        resolve_runtime_assets(
            model_id="test-model",
            quality="fp32",
            registry=_registry(),
            cache_dir=tmp_path,
            offline=True,
            progress_callback=events.append,
        )

    assert events == []


def test_resolver_failed_download_has_no_completion_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_download(*args, **kwargs):
        raise OSError("network failed")

    monkeypatch.setattr("pykokoro.runtime.model_assets.download_artifact", fail_download)
    events = []

    with pytest.raises(OSError, match="network failed"):
        resolve_runtime_assets(
            model_id="test-model",
            quality="fp32",
            registry=_registry(),
            cache_dir=tmp_path,
            progress_callback=events.append,
        )

    assert [event.phase for event in events] == ["download-start"]
