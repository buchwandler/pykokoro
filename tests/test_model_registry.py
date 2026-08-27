from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pykokoro.model_registry import (
    ModelRegistryError,
    RegistryClient,
    RuntimeArtifact,
    _distribution,
    download_artifact,
    select_distribution,
)


def _registry(url: str = "https://github.test/model.onnx") -> dict:
    digest = hashlib.sha256(b"model").hexdigest()
    return {
        "schema": 1,
        "runtime_contract": 1,
        "models": {
            "v1.0": {
                "runtime": {"default_voice": "af_heart", "voices": ["af_heart", "af_alloy"]},
                "distributions": [
                    {
                        "id": "github-v1",
                        "provider": "github-release",
                        "transport": "https",
                        "runtime_ready": True,
                        "artifacts": [
                            {"id": "model", "role": "model", "quality": "fp32", "url": url, "local_name": "model.onnx", "format": "onnx", "size": 5, "sha256": digest},
                            {"id": "voices", "role": "voices", "url": "https://github.test/voices.npz", "local_name": "voices.npz", "format": "numpy-npz", "size": 5, "sha256": digest},
                        ],
                    },
                    {
                        "id": "hf-v1",
                        "provider": "huggingface",
                        "transport": "https",
                        "runtime_ready": True,
                        "repository": "source/repo",
                        "revision": "commit",
                        "artifacts": [
                            {"id": "model-hf", "role": "model", "quality": "fp32", "url": "https://hf.test/model", "local_name": "model.onnx", "format": "onnx", "size": 5, "sha256": digest},
                            {"id": "voices-hf", "role": "voices", "url": "https://hf.test/voices", "local_name": "voices.npz", "format": "numpy-npz", "size": 5, "sha256": digest},
                        ],
                    },
                ],
            }
        },
    }


class Response:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.read_once = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size: int = -1):
        if size < 0:
            return self.payload
        if self.read_once:
            return b""
        self.read_once = True
        return self.payload


def test_registry_fetches_and_caches_valid_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(_registry()).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(payload))
    cache = tmp_path / "models.json"

    registry = RegistryClient(url="https://registry.test/models.json", cache_path=cache).load()

    assert registry.model("v1.0").distribution().provider == "github-release"
    assert cache.is_file()


def test_registry_uses_last_valid_cache_after_bad_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "models.json"
    cache.write_text(json.dumps(_registry()), encoding="utf-8")
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(b"invalid"))

    registry = RegistryClient(cache_path=cache).load()

    assert registry.source == str(cache)
    assert cache.read_text(encoding="utf-8") == json.dumps(_registry())


def test_registry_rejects_invalid_schema_without_cache(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"schema": 2}), encoding="utf-8")

    with pytest.raises(ModelRegistryError, match="Unsupported"):
        RegistryClient(path=path).load()


def test_distribution_preference_and_quality_are_registry_authoritative() -> None:
    data = _registry()["models"]["v1.0"]["distributions"]

    parsed = tuple(_distribution(item) for item in data)
    assert select_distribution(parsed, "auto").provider == "github-release"
    assert select_distribution(parsed, "huggingface").provider == "huggingface"
    assert parsed[0].artifact("model", quality="fp32").local_name == "model.onnx"
    with pytest.raises(ModelRegistryError, match="no model"):
        parsed[0].artifact("model", quality="q8f16")


def test_failed_artifact_download_does_not_replace_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "model.onnx"
    target.write_bytes(b"known-good")
    artifact = RuntimeArtifact("model", "model", "onnx", "https://example.test/model", "model.onnx", 5, hashlib.sha256(b"model").hexdigest())
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(b"wrong"))

    with pytest.raises(ModelRegistryError, match="mismatch"):
        download_artifact(artifact, target)
    assert target.read_bytes() == b"known-good"
