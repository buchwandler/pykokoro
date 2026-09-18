from __future__ import annotations

import hashlib
import json
import os
import time
from io import StringIO
from pathlib import Path

import pytest

from pykokoro.asset_progress import AssetProgressEvent, ConsoleAssetProgress
from pykokoro.model_registry import (
    ArtifactIntegrityError,
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
                            {
                                "id": "model",
                                "role": "model",
                                "quality": "fp32",
                                "url": url,
                                "local_name": "model.onnx",
                                "format": "onnx",
                                "size": 5,
                                "sha256": digest,
                            },
                            {
                                "id": "voices",
                                "role": "voices",
                                "url": "https://github.test/voices.npz",
                                "local_name": "voices.npz",
                                "format": "numpy-npz",
                                "size": 5,
                                "sha256": digest,
                            },
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
                            {
                                "id": "model-hf",
                                "role": "model",
                                "quality": "fp32",
                                "url": "https://hf.test/model",
                                "local_name": "model.onnx",
                                "format": "onnx",
                                "size": 5,
                                "sha256": digest,
                            },
                            {
                                "id": "voices-hf",
                                "role": "voices",
                                "url": "https://hf.test/voices",
                                "local_name": "voices.npz",
                                "format": "numpy-npz",
                                "size": 5,
                                "sha256": digest,
                            },
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


def test_registry_fetches_and_caches_valid_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.dumps(_registry()).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(payload))
    cache = tmp_path / "models.json"

    registry = RegistryClient(url="https://registry.test/models.json", cache_path=cache).load()

    assert registry.model("v1.0").distribution().provider == "github-release"
    assert cache.is_file()
    assert registry.cache_fallback is False


def test_registry_uses_fresh_cache_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "models.json"
    cache.write_text(json.dumps(_registry()), encoding="utf-8")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: pytest.fail("fresh cache unexpectedly used the network"),
    )

    registry = RegistryClient(cache_path=cache).load()

    assert registry.source == str(cache)
    assert registry.cache_fallback is False


def test_registry_refreshes_stale_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "models.json"
    cache.write_text(json.dumps(_registry()), encoding="utf-8")
    stale_time = time.time() - 2 * 24 * 60 * 60
    os.utime(cache, (stale_time, stale_time))
    payload = json.dumps(_registry()).encode()
    calls = 0

    def open_url(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    RegistryClient(cache_path=cache).load()

    assert calls == 1


def test_registry_offline_uses_cache_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "models.json"
    cache.write_text(json.dumps(_registry()), encoding="utf-8")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: pytest.fail("offline registry unexpectedly used the network"),
    )

    registry = RegistryClient(cache_path=cache).load(offline=True)

    assert registry.source == str(cache)
    assert registry.cache_fallback is False


def test_registry_uses_last_valid_cache_after_bad_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cache = tmp_path / "models.json"
    cache.write_text(json.dumps(_registry()), encoding="utf-8")
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(b"invalid"))

    registry = RegistryClient(cache_path=cache).load(max_cache_age_s=0)

    assert registry.cache_fallback is True
    assert registry.source == str(cache)
    assert "using cached registry" in caplog.text
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


def test_failed_artifact_download_does_not_replace_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.onnx"
    target.write_bytes(b"known-good")
    artifact = RuntimeArtifact(
        "model",
        "model",
        "onnx",
        "https://example.test/model",
        "model.onnx",
        5,
        hashlib.sha256(b"model").hexdigest(),
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(b"wrong"))

    with pytest.raises(ModelRegistryError, match="mismatch"):
        download_artifact(artifact, target)
    assert target.read_bytes() == b"known-good"


def test_registry_forced_refresh_does_not_fall_back_to_stale_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "models.json"
    cache.write_text(json.dumps(_registry()), encoding="utf-8")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    with pytest.raises(ModelRegistryError, match="fresh model registry"):
        RegistryClient(cache_path=cache).load(refresh=True, allow_cache_fallback=False)


def test_registry_forced_refresh_requests_cache_revalidation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requests = []
    payload = json.dumps(_registry()).encode()

    def open_url(request, **kwargs):
        requests.append(request)
        return Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    registry = RegistryClient(
        url="https://registry.test/models.json?existing=1", cache_path=tmp_path / "models.json"
    ).load(refresh=True, allow_cache_fallback=False)

    assert registry.cache_fallback is False
    headers = {key.lower(): value for key, value in requests[0].header_items()}
    assert headers["cache-control"] == "no-cache"
    assert headers["pragma"] == "no-cache"
    assert "existing=1" in requests[0].full_url
    assert "pykokoro_refresh=" in requests[0].full_url


def test_registry_rejects_offline_refresh_combination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="offline and refresh cannot be combined"):
        RegistryClient(cache_path=tmp_path / "models.json").load(offline=True, refresh=True)


def test_verify_artifact_reports_expected_and_actual_size(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"new")
    artifact = RuntimeArtifact(
        "artifact",
        "model",
        "bin",
        "https://example.test/artifact",
        "artifact.bin",
        5,
        hashlib.sha256(b"old!!").hexdigest(),
    )

    with pytest.raises(ArtifactIntegrityError) as exc_info:
        from pykokoro.model_registry import verify_artifact

        verify_artifact(path, artifact)

    assert exc_info.value.expected == 5
    assert exc_info.value.actual == 3
    assert "expected 5" in str(exc_info.value)
    assert "got 3" in str(exc_info.value)


def test_verify_artifact_detects_same_size_sha_change(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"new!!")
    artifact = RuntimeArtifact(
        "artifact",
        "model",
        "bin",
        "https://example.test/artifact",
        "artifact.bin",
        5,
        hashlib.sha256(b"old!!").hexdigest(),
    )

    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch") as exc_info:
        from pykokoro.model_registry import verify_artifact

        verify_artifact(path, artifact)

    assert exc_info.value.expected == hashlib.sha256(b"old!!").hexdigest()
    assert exc_info.value.actual == hashlib.sha256(b"new!!").hexdigest()


def test_download_artifact_reports_byte_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"model-payload"
    artifact = RuntimeArtifact(
        "model",
        "model",
        "onnx",
        "https://example.test/model",
        "model.onnx",
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(payload))
    events: list[tuple[int, int]] = []
    target = tmp_path / "model.onnx"

    result = download_artifact(
        artifact,
        target,
        progress_callback=lambda _artifact, done, total: events.append((done, total)),
    )

    assert result == target
    assert events
    assert [done for done, _total in events] == sorted(done for done, _total in events)
    assert all(total == artifact.size for _done, total in events)
    assert events[-1][0] == artifact.size
    assert target.read_bytes() == payload
    assert list(tmp_path.glob("tmp*")) == []


def test_console_asset_progress_reports_lifecycle() -> None:
    stream = StringIO()
    reporter = ConsoleAssetProgress(stream)
    base = {
        "model_id": "v1.0",
        "distribution_id": "github-v1",
        "artifact_id": "model",
        "role": "model",
        "filename": "model.onnx",
        "bytes_total": 13 * 1024 * 1024,
        "target": "/cache/model.onnx",
    }
    reporter(AssetProgressEvent(phase="download-start", bytes_done=0, **base))
    reporter(
        AssetProgressEvent(
            phase="download-progress",
            bytes_done=base["bytes_total"] // 2,
            **base,
        )
    )
    reporter(
        AssetProgressEvent(
            phase="verify-start",
            bytes_done=base["bytes_total"],
            **base,
        )
    )
    reporter(
        AssetProgressEvent(
            phase="verify-complete",
            bytes_done=base["bytes_total"],
            **base,
        )
    )
    reporter(
        AssetProgressEvent(
            phase="install-complete",
            bytes_done=base["bytes_total"],
            artifact_id="",
            filename="",
            role="artifact",
            distribution_id=base["distribution_id"],
            model_id=base["model_id"],
            bytes_total=None,
            target="/cache/onnxvoice/kokoro/v1.0",
        )
    )

    text = stream.getvalue()
    assert "Downloading model:" in text
    assert "13.0 MiB" in text
    assert "Verifying model:" in text
    assert "Runtime assets ready:" in text
    assert "/cache/onnxvoice/kokoro/v1.0" in text

def test_console_asset_progress_reports_cache_unknown_size_and_install_state() -> None:
    cached_stream = StringIO()
    cached = ConsoleAssetProgress(cached_stream)
    cached(
        AssetProgressEvent(
            phase="cache-hit",
            model_id="v1.0",
            distribution_id="github-v1",
            artifact_id="voices",
            role="voices",
            filename="voices.npz",
            bytes_done=0,
            bytes_total=123,
            target="/cache/voices.npz",
        )
    )
    assert cached_stream.getvalue() == "Using cached voices: voices.npz\n"

    unknown_stream = StringIO()
    unknown = ConsoleAssetProgress(unknown_stream)
    unknown(
        AssetProgressEvent(
            phase="download-start",
            model_id="v1.0",
            distribution_id="github-v1",
            artifact_id="model",
            role="model",
            filename="model.onnx",
            bytes_done=0,
            bytes_total=None,
            target="/cache/model.onnx",
        )
    )
    unknown(
        AssetProgressEvent(
            phase="download-progress",
            model_id="v1.0",
            distribution_id="github-v1",
            artifact_id="model",
            role="model",
            filename="model.onnx",
            bytes_done=2 * 1024 * 1024,
            bytes_total=None,
            target="/cache/model.onnx",
        )
    )
    unknown_text = unknown_stream.getvalue()
    assert "model.onnx" in unknown_text
    assert "size unknown" in unknown_text
    assert "0 B" not in unknown_text

    installed_stream = StringIO()
    installed = ConsoleAssetProgress(installed_stream)
    installed(
        AssetProgressEvent(
            phase="install-complete",
            model_id="v1.0",
            distribution_id="github-v1",
            artifact_id="",
            role="artifact",
            filename="",
            bytes_done=0,
            bytes_total=None,
            target="/cache/onnxvoice/kokoro/v1.0",
            message="already installed",
        )
    )
    installed_text = installed_stream.getvalue()
    assert "Runtime assets already installed:" in installed_text
    assert "/cache/onnxvoice/kokoro/v1.0" in installed_text
    assert "Downloading" not in installed_text
