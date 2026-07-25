"""Tests for download helpers."""

from __future__ import annotations

import urllib.request

import pytest

import pykokoro.onnx_backend as backend
from pykokoro.onnx_backend import _download_from_github


class FakeResponse:
    def __init__(self, data: bytes, fail_after: int | None = None):
        self._data = data
        self._offset = 0
        self._fail_after = fail_after

    def read(self, size: int = -1) -> bytes:
        if self._fail_after is not None and self._offset >= self._fail_after:
            raise TimeoutError("timeout")
        if size == -1:
            chunk = self._data[self._offset :]
        else:
            chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_download_streaming_success(tmp_path, monkeypatch):
    payload = b"ok" * 1024

    def fake_urlopen(url, timeout=None):
        return FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    destination = tmp_path / "model.onnx"
    result = _download_from_github(
        "https://example.com/model.onnx",
        destination,
        min_size=1,
        retries=1,
        lock_timeout=1,
    )

    assert result == destination
    assert destination.read_bytes() == payload


def test_download_validation_failure(tmp_path, monkeypatch):
    payload = b"short"

    def fake_urlopen(url, timeout=None):
        return FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    destination = tmp_path / "model.onnx"
    with pytest.raises(RuntimeError, match="too small"):
        _download_from_github(
            "https://example.com/model.onnx",
            destination,
            min_size=1024,
            retries=1,
            lock_timeout=1,
        )

    assert not destination.exists()


def test_download_retries_on_timeout(tmp_path, monkeypatch):
    payload = b"ok" * 256
    calls = {"count": 0}

    def fake_urlopen(url, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("timeout")
        return FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    destination = tmp_path / "model.onnx"
    result = _download_from_github(
        "https://example.com/model.onnx",
        destination,
        min_size=1,
        retries=2,
        lock_timeout=1,
    )

    assert result.exists()
    assert calls["count"] == 2


def test_download_lock_timeout(tmp_path, monkeypatch):
    destination = tmp_path / "model.onnx"
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    lock_path.write_text("locked")

    def fake_urlopen(url, timeout=None):
        return FakeResponse(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="download lock"):
        _download_from_github(
            "https://example.com/model.onnx",
            destination,
            min_size=1,
            retries=1,
            lock_timeout=0.01,
        )


def test_hf_v1_model_cache_path_uses_timestamped_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(
        backend, "get_user_cache_path", lambda folder=None: tmp_path / folder
    )

    model_path = backend.get_model_path(
        quality="fp32", source="huggingface", variant="v1.0"
    )

    assert model_path == (
        tmp_path / "models" / "huggingface" / "v1.0" / "onnx" / "model-timestamped.onnx"
    )


def test_hf_v1_download_ignores_old_non_timestamped_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        backend, "get_user_cache_path", lambda folder=None: tmp_path / folder
    )
    monkeypatch.setattr(backend, "_validate_onnx_file", lambda path: None)

    old_path = tmp_path / "models" / "huggingface" / "v1.0" / "onnx" / "model.onnx"
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(b"old non timestamped model")

    hub_path = tmp_path / "hub" / "model.onnx"
    hub_path.parent.mkdir()
    hub_path.write_bytes(b"new timestamped model")

    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        assert kwargs["filename"] == "model.onnx"
        assert kwargs["subfolder"] == "onnx"
        assert kwargs["local_dir"] is None
        return str(hub_path)

    monkeypatch.setattr(backend, "hf_hub_download", fake_hf_hub_download)

    result = backend.download_model(variant="v1.0", quality="fp32")

    assert result == old_path.with_name("model-timestamped.onnx")
    assert result.read_bytes() == b"new timestamped model"
    assert old_path.read_bytes() == b"old non timestamped model"
    assert len(calls) == 1
