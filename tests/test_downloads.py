"""Tests for download helpers."""

from __future__ import annotations

import urllib.request
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typing_extensions import Self

import pykokoro.onnx_backend as backend
from pykokoro.model_assets import (
    are_models_downloaded,
    are_voices_downloaded,
    get_model_asset_paths,
    get_voices_archive_path,
)
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

    def __enter__(self) -> Self:
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
    monkeypatch.setattr(backend, "get_user_cache_path", lambda folder=None: tmp_path / folder)

    model_path = backend.get_model_path(quality="fp32", source="huggingface", variant="v1.0")

    assert model_path == (
        tmp_path / "models" / "huggingface" / "v1.0" / "onnx" / "model-timestamped.onnx"
    )


def test_hf_v1_download_ignores_old_non_timestamped_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "get_user_cache_path", lambda folder=None: tmp_path / folder)
    monkeypatch.setattr(backend, "_validate_onnx_file", lambda path: None)
    monkeypatch.setattr(
        backend,
        "hf_model_spec",
        lambda variant, filename: SimpleNamespace(revision="test", sha256=None),
    )

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


@pytest.mark.parametrize(
    ("source", "variant", "filename"),
    [
        ("huggingface", "v1.0", "voices.bin.npz"),
        ("huggingface", "v1.1-zh", "voices.bin.npz"),
        ("github", "v1.0", "voices-v1.0.bin"),
        ("github", "v1.1-zh", "voices-v1.1-zh.bin"),
        ("github", "v1.1-de", "voices-german-v1.1.bin"),
    ],
)
def test_voice_archive_paths_are_source_and_variant_aware(
    tmp_path, monkeypatch, source, variant, filename
):
    monkeypatch.setattr("pykokoro.model_assets.get_user_cache_path", lambda: tmp_path)
    assert get_voices_archive_path(source, variant) == (
        tmp_path / "voices" / source / variant / filename
    )


def test_model_asset_paths_are_source_variant_and_quality_aware(tmp_path, monkeypatch):
    monkeypatch.setattr("pykokoro.model_assets.get_user_cache_path", lambda: tmp_path)
    assets = get_model_asset_paths(quality="fp32", source="github", variant="v1.0")
    assert assets.config == tmp_path / "config" / "v1.0" / "config.json"
    assert assets.model == tmp_path / "models" / "github" / "v1.0" / "kokoro-v1.0.onnx"
    assert assets.voices == tmp_path / "voices" / "github" / "v1.0" / "voices-v1.0.bin"
    assert not (tmp_path / "models" / "github" / "v1.0").exists()


def test_model_asset_completeness_requires_three_nonempty_regular_files(tmp_path, monkeypatch):
    monkeypatch.setattr("pykokoro.model_assets.get_user_cache_path", lambda: tmp_path)
    assets = get_model_asset_paths(quality="fp32", source="github", variant="v1.0")
    assert assets.missing == ("config", "model", "voices")
    assert not assets.complete

    assets.config.parent.mkdir(parents=True)
    assets.config.write_bytes(b"config")
    assets.model.parent.mkdir(parents=True)
    assets.model.write_bytes(b"model")
    assets.voices.parent.mkdir(parents=True)
    assets.voices.write_bytes(b"voices")
    assert assets.missing == ()
    assert assets.complete
    assert are_models_downloaded("fp32", "github", "v1.0")
    assert are_voices_downloaded("github", "v1.0")

    assets.model.write_bytes(b"")
    assert assets.missing == ("model",)
    assert not assets.complete


def test_model_asset_queries_do_not_leak_between_source_variant_or_quality(tmp_path, monkeypatch):
    monkeypatch.setattr("pykokoro.model_assets.get_user_cache_path", lambda: tmp_path)
    github = get_model_asset_paths(quality="fp32", source="github", variant="v1.0")
    for path in (github.config, github.model, github.voices):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")

    assert are_models_downloaded("fp32", "github", "v1.0")
    assert not are_models_downloaded("fp32", "huggingface", "v1.0")
    assert not are_models_downloaded("fp32", "github", "v1.1-zh")
    assert not are_models_downloaded("q8", "github", "v1.0")


def test_kokoro_forwards_model_quality_to_session_manager(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    voices_path = tmp_path / "voices.bin"
    kokoro = backend.Kokoro(
        model_path=model_path,
        voices_path=voices_path,
        model_quality="q8",
        model_source="github",
        model_variant="v1.0",
    )
    kokoro._tokenizer = MagicMock()
    kokoro._ensure_models = lambda: None

    session_manager = MagicMock()
    session_manager.create_session.return_value = MagicMock()
    manager_kwargs = {}

    def make_session_manager(**kwargs):
        manager_kwargs.update(kwargs)
        return session_manager

    monkeypatch.setattr(backend, "OnnxSessionManager", make_session_manager)
    voice_manager = MagicMock()
    monkeypatch.setattr(backend, "VoiceManager", lambda **kwargs: voice_manager)
    monkeypatch.setattr(backend, "AudioGenerator", MagicMock())

    kokoro._init_kokoro()

    assert session_manager.create_session.called
    assert manager_kwargs["model_quality"] == "q8"
