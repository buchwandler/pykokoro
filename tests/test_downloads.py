"""Tests for download helpers."""

from __future__ import annotations

import urllib.request
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typing_extensions import Self

import pykokoro.onnx_backend as backend
from pykokoro.exceptions import ConfigurationError
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


def test_download_resumes_partial_file_with_http_range(tmp_path, monkeypatch):
    destination = tmp_path / "model.onnx"
    part_path = destination.with_suffix(destination.suffix + ".part")
    part_path.write_bytes(b"prefix-")
    seen_range: list[str] = []

    class RangeResponse(FakeResponse):
        status = 206
        headers = {"Content-Range": "bytes 7-12/13"}

    def fake_urlopen(request, timeout=None):
        seen_range.append(request.headers["Range"])
        return RangeResponse(b"suffix")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _download_from_github(
        "https://example.com/model.onnx",
        destination,
        min_size=1,
        retries=1,
        lock_timeout=1,
    )

    assert seen_range == ["bytes=7-"]
    assert destination.read_bytes() == b"prefix-suffix"
    assert not part_path.exists()


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

    monkeypatch.setattr(backend, "_hf_hub_download", fake_hf_hub_download)

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
    assert assets.config is None
    assert assets.model == tmp_path / "models" / "github" / "v1.0" / "kokoro-v1.0.onnx"
    assert assets.voices == tmp_path / "voices" / "github" / "v1.0" / "voices-v1.0.bin"
    assert not (tmp_path / "models" / "github" / "v1.0").exists()


def test_github_v1_model_asset_completeness_requires_model_and_voices(tmp_path, monkeypatch):
    monkeypatch.setattr("pykokoro.model_assets.get_user_cache_path", lambda: tmp_path)
    assets = get_model_asset_paths(quality="fp32", source="github", variant="v1.0")
    assert assets.missing == ("model", "voices")
    assert not assets.complete
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


def test_huggingface_model_asset_completeness_requires_config(tmp_path, monkeypatch):
    monkeypatch.setattr("pykokoro.model_assets.get_user_cache_path", lambda: tmp_path)
    assets = get_model_asset_paths(quality="fp32", source="huggingface", variant="v1.0")
    assert assets.config is not None
    assert assets.missing == ("config", "model", "voices")
    assert not assets.complete
    for path in (assets.config, assets.model, assets.voices):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")
    assert assets.missing == ()
    assert assets.complete
    assert are_models_downloaded("fp32", "huggingface", "v1.0")


def test_martin_github_downloads_use_exact_urls_and_checksums(tmp_path, monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_download(url, local_path, *args, **kwargs):
        calls.append((url, kwargs))
        return local_path

    monkeypatch.setattr(backend, "_download_from_github", fake_download)
    monkeypatch.setattr(backend, "_validate_onnx_file", lambda path: None)
    monkeypatch.setattr(backend, "get_model_dir", lambda source, variant: tmp_path / "models")
    monkeypatch.setattr(
        backend,
        "get_voices_archive_path",
        lambda source, variant: tmp_path / "voices" / "nested" / "voices.bin",
    )

    backend.download_model_github(variant="v1.2-de-martin", quality="fp32")
    backend.download_voices_github(variant="v1.2-de-martin")

    assert calls[0][0].endswith("model-files-german-martin-v1.2/kokoro-german-martin-v1.2.onnx")
    assert calls[0][1]["expected_sha256"].startswith("c302f1d8")
    assert calls[1][0].endswith("model-files-german-martin-v1.2/voices-german-martin-v1.2.bin")
    assert calls[1][1]["expected_sha256"].startswith("5b9c8553")
    assert (tmp_path / "voices" / "nested").is_dir()


def test_martin_github_downloads_forward_offline_and_exact_sizes(tmp_path, monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_download(url, local_path, *args, **kwargs):
        calls.append(kwargs)
        return local_path

    monkeypatch.setattr(backend, "_download_from_github", fake_download)
    monkeypatch.setattr(backend, "_validate_onnx_file", lambda path: None)
    monkeypatch.setattr(backend, "get_model_dir", lambda source, variant: tmp_path / "models")
    monkeypatch.setattr(
        backend, "get_voices_archive_path", lambda source, variant: tmp_path / "voices.bin"
    )

    backend.download_model_github("v1.2-de-martin", "fp32", offline=True)
    backend.download_voices_github("v1.2-de-martin", offline=True)

    assert calls[0]["offline"] is True
    assert calls[0]["expected_size"] == 325_512_630
    assert calls[1]["offline"] is True
    assert calls[1]["expected_size"] == 522_506


def test_ensure_models_revalidates_managed_nonempty_paths(tmp_path, monkeypatch):
    managed_model = tmp_path / "managed.onnx"
    managed_voices = tmp_path / "managed.bin"
    managed_model.write_bytes(b"stale model")
    managed_voices.write_bytes(b"stale voices")
    calls: list[str] = []

    kokoro = object.__new__(backend.Kokoro)
    kokoro._model_path = tmp_path / "cached.onnx"
    kokoro._voices_path = tmp_path / "cached.bin"
    kokoro._model_path.write_bytes(b"non-empty but untrusted")
    kokoro._voices_path.write_bytes(b"non-empty but untrusted")
    kokoro._model_path_provided = False
    kokoro._voices_path_provided = False
    kokoro._model_source = "github"
    kokoro._model_variant = "v1.2-de-martin"
    kokoro._model_quality = "fp32"

    def download_model(**kwargs):
        calls.append("model")
        return managed_model

    def download_voices(**kwargs):
        calls.append("voices")
        return managed_voices

    monkeypatch.setattr(backend, "download_model_github", download_model)
    monkeypatch.setattr(backend, "download_voices_github", download_voices)
    kokoro._ensure_models()

    assert calls == ["model", "voices"]
    assert kokoro._model_path == managed_model
    assert kokoro._voices_path == managed_voices


def test_ensure_models_huggingface_stores_archive_file(tmp_path, monkeypatch):
    managed_model = tmp_path / "managed.onnx"
    voices_directory = tmp_path / "voices" / "huggingface" / "v1.0"
    combined_archive = voices_directory / "voices.bin.npz"
    managed_model.write_bytes(b"model")
    combined_archive.parent.mkdir(parents=True)
    combined_archive.write_bytes(b"voices")

    kokoro = object.__new__(backend.Kokoro)
    kokoro._model_path = tmp_path / "cached.onnx"
    kokoro._voices_path = tmp_path / "cached.bin"
    kokoro._model_path_provided = False
    kokoro._voices_path_provided = False
    kokoro._model_source = "huggingface"
    kokoro._model_variant = "v1.0"
    kokoro._model_quality = "fp32"

    monkeypatch.setattr(backend, "download_model", lambda **kwargs: managed_model)

    def fake_download_all_voices(**kwargs):
        return voices_directory

    monkeypatch.setattr(backend, "download_all_voices", fake_download_all_voices)
    monkeypatch.setattr(
        backend,
        "get_voices_archive_path",
        lambda source, variant: combined_archive,
    )
    monkeypatch.setattr(backend, "is_config_downloaded", lambda variant: True)

    kokoro._ensure_models()

    assert kokoro._voices_path == combined_archive
    assert kokoro._voices_path.is_file()
    assert kokoro._voices_path != voices_directory


def test_redownload_huggingface_voices_stores_archive_file(tmp_path, monkeypatch):
    voices_directory = tmp_path / "voices" / "huggingface" / "v1.0"
    combined_archive = voices_directory / "voices.bin.npz"
    combined_archive.parent.mkdir(parents=True)
    combined_archive.write_bytes(b"voices")

    kokoro = object.__new__(backend.Kokoro)
    kokoro._model_source = "huggingface"
    kokoro._model_variant = "v1.0"

    monkeypatch.setattr(backend, "download_all_voices", lambda **kwargs: voices_directory)
    monkeypatch.setattr(
        backend,
        "get_voices_archive_path",
        lambda source, variant: combined_archive,
    )

    kokoro._redownload_voices(force=True)

    assert kokoro._voices_path == combined_archive
    assert kokoro._voices_path.is_file()
    assert kokoro._voices_path != voices_directory


def test_github_v1_model_setup_does_not_download_config_or_call_hf_client(tmp_path, monkeypatch):
    managed_model = tmp_path / "model.onnx"
    managed_voices = tmp_path / "voices.bin"
    managed_model.write_bytes(b"model")
    managed_voices.write_bytes(b"voices")

    kokoro = object.__new__(backend.Kokoro)
    kokoro._model_path = tmp_path / "cached.onnx"
    kokoro._voices_path = tmp_path / "cached.bin"
    kokoro._model_path_provided = False
    kokoro._voices_path_provided = False
    kokoro._model_source = "github"
    kokoro._model_variant = "v1.0"
    kokoro._model_quality = "fp32"

    monkeypatch.setattr(backend, "download_model_github", lambda **kwargs: managed_model)
    monkeypatch.setattr(backend, "download_voices_github", lambda **kwargs: managed_voices)
    monkeypatch.setattr(
        backend,
        "download_config",
        lambda **kwargs: pytest.fail("GitHub v1.0 must not download config.json"),
    )
    monkeypatch.setattr(
        backend,
        "_hf_hub_download",
        lambda **kwargs: pytest.fail("GitHub-only setup must not call Hugging Face"),
    )

    kokoro._ensure_models()

    assert kokoro._model_path == managed_model
    assert kokoro._voices_path == managed_voices


def test_github_v1_uses_embedded_vocabulary(tmp_path, monkeypatch):
    kokoro = object.__new__(backend.Kokoro)
    kokoro._model_source = "github"
    kokoro._model_variant = "v1.0"
    monkeypatch.setattr(
        backend,
        "load_vocab_from_config",
        lambda variant: pytest.fail("GitHub v1.0 must use embedded vocabulary"),
    )

    vocabulary = kokoro._get_vocabulary()

    assert vocabulary


def test_ensure_models_does_not_download_for_missing_custom_paths(tmp_path, monkeypatch):
    kokoro = object.__new__(backend.Kokoro)
    kokoro._model_path = tmp_path / "missing.onnx"
    kokoro._voices_path = tmp_path / "missing.bin"
    kokoro._model_path_provided = True
    kokoro._voices_path_provided = True
    kokoro._model_source = "github"
    kokoro._model_variant = "v1.2-de-martin"
    kokoro._model_quality = "fp32"
    monkeypatch.setattr(
        backend,
        "download_model_github",
        lambda **kwargs: pytest.fail("custom model path must not download"),
    )

    with pytest.raises(ConfigurationError, match="Explicit model_path"):
        kokoro._ensure_models()


def test_model_asset_queries_do_not_leak_between_source_variant_or_quality(tmp_path, monkeypatch):
    monkeypatch.setattr("pykokoro.model_assets.get_user_cache_path", lambda: tmp_path)
    github = get_model_asset_paths(quality="fp32", source="github", variant="v1.0")
    for path in (github.model, github.voices):
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
