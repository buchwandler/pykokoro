"""Deterministic download and disk-cache regression coverage."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import pykokoro.onnx_backend as backend
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.runtime.cache import DiskCache, make_g2p_key
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import Segment, Trace


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            payload, self.payload = self.payload, b""
            return payload
        payload, self.payload = self.payload[:size], self.payload[size:]
        return payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_existing_invalid_github_cache_is_replaced(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"bad")
    monkeypatch.setattr(
        backend.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _Response(b"good"),
    )

    result = backend._download_from_github(
        "https://example.com/artifact.bin",
        destination,
        min_size=4,
        retries=1,
        lock_timeout=1,
    )

    assert result == destination
    assert destination.read_bytes() == b"good"


def test_existing_valid_cache_skips_download_and_honors_checksum(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "artifact.bin"
    payload = b"valid artifact"
    destination.write_bytes(payload)
    monkeypatch.setattr(
        backend.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("valid cache should not download"),
    )

    result = backend._download_from_github(
        "https://example.com/artifact.bin",
        destination,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        retries=1,
        lock_timeout=1,
    )

    assert result == destination


def test_validator_failure_retries_and_then_succeeds(tmp_path, monkeypatch) -> None:
    calls = 0

    def fake_urlopen(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(b"bad" if calls == 1 else b"good")

    def validator(path: Path) -> None:
        if path.read_bytes() != b"good":
            raise backend.ArtifactValidationError("temporary invalid payload")

    monkeypatch.setattr(backend.urllib.request, "urlopen", fake_urlopen)
    destination = backend._download_from_github(
        "https://example.com/artifact.bin",
        tmp_path / "artifact.bin",
        validator=validator,
        retries=2,
        lock_timeout=1,
    )

    assert destination.read_bytes() == b"good"
    assert calls == 2


def test_stale_dead_download_lock_is_recovered(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "artifact.bin"
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    lock_path.write_text(
        json.dumps({"pid": 99_999_999, "created_at": time.time() - 60}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        backend.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _Response(b"ok"),
    )

    backend._download_from_github(
        "https://example.com/artifact.bin",
        destination,
        min_size=2,
        retries=1,
        lock_timeout=0.01,
    )

    assert destination.read_bytes() == b"ok"
    assert not lock_path.exists()


def test_stale_download_lock_handles_windows_missing_process_error(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / "artifact.bin.lock"
    lock_path.write_text(
        json.dumps({"pid": 99_999_999, "created_at": time.time() - 60}),
        encoding="utf-8",
    )

    def fake_kill(pid: int, signal: int) -> None:
        error = OSError("missing process")
        error.winerror = 87
        raise error

    monkeypatch.setattr(backend.os, "kill", fake_kill)

    assert backend._is_stale_download_lock(lock_path, timeout=1)


def test_offline_mode_uses_valid_cache_and_rejects_missing_cache(tmp_path) -> None:
    destination = tmp_path / "cached.bin"
    destination.write_bytes(b"cached")

    assert (
        backend._download_from_github("https://example.com/cached.bin", destination, offline=True)
        == destination
    )

    with pytest.raises(RuntimeError, match="Offline mode"):
        backend._download_from_github(
            "https://example.com/missing.bin", tmp_path / "missing.bin", offline=True
        )


def test_g2p_cache_schema_recomputes_and_preserves_warnings(tmp_path, monkeypatch) -> None:
    segment = Segment(
        id="segment-0",
        text="hello",
        char_start=0,
        char_end=5,
        paragraph_idx=0,
        sentence_idx=0,
        clause_idx=0,
    )
    cfg = PipelineConfig(cache_dir=str(tmp_path))
    key = make_g2p_key(
        text=segment.text,
        lang=cfg.generation.lang,
        is_phonemes=False,
        tokenizer_config=None,
        phoneme_override=None,
        kokorog2p_version="test",
        model_quality=cfg.model_quality,
        model_source=cfg.model_source,
        model_variant=cfg.model_variant,
    )
    DiskCache(tmp_path).set(key, ["obsolete payload"])

    class FakeG2PModule:
        __version__ = "test"

        @staticmethod
        def get_g2p(**kwargs):
            return object()

        @staticmethod
        def phonemize(*args, **kwargs):
            return SimpleNamespace(phonemes="həˈloʊ", ids=[1, 2], warnings=["fallback"])

        @staticmethod
        def ids_to_phonemes(tokens, model):
            return "həˈloʊ"

    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())
    doc = DocumentResult(clean_text=segment.text, segments=[segment])
    first_trace = Trace()
    adapter.phonemize([segment], doc, cfg, first_trace)
    second_trace = Trace()
    adapter.phonemize([segment], doc, cfg, second_trace)

    assert first_trace.warnings == ["fallback"]
    assert second_trace.warnings == ["fallback"]
    cached = DiskCache(tmp_path).get(key)
    assert cached == {
        "schema": 3,
        "phonemes": "həˈloʊ",
        "tokens": [1, 2],
        "alignment_tokens": [],
        "warnings": ["fallback"],
    }
