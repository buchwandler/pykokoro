from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any) -> None: ...
    def delete(self, key: str) -> None: ...


def make_cache_key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, bytes | bytearray):
            b = bytes(p)
        else:
            b = json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        h.update(b)
        h.update(b"\x1f")
    return h.hexdigest()


def cache_from_dir(cache_dir: str | None) -> Cache:
    if not cache_dir:
        return NullCache()
    return DiskCache(Path(cache_dir))


def make_g2p_key(
    *,
    text: str,
    lang: str,
    is_phonemes: bool,
    tokenizer_config: dict[str, Any] | None,
    phoneme_override: str | None,
    kokorog2p_version: str | None = None,
    model_quality: str | None = None,
    model_source: str | None = None,
    model_variant: str | None = None,
) -> str:
    return make_cache_key(
        {
            "text": text,
            "lang": lang,
            "is_phonemes": is_phonemes,
            "tokenizer_config": tokenizer_config,
            "phoneme_override": phoneme_override,
            "kokorog2p_version": kokorog2p_version,
            "model_quality": model_quality,
            "model_source": model_source,
            "model_variant": model_variant,
        }
    )


class NullCache:
    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any) -> None:
        return None

    def delete(self, key: str) -> None:
        return None


class DiskCache:
    _replace_retries = 6
    _replace_backoff = 0.01

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def _tmp_path(self, key: str) -> Path:
        token = secrets.token_hex(6)
        return self.root / f"{key}.tmp.{os.getpid()}.{token}"

    def _replace_atomic(self, src: Path, dst: Path) -> None:
        for attempt in range(1, self._replace_retries + 1):
            try:
                os.replace(src, dst)
                return
            except PermissionError:
                if attempt >= self._replace_retries:
                    raise
                time.sleep(self._replace_backoff * attempt)

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            with suppress(OSError):
                p.unlink()
            return None

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        tmp_path = self._tmp_path(key)
        payload = json.dumps(value, ensure_ascii=False)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            with self._write_lock:
                self._replace_atomic(tmp_path, p)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def delete(self, key: str) -> None:
        with suppress(FileNotFoundError):
            self._path(key).unlink()
