"""Portable, static Kokoro voice packs."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .exceptions import InvalidVoiceError

_FORMAT = "pykokoro-voicepack-v1"
_SCHEMA = 1
_VOICE_SHAPE = (510, 1, 256)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_metadata(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(metadata, Mapping):
        raise InvalidVoiceError("voice pack metadata must be a mapping")
    try:
        encoded = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False)
        canonical = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise InvalidVoiceError("voice pack metadata must contain JSON-compatible values") from exc
    if not isinstance(canonical, dict):
        raise InvalidVoiceError("voice pack metadata must be a JSON object")
    return canonical, _freeze_json(canonical)


def _copy_voice_data(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value)
    if data.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise InvalidVoiceError("voice pack data must have float16 or float32 dtype")
    if data.shape != _VOICE_SHAPE:
        raise InvalidVoiceError(f"voice pack data must have shape {_VOICE_SHAPE}")
    if not np.isfinite(data).all():
        raise InvalidVoiceError("voice pack data must contain only finite values")
    contiguous = np.array(data, order="C", copy=True)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        _VOICE_SHAPE
    )


def _fingerprint(
    data: np.ndarray, name: str | None, engine: str | None, metadata: Mapping[str, Any]
) -> str:
    digest = hashlib.sha256()
    identity = {"name": name, "engine": engine, "metadata": _thaw_json(metadata)}
    digest.update(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    digest.update(data.dtype.str.encode("ascii"))
    digest.update(json.dumps(data.shape, separators=(",", ":")).encode("ascii"))
    digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class KokoroVoicePack:
    """A normal static Kokoro voice style with safe, portable persistence."""

    data: np.ndarray
    name: str | None = None
    engine: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise InvalidVoiceError("voice pack name must be non-empty when supplied")
        if self.engine is not None and (
            not isinstance(self.engine, str) or not self.engine.strip()
        ):
            raise InvalidVoiceError("voice pack engine must be non-empty when supplied")
        data = _copy_voice_data(self.data)
        canonical, metadata = _canonical_metadata(self.metadata)
        del canonical
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self, "fingerprint", _fingerprint(data, self.name, self.engine, metadata)
        )

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        name: str | None = None,
        engine: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> KokoroVoicePack:
        """Create a pack from a compatible array, canonicalizing it to float32."""
        value = np.asarray(array)
        if (
            value.dtype.hasobject
            or not np.issubdtype(value.dtype, np.number)
            or np.issubdtype(value.dtype, np.complexfloating)
        ):
            raise InvalidVoiceError("voice pack data must be a real numeric tensor")
        if value.shape != _VOICE_SHAPE:
            raise InvalidVoiceError(f"voice pack data must have shape {_VOICE_SHAPE}")
        if not np.isfinite(value).all():
            raise InvalidVoiceError("voice pack data must contain only finite values")
        with np.errstate(over="ignore", invalid="ignore"):
            canonical = np.asarray(value, dtype=np.float32)
        if not np.isfinite(canonical).all():
            raise InvalidVoiceError(
                "voice pack data must be representable as finite float32 values"
            )
        return cls(
            data=canonical,
            name=name,
            engine=engine,
            metadata={} if metadata is None else metadata,
        )

    def save(self, path: str | Path) -> None:
        """Save the pack as a pickle-free NumPy archive."""
        metadata: dict[str, Any] = {
            "schema": _SCHEMA,
            "format": _FORMAT,
            "base_model": self.metadata.get("base_model", "v1.0"),
            "fingerprint": self.fingerprint,
            "metadata": _thaw_json(self.metadata),
        }
        if self.name is not None:
            metadata["name"] = self.name
        if self.engine is not None:
            metadata["engine"] = self.engine
        metadata_bytes = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        with Path(path).open("wb") as file:
            np.savez_compressed(
                file,
                data=self.data,
                metadata_json_utf8=np.frombuffer(metadata_bytes, dtype=np.uint8),
            )

    @classmethod
    def load(cls, path: str | Path) -> KokoroVoicePack:
        """Load and validate a pickle-free NumPy archive."""
        try:
            with np.load(Path(path), allow_pickle=False) as archive:
                expected_fields = {"data", "metadata_json_utf8"}
                if set(archive.files) != expected_fields:
                    raise InvalidVoiceError("voice pack archive has missing or unexpected fields")
                data = archive["data"]
                metadata_array = archive["metadata_json_utf8"]
                if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
                    raise InvalidVoiceError("voice pack metadata must be a UTF-8 byte array")
                document = json.loads(metadata_array.tobytes().decode("utf-8"))
        except InvalidVoiceError:
            raise
        except (
            OSError,
            EOFError,
            zipfile.BadZipFile,
            ValueError,
            TypeError,
            UnicodeDecodeError,
        ) as exc:
            raise InvalidVoiceError(f"could not load voice pack from {path!s}: {exc}") from exc

        if not isinstance(document, dict):
            raise InvalidVoiceError("voice pack metadata must be a JSON object")
        required = {"schema", "format", "base_model", "fingerprint", "metadata"}
        allowed = required | {"name", "engine"}
        if not required <= document.keys() or not document.keys() <= allowed:
            raise InvalidVoiceError("voice pack metadata has missing or unsupported fields")
        if (
            type(document["schema"]) is not int
            or document["schema"] != _SCHEMA
            or document["format"] != _FORMAT
        ):
            raise InvalidVoiceError("unsupported voice pack archive schema or format")
        if not isinstance(document["base_model"], str) or not isinstance(
            document["fingerprint"], str
        ):
            raise InvalidVoiceError(
                "voice pack base_model and fingerprint metadata must be strings"
            )
        name = document.get("name")
        engine = document.get("engine")
        if name is not None and not isinstance(name, str):
            raise InvalidVoiceError("voice pack name metadata must be a string")
        if engine is not None and not isinstance(engine, str):
            raise InvalidVoiceError("voice pack engine metadata must be a string")
        metadata = document["metadata"]
        if (
            not isinstance(metadata, dict)
            or metadata.get("base_model", "v1.0") != document["base_model"]
        ):
            raise InvalidVoiceError("voice pack base_model metadata does not match")
        pack = cls(data=data, name=name, engine=engine, metadata=metadata)
        if pack.fingerprint != document["fingerprint"]:
            raise InvalidVoiceError("voice pack fingerprint does not match its data")
        return pack
