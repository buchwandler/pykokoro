"""Portable, model-bound state for ONNX reference voice synthesis."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .exceptions import InvalidVoiceError

_FORMAT = "pykokoro-reference-voice-v1"
_SCHEMA = 1
_STYLE_SHAPE = (1, 256)
_MEMORY_WIDTH = 192
_METADATA_HASH_FIELDS = frozenset({"reference_audio_sha256", "reference_text_sha256"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_metadata(
    model_id: str,
    model_fingerprint: str,
    name: str | None,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": _SCHEMA,
        "format": _FORMAT,
        "model_id": model_id,
        "model_fingerprint": model_fingerprint,
    }
    if name is not None:
        result["name"] = name
    result.update(metadata)
    return result


def _fingerprint(
    style: np.ndarray,
    memory: np.ndarray,
    memory_mask: np.ndarray,
    metadata: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, tensor in (("style", style), ("memory", memory), ("memory_mask", memory_mask)):
        digest.update(name.encode("ascii"))
        digest.update(tensor.dtype.str.encode("ascii"))
        digest.update(json.dumps(tensor.shape, separators=(",", ":")).encode("ascii"))
        digest.update(tensor.tobytes(order="C"))
    return digest.hexdigest()


def _copy_float_tensor(value: np.ndarray, name: str) -> np.ndarray:
    tensor = np.asarray(value)
    if (
        tensor.dtype.hasobject
        or not np.issubdtype(tensor.dtype, np.number)
        or np.issubdtype(tensor.dtype, np.complexfloating)
    ):
        raise InvalidVoiceError(f"reference voice {name} must be a real numeric tensor")
    tensor = np.array(tensor, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(tensor).all():
        raise InvalidVoiceError(f"reference voice {name} must contain only finite values")
    return np.frombuffer(tensor.tobytes(), dtype=np.float32).reshape(tensor.shape)


@dataclass(frozen=True, slots=True)
class ReferenceVoice:
    """Immutable, reusable conditioning state created from reference speech.

    Arrays are copied into canonical dtypes and marked read-only. The fingerprint
    identifies both the model binding and all conditioning tensors.
    """

    style: np.ndarray
    memory: np.ndarray
    memory_mask: np.ndarray
    model_id: str
    model_fingerprint: str
    name: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise InvalidVoiceError("reference voice model_id must be non-empty")
        if not isinstance(self.model_fingerprint, str) or not self.model_fingerprint.strip():
            raise InvalidVoiceError("reference voice model_fingerprint must be non-empty")
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise InvalidVoiceError("reference voice name must be non-empty when supplied")
        if not isinstance(self.metadata, Mapping):
            raise InvalidVoiceError("reference voice metadata must be a mapping")

        metadata: dict[str, str] = {}
        for key, value in self.metadata.items():
            if key not in _METADATA_HASH_FIELDS:
                raise InvalidVoiceError(f"unsupported reference voice metadata field: {key!r}")
            if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
                raise InvalidVoiceError(f"reference voice {key} must be a lowercase SHA-256 digest")
            metadata[key] = value

        style = _copy_float_tensor(self.style, "style")
        memory = _copy_float_tensor(self.memory, "memory")
        mask = np.asarray(self.memory_mask)
        if mask.dtype != np.bool_:
            raise InvalidVoiceError("reference voice memory_mask must have boolean dtype")
        mask = np.array(mask, dtype=np.bool_, order="C", copy=True)
        if style.shape != _STYLE_SHAPE:
            raise InvalidVoiceError(f"reference voice style must have shape {_STYLE_SHAPE}")
        if memory.ndim != 3 or memory.shape[0] != 1 or memory.shape[2] != _MEMORY_WIDTH:
            raise InvalidVoiceError("reference voice memory must have shape [1, M, 192]")
        if memory.shape[1] == 0:
            raise InvalidVoiceError("reference voice memory must not be empty")
        if mask.shape != memory.shape[:2]:
            raise InvalidVoiceError(
                "reference voice memory_mask must match the memory batch and length"
            )
        mask = np.frombuffer(mask.tobytes(), dtype=np.bool_).reshape(mask.shape)

        metadata_proxy = MappingProxyType(metadata)
        identity = _canonical_metadata(
            self.model_id, self.model_fingerprint, self.name, metadata_proxy
        )
        object.__setattr__(self, "style", style)
        object.__setattr__(self, "memory", memory)
        object.__setattr__(self, "memory_mask", mask)
        object.__setattr__(self, "metadata", metadata_proxy)
        object.__setattr__(self, "fingerprint", _fingerprint(style, memory, mask, identity))

    def save(self, path: str | Path) -> None:
        """Save the state as a pickle-free NumPy archive."""
        metadata = _canonical_metadata(
            self.model_id, self.model_fingerprint, self.name, self.metadata
        )
        metadata["fingerprint"] = self.fingerprint
        metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with Path(path).open("wb") as file:
            np.savez_compressed(
                file,
                style=self.style,
                memory=self.memory,
                memory_mask=self.memory_mask,
                metadata_json_utf8=np.frombuffer(metadata_bytes, dtype=np.uint8),
            )

    @classmethod
    def load(cls, path: str | Path) -> ReferenceVoice:
        """Load and validate a pickle-free NumPy archive."""
        try:
            with np.load(Path(path), allow_pickle=False) as archive:
                expected_fields = {"style", "memory", "memory_mask", "metadata_json_utf8"}
                if set(archive.files) != expected_fields:
                    raise InvalidVoiceError(
                        "reference voice archive has missing or unexpected fields"
                    )
                style = archive["style"]
                memory = archive["memory"]
                memory_mask = archive["memory_mask"]
                metadata_array = archive["metadata_json_utf8"]
                if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
                    raise InvalidVoiceError("reference voice metadata must be a UTF-8 byte array")
                data = json.loads(metadata_array.tobytes().decode("utf-8"))
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
            raise InvalidVoiceError(f"could not load reference voice from {path!s}: {exc}") from exc

        if not isinstance(data, dict):
            raise InvalidVoiceError("reference voice metadata must be a JSON object")
        required = {"schema", "format", "model_id", "model_fingerprint", "fingerprint"}
        allowed = required | {"name"} | _METADATA_HASH_FIELDS
        if not required <= data.keys() or not data.keys() <= allowed:
            raise InvalidVoiceError("reference voice metadata has missing or unsupported fields")
        if (
            type(data["schema"]) is not int
            or data["schema"] != _SCHEMA
            or data["format"] != _FORMAT
        ):
            raise InvalidVoiceError("unsupported reference voice archive schema or format")
        if not isinstance(data["fingerprint"], str):
            raise InvalidVoiceError("reference voice fingerprint metadata must be a string")
        name = data.get("name")
        if name is not None and not isinstance(name, str):
            raise InvalidVoiceError("reference voice name metadata must be a string")
        state_metadata = {key: data[key] for key in _METADATA_HASH_FIELDS if key in data}
        voice = cls(
            style=style,
            memory=memory,
            memory_mask=memory_mask,
            model_id=data["model_id"],
            model_fingerprint=data["model_fingerprint"],
            name=name,
            metadata=state_metadata,
        )
        if voice.fingerprint != data["fingerprint"]:
            raise InvalidVoiceError("reference voice fingerprint does not match its state")
        return voice
