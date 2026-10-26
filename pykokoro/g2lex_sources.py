"""Immutable references and metadata-only inspection for local G2Lex files."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import g2lex

if TYPE_CHECKING:
    from kokorog2p.lexicons.registry import FileLexiconSource

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _normalize_sha256(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a SHA-256 digest string")
    digest = value.strip().casefold().removeprefix("sha256:")
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return f"sha256:{digest}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class G2LexFileRef:
    """Pinned, immutable reference to a local G2Lex source file.

    Digests are optional for interactive use. Applications that persist a source
    should pin ``sha256`` (and may also pin ``logical_sha256``) from inspection.
    Local files are runtime inputs, not named Lexphon assets to be installed.
    """

    path: str | os.PathLike[str]
    sha256: str | None = None
    logical_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, os.PathLike)):
            raise TypeError("G2Lex file path must be a string or path-like value")
        path = os.fspath(self.path).strip()
        if not path:
            raise ValueError("G2Lex file path must not be empty")
        object.__setattr__(self, "path", str(Path(path).expanduser()))
        object.__setattr__(self, "sha256", _normalize_sha256(self.sha256, label="sha256"))
        object.__setattr__(
            self,
            "logical_sha256",
            _normalize_sha256(self.logical_sha256, label="logical_sha256"),
        )

    def to_source(self) -> FileLexiconSource:
        """Convert to the KokoroG2P runtime source type."""
        from kokorog2p.lexicons.registry import FileLexiconSource

        return FileLexiconSource(
            path=self.path,
            sha256=self.sha256,
            logical_sha256=self.logical_sha256,
        )


@dataclass(frozen=True, slots=True)
class G2LexFileInspection:
    """Read-only G2Lex manifest inspection with reproducible content pins."""

    path: str
    sha256: str
    logical_sha256: str | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_ref(self) -> G2LexFileRef:
        """Create a file reference pinned to this inspection's hashes."""
        return G2LexFileRef(
            path=self.path,
            sha256=self.sha256,
            logical_sha256=self.logical_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly inspection summary."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "logical_sha256": self.logical_sha256,
            "metadata": dict(self.metadata),
        }


def inspect_g2lex_file(path: str | os.PathLike[str]) -> G2LexFileInspection:
    """Inspect a local G2Lex manifest without iterating pronunciation entries.

    The container is opened through G2Lex's validated reader, and the content hash
    is streamed from disk. Only manifest metadata is returned; the lexical mapping
    is never traversed or materialized by this helper.
    """
    raw_path = Path(path).expanduser()
    try:
        resolved_path = raw_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"G2Lex file does not exist: {raw_path}") from exc
    if not resolved_path.is_file():
        raise ValueError(f"G2Lex path is not a file: {resolved_path}")

    try:
        handle = g2lex.open(resolved_path)
    except Exception as exc:
        raise ValueError(f"Invalid G2Lex file {resolved_path}: {exc}") from exc
    try:
        manifest = getattr(handle, "metadata", None)
        if not isinstance(manifest, Mapping):
            raise ValueError(f"G2Lex file {resolved_path} has no manifest metadata")
        metadata = dict(manifest)
    finally:
        close = getattr(handle, "close", None)
        if close is not None:
            close()

    language = metadata.get("language") or metadata.get("locale")
    if not isinstance(language, str) or not language.strip():
        raise ValueError(f"G2Lex file {resolved_path} has no language metadata")
    encoding = metadata.get("phoneme_encoding")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError(f"G2Lex file {resolved_path} has no phoneme_encoding metadata")

    logical_sha256 = metadata.get("logical_sha256")
    normalized_logical_sha256 = _normalize_sha256(
        str(logical_sha256) if logical_sha256 is not None else None,
        label="logical_sha256",
    )
    return G2LexFileInspection(
        path=str(resolved_path),
        sha256=_file_sha256(resolved_path),
        logical_sha256=normalized_logical_sha256,
        metadata=metadata,
    )


__all__ = ["G2LexFileInspection", "G2LexFileRef", "inspect_g2lex_file"]
