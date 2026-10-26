"""Public local-G2Lex reference and metadata inspection regressions."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import g2lex
import pytest

import pykokoro
from pykokoro.g2lex_sources import G2LexFileInspection, G2LexFileRef, inspect_g2lex_file
from pykokoro.tokenizer import TokenizerConfig


def _write_g2lex(tmp_path: Path) -> Path:
    source = tmp_path / "lexicon.json"
    output = tmp_path / "local.g2lex"
    source.write_text(json.dumps({"hello": "hɛˈloʊ"}), encoding="utf-8")
    g2lex.pack_file(
        source,
        output,
        input_format="kokoro-json",
        metadata={
            "id": "fixture:local",
            "name": "local",
            "language": "en-us",
            "phoneme_encoding": "ipa",
        },
    )
    return output


def test_inspection_pins_file_and_logical_hashes(tmp_path: Path) -> None:
    path = _write_g2lex(tmp_path)
    inspection = pykokoro.inspect_g2lex_file(path)

    assert isinstance(inspection, G2LexFileInspection)
    assert inspection.path == str(path.resolve())
    assert inspection.sha256.startswith("sha256:")
    assert inspection.logical_sha256 is not None
    assert inspection.metadata["language"] == "en-us"
    assert inspection.to_ref() == G2LexFileRef(
        path=path.resolve(),
        sha256=inspection.sha256,
        logical_sha256=inspection.logical_sha256,
    )
    assert pykokoro.G2LexFileRef is G2LexFileRef
    with pytest.raises(TypeError):
        inspection.metadata["language"] = "de-de"  # type: ignore[index]


def test_inspection_reads_metadata_without_iterating_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "metadata-only.g2lex"
    path.write_bytes(b"container bytes")

    class MetadataHandle:
        metadata = {"language": "en-us", "phoneme_encoding": "ipa"}

        def __iter__(self):
            pytest.fail("inspection must not iterate pronunciation entries")

        def close(self) -> None:
            pass

    monkeypatch.setattr("pykokoro.g2lex_sources.g2lex.open", lambda _path: MetadataHandle())
    result = inspect_g2lex_file(path)
    assert result.metadata["phoneme_encoding"] == "ipa"


def test_file_ref_is_immutable_pinned_and_config_normalizes_to_tuple() -> None:
    ref = G2LexFileRef("lexicons/local.g2lex", sha256="A" * 64)
    config = TokenizerConfig(g2lex_files=ref, lexicons=("gold", "lexhint"))

    assert ref.sha256 == f"sha256:{'a' * 64}"
    assert config.g2lex_files == (ref,)
    assert config.lexicons == ("gold", "lexhint")
    with pytest.raises(FrozenInstanceError):
        ref.path = "changed.g2lex"  # type: ignore[misc]


def test_file_ref_rejects_invalid_pins_duplicates_and_other_backends() -> None:
    with pytest.raises(ValueError, match="64-character"):
        G2LexFileRef("local.g2lex", sha256="not-a-hash")
    ref = G2LexFileRef("local.g2lex")
    with pytest.raises(ValueError, match="duplicate"):
        TokenizerConfig(g2lex_files=(ref, ref))
    with pytest.raises(TypeError, match="G2LexFileRef"):
        TokenizerConfig(g2lex_files=("local.g2lex",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kokorog2p backend"):
        TokenizerConfig(backend="espeak", g2lex_files=(ref,))
