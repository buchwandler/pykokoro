from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pykokoro.exceptions import InvalidVoiceError
from pykokoro.voice_pack import KokoroVoicePack


def make_data(dtype: np.dtype | type = np.float32) -> np.ndarray:
    return np.zeros((510, 1, 256), dtype=dtype)


def test_voice_pack_copies_read_only_data_and_fingerprints_metadata() -> None:
    data = make_data()
    pack = KokoroVoicePack(data, name="speaker", engine="inno-v0.2", metadata={"version": 2})
    data[0, 0, 0] = 1

    assert pack.data[0, 0, 0] == 0
    assert pack.data.dtype == np.float32
    with pytest.raises(ValueError):
        pack.data[0, 0, 0] = 2
    with pytest.raises(TypeError):
        pack.metadata["new"] = "value"  # type: ignore[index]
    assert (
        pack.fingerprint
        != KokoroVoicePack(
            pack.data, name="speaker", engine="inno-v0.2", metadata={"version": 3}
        ).fingerprint
    )


def test_voice_pack_accepts_float16_and_from_array_canonicalizes_to_float32() -> None:
    pack = KokoroVoicePack(make_data(np.dtype(np.float16)))
    generated = KokoroVoicePack.from_array(make_data(np.dtype(np.float16)), engine="inno-v0.2")

    assert pack.data.dtype == np.float16
    assert generated.data.dtype == np.float32


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.zeros((1, 256), dtype=np.float32), "shape"),
        (np.zeros((510, 1, 256), dtype=np.float64), "float16 or float32"),
        (np.full((510, 1, 256), np.nan, dtype=np.float32), "finite"),
        (np.zeros((510, 1, 256), dtype=object), "float16 or float32"),
    ],
)
def test_voice_pack_rejects_invalid_data(data: np.ndarray, message: str) -> None:
    with pytest.raises(InvalidVoiceError, match=message):
        KokoroVoicePack(data)


def test_voice_pack_from_array_rejects_non_numeric_and_non_finite_data() -> None:
    with pytest.raises(InvalidVoiceError, match="real numeric"):
        KokoroVoicePack.from_array(np.full((510, 1, 256), "voice"))
    with pytest.raises(InvalidVoiceError, match="finite"):
        KokoroVoicePack.from_array(np.full((510, 1, 256), np.inf))


def test_voice_pack_round_trip_uses_safe_numpy_archive(tmp_path: Path) -> None:
    pack = KokoroVoicePack.from_array(
        make_data(),
        name="speaker",
        engine="inno-v0.2",
        metadata={"base_model": "v1.0", "runtime": {"version": "0.2"}},
    )
    path = tmp_path / "speaker.npz"
    pack.save(path)
    restored = KokoroVoicePack.load(path)

    assert restored.fingerprint == pack.fingerprint
    assert restored.name == "speaker"
    assert restored.engine == "inno-v0.2"
    assert restored.metadata["runtime"]["version"] == "0.2"
    np.testing.assert_array_equal(restored.data, pack.data)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"data", "metadata_json_utf8"}
        assert all(archive[key].dtype != object for key in archive.files)
        metadata = json.loads(archive["metadata_json_utf8"].tobytes())
    assert metadata["format"] == "pykokoro-voicepack-v1"
    assert metadata["base_model"] == "v1.0"


def test_voice_pack_load_rejects_corrupt_or_incomplete_archives(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not an npz archive")
    with pytest.raises(InvalidVoiceError, match="could not load"):
        KokoroVoicePack.load(corrupt)

    incomplete = tmp_path / "incomplete.npz"
    np.savez(incomplete, data=make_data())
    with pytest.raises(InvalidVoiceError, match="missing or unexpected"):
        KokoroVoicePack.load(incomplete)


def test_voice_pack_load_rejects_tampered_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "pack.npz"
    KokoroVoicePack(make_data()).save(path)
    with np.load(path, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    metadata = json.loads(fields["metadata_json_utf8"].tobytes())
    metadata["fingerprint"] = "0" * 64
    fields["metadata_json_utf8"] = np.frombuffer(
        json.dumps(metadata).encode("utf-8"), dtype=np.uint8
    )
    np.savez(path, **fields)

    with pytest.raises(InvalidVoiceError, match="fingerprint does not match"):
        KokoroVoicePack.load(path)
