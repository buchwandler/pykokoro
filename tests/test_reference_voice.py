from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pykokoro import ReferenceVoice
from pykokoro.exceptions import InvalidVoiceError


def make_voice(
    *,
    style: np.ndarray | None = None,
    memory: np.ndarray | None = None,
    memory_mask: np.ndarray | None = None,
    model_id: str = "en-akinvox-cloning-v1",
    model_fingerprint: str = "model-sha256",
    name: str | None = None,
    metadata: dict[str, str] | None = None,
) -> ReferenceVoice:
    return ReferenceVoice(
        style=np.zeros((1, 256), dtype=np.float32) if style is None else style,
        memory=np.zeros((1, 3, 192), dtype=np.float32) if memory is None else memory,
        memory_mask=np.ones((1, 3), dtype=np.bool_) if memory_mask is None else memory_mask,
        model_id=model_id,
        model_fingerprint=model_fingerprint,
        name=name,
        metadata={} if metadata is None else metadata,
    )


def test_reference_voice_round_trip_uses_safe_numpy_archive(tmp_path: Path) -> None:
    voice = make_voice(
        name="speaker",
        metadata={
            "reference_audio_sha256": "a" * 64,
            "reference_text_sha256": "b" * 64,
        },
    )
    path = tmp_path / "speaker.npz"

    voice.save(path)
    restored = ReferenceVoice.load(path)

    assert restored.fingerprint == voice.fingerprint
    assert restored.model_id == voice.model_id
    assert restored.model_fingerprint == voice.model_fingerprint
    assert restored.name == "speaker"
    assert dict(restored.metadata) == dict(voice.metadata)
    np.testing.assert_array_equal(restored.style, voice.style)
    np.testing.assert_array_equal(restored.memory, voice.memory)
    np.testing.assert_array_equal(restored.memory_mask, voice.memory_mask)

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"style", "memory", "memory_mask", "metadata_json_utf8"}
        assert all(archive[name].dtype != object for name in archive.files)
        metadata = json.loads(archive["metadata_json_utf8"].tobytes())
    assert "reference_audio_sha256" in metadata
    assert "reference_text_sha256" in metadata
    assert "reference_text" not in metadata
    assert "audio" not in metadata


def test_reference_voice_arrays_are_copied_and_read_only() -> None:
    style = np.zeros((1, 256), dtype=np.float32)
    voice = make_voice(style=style)
    style[0, 0] = 1

    assert voice.style[0, 0] == 0
    with pytest.raises(ValueError):
        voice.style[0, 0] = 2
    with pytest.raises(ValueError):
        voice.style.setflags(write=True)
    with pytest.raises(TypeError):
        voice.metadata["reference_audio_sha256"] = "a" * 64  # type: ignore[index]


def test_reference_voice_fingerprint_covers_conditioning_and_model_metadata() -> None:
    base = make_voice()
    changed_memory = np.zeros((1, 3, 192), dtype=np.float32)
    changed_memory[0, 0, 0] = 1

    assert make_voice(memory=changed_memory).fingerprint != base.fingerprint
    assert make_voice(model_fingerprint="different-model").fingerprint != base.fingerprint
    assert make_voice(name="named").fingerprint != base.fingerprint


def test_reference_voice_rejects_invalid_shapes_and_empty_memory() -> None:
    with pytest.raises(InvalidVoiceError, match="style"):
        make_voice(style=np.zeros((256,), dtype=np.float32))
    with pytest.raises(InvalidVoiceError, match="memory must have shape"):
        make_voice(memory=np.zeros((1, 3, 191), dtype=np.float32))
    with pytest.raises(InvalidVoiceError, match="must not be empty"):
        make_voice(
            memory=np.zeros((1, 0, 192), dtype=np.float32),
            memory_mask=np.zeros((1, 0), dtype=np.bool_),
        )
    with pytest.raises(InvalidVoiceError, match="memory_mask"):
        make_voice(memory_mask=np.ones((1, 2), dtype=np.bool_))


def test_reference_voice_rejects_object_and_non_finite_tensors() -> None:
    with pytest.raises(InvalidVoiceError, match="numeric tensor"):
        make_voice(style=np.full((1, 256), object(), dtype=object))
    with pytest.raises(InvalidVoiceError, match="finite"):
        make_voice(memory=np.full((1, 3, 192), np.nan, dtype=np.float32))


def test_reference_voice_rejects_invalid_model_metadata_and_hashes() -> None:
    with pytest.raises(InvalidVoiceError, match="model_id"):
        make_voice(model_id="")
    with pytest.raises(InvalidVoiceError, match="model_fingerprint"):
        make_voice(model_fingerprint=" ")
    with pytest.raises(InvalidVoiceError, match="SHA-256"):
        make_voice(metadata={"reference_audio_sha256": "not-a-digest"})
    with pytest.raises(InvalidVoiceError, match="unsupported"):
        make_voice(metadata={"reference_text": "private transcript"})


def test_reference_voice_load_rejects_corrupt_or_missing_metadata(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not an npz archive")
    with pytest.raises(InvalidVoiceError, match="could not load"):
        ReferenceVoice.load(corrupt)

    missing = tmp_path / "missing.npz"
    np.savez(missing, style=np.zeros((1, 256), dtype=np.float32))
    with pytest.raises(InvalidVoiceError, match="missing or unexpected"):
        ReferenceVoice.load(missing)


def test_reference_voice_load_rejects_tampered_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "state.npz"
    make_voice().save(path)
    with np.load(path, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    metadata = json.loads(fields["metadata_json_utf8"].tobytes())
    metadata["fingerprint"] = "0" * 64
    fields["metadata_json_utf8"] = np.frombuffer(
        json.dumps(metadata).encode("utf-8"), dtype=np.uint8
    )
    np.savez(path, **fields)

    with pytest.raises(InvalidVoiceError, match="fingerprint does not match"):
        ReferenceVoice.load(path)
