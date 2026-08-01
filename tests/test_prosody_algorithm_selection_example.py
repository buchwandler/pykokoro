"""Regression tests for the prosody algorithm selection example."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from examples.prosody_algorithm_selection import _mono_float32, _write_verified_pcm16


def _speech_like(sample_rate: int, seconds: float) -> np.ndarray:
    count = round(sample_rate * seconds)
    time = np.arange(count, dtype=np.float64) / sample_rate
    envelope = 0.55 + 0.45 * np.square(np.sin(2.0 * np.pi * 2.2 * time))
    signal = envelope * (
        np.sin(2.0 * np.pi * 140.0 * time)
        + 0.35 * np.sin(2.0 * np.pi * 280.0 * time)
        + 0.15 * np.sin(2.0 * np.pi * 420.0 * time)
    )
    signal *= 0.25 / np.max(np.abs(signal))
    return signal.astype(np.float32)


def _run_example(input_wav: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "examples/prosody_algorithm_selection.py",
            "--input-wav",
            str(input_wav),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_algorithm_selection_uses_one_valid_reference(tmp_path: Path) -> None:
    sample_rate = 24_000
    source = _speech_like(sample_rate, 1.0)
    input_wav = tmp_path / "input.wav"
    output_dir = tmp_path / "output"
    sf.write(input_wav, source, sample_rate, subtype="PCM_16")

    completed = _run_example(input_wav, output_dir)

    assert "reference" in completed.stdout
    assert (output_dir / "reference.wav").exists()
    assert (output_dir / "metrics.json").exists()

    methods = ("wsola", "esola", "td_psola", "phase_vocoder")
    expected_frames = round(source.size / 0.87)

    for method in methods:
        path = output_dir / f"prosody_{method}.wav"
        info = sf.info(path)
        audio, decoded_rate = sf.read(path, dtype="float32", always_2d=False)

        assert info.format == "WAV"
        assert info.subtype == "PCM_16"
        assert info.channels == 1
        assert decoded_rate == sample_rate
        assert audio.shape == (expected_frames,)
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) <= 1.0
        assert np.sqrt(np.mean(np.square(audio, dtype=np.float64))) > 1e-4

    report = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    records = report["records"]
    reference = next(record for record in records if record["kind"] == "reference")
    processed = [record for record in records if record["kind"] == "processed"]

    assert len(processed) == len(methods)
    assert report["volume"] is None
    assert all(record["length_error"] == 0 for record in processed)
    assert all(record["finite"] for record in processed)
    assert all(record["samples_at_or_above_full_scale"] == 0 for record in records)
    assert all(
        record["source_sha256_float_bytes"] == reference["sha256_float_bytes"]
        for record in processed
    )


def test_input_wav_path_does_not_import_onnxruntime(tmp_path: Path) -> None:
    sample_rate = 24_000
    input_wav = tmp_path / "input.wav"
    output_dir = tmp_path / "output"
    sf.write(input_wav, _speech_like(sample_rate, 0.25), sample_rate, subtype="PCM_16")

    blocker = """
import builtins
import runpy
import sys

real_import = builtins.__import__

def blocked(name, *args, **kwargs):
    if name == "onnxruntime" or name.startswith("onnxruntime."):
        raise ModuleNotFoundError("blocked onnxruntime", name="onnxruntime")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked
sys.argv = [
    "examples/prosody_algorithm_selection.py",
    "--input-wav",
    sys.argv[1],
    "--output-dir",
    sys.argv[2],
]
runpy.run_path("examples/prosody_algorithm_selection.py", run_name="__main__")
"""
    completed = subprocess.run(
        [sys.executable, "-c", blocker, str(input_wav), str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Metrics:" in completed.stdout
    assert (output_dir / "reference.wav").exists()


def test_pcm16_writer_rejects_silent_hard_clipping(tmp_path: Path) -> None:
    audio = np.array([0.0, 1.01], dtype=np.float32)

    with pytest.raises(ValueError, match="exceeds full scale"):
        _write_verified_pcm16(tmp_path / "bad.wav", audio, 24_000)


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (np.array([], dtype=np.float32), "empty"),
        (np.array([np.nan], dtype=np.float32), "NaN or infinity"),
        (np.zeros((2, 2, 1), dtype=np.float32), "expected mono"),
    ],
)
def test_reference_validation_rejects_invalid_audio(audio: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _mono_float32(audio)
