"""Smoke tests for prosody comparison examples."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def test_comparison_script_writes_complete_strict_artifacts(tmp_path: Path) -> None:
    sample_rate = 24000
    time = np.arange(12000, dtype=np.float32) / sample_rate
    source = (0.2 * np.sin(2 * np.pi * 180 * time)).astype(np.float32)
    input_wav = tmp_path / "input.wav"
    output_dir = tmp_path / "comparison"
    sf.write(input_wav, source, sample_rate)

    subprocess.run(
        [
            sys.executable,
            "examples/compare_prosody_algorithms.py",
            "--input-wav",
            str(input_wav),
            "--output-dir",
            str(output_dir),
            "--blind-seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    methods = {"phase_vocoder", "wsola", "esola", "td_psola"}
    presets = {"rate_slow", "rate_fast", "pitch_down", "pitch_up", "emphasis", "combined"}
    rendered = [path for path in output_dir.glob("*/*.wav") if path.parent.name != "blind"]
    assert len(rendered) == len(methods) * len(presets)
    assert (output_dir / "reference.wav").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "blind_key.csv").exists()
    assert (output_dir / "blind" / "LISTENING_INSTRUCTIONS.txt").exists()

    with (output_dir / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert {record["method"] for record in metrics} == methods
    assert {record["preset"] for record in metrics} == presets

    with (output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == len(rendered)
    with (output_dir / "blind_key.csv").open(newline="", encoding="utf-8") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == len(rendered)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["blind_seed"] == 7
    assert "Objective metrics are diagnostic only" in (
        output_dir / "blind" / "LISTENING_INSTRUCTIONS.txt"
    ).read_text(encoding="utf-8")
