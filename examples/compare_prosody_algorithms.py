#!/usr/bin/env python3
"""Create a reproducible listening comparison for PyKokoro prosody backends."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

try:
    from ._output import artifact_dir
except ImportError:
    from _output import artifact_dir

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig, ProsodyConfig
from pykokoro.prosody import apply_prosody

METHODS = (
    "phase_vocoder",
    "wsola",
    "esola",
    "td_psola",
)


@dataclass(frozen=True)
class Preset:
    name: str
    rate: float
    semitones: float
    gain_db: float


PRESETS = (
    Preset("rate_slow", 0.85, 0.0, 0.0),
    Preset("rate_fast", 1.20, 0.0, 0.0),
    Preset("pitch_down", 1.0, -2.0, 0.0),
    Preset("pitch_up", 1.0, 2.0, 0.0),
    Preset("emphasis", 0.87, 1.2, 2.0),
    Preset("combined", 1.15, -1.5, 0.0),
)


def _rate_string(rate: float) -> str:
    return f"{rate * 100:g}%"


def _pitch_string(semitones: float) -> str:
    return f"{semitones:+g}st"


def _gain_string(gain_db: float) -> str:
    return f"{gain_db:+g}dB"


def _mono_float32(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio)
    if values.ndim == 2:
        # soundfile uses frames x channels.
        values = np.mean(values, axis=1)
    if values.ndim != 1:
        raise ValueError(f"expected mono or frames-by-channels audio, got {values.shape}")
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("input audio contains non-finite values")
    return values


def _synthesize_reference(
    text: str,
    *,
    voice: str,
    lang: str,
) -> tuple[np.ndarray, int]:
    config = PipelineConfig(
        voice=voice,
        generation=GenerationConfig(lang=lang, speed=1.0),
    )
    pipeline = KokoroPipeline(config)
    try:
        result = pipeline.run(text)
    finally:
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()
    return _mono_float32(result.audio), int(result.sample_rate)


def _load_reference(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False)
    return _mono_float32(audio), int(sample_rate)


def _metrics(
    audio: np.ndarray,
    *,
    sample_rate: int,
    expected_samples: int,
    runtime_seconds: float,
) -> dict[str, object]:
    values = np.asarray(audio, dtype=np.float64)
    differences = np.diff(values) if values.size > 1 else np.empty(0)
    clipped = int(np.count_nonzero(np.abs(values) >= 1.0))
    duration = values.size / sample_rate

    return {
        "samples": int(values.size),
        "expected_samples": expected_samples,
        "length_error": int(values.size - expected_samples),
        "duration_seconds": duration,
        "runtime_seconds": runtime_seconds,
        "real_time_factor": runtime_seconds / duration if duration > 0.0 else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
        "peak": float(np.max(np.abs(values))) if values.size else 0.0,
        "clipped_samples": clipped,
        "max_adjacent_jump": float(np.max(np.abs(differences))) if differences.size else 0.0,
        "rms_adjacent_jump": float(np.sqrt(np.mean(np.square(differences))))
        if differences.size
        else 0.0,
        "finite": bool(np.isfinite(values).all()),
    }


def _render(
    reference: np.ndarray,
    sample_rate: int,
    output_dir: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for preset in PRESETS:
        preset_dir = output_dir / preset.name
        preset_dir.mkdir(parents=True, exist_ok=True)
        expected_samples = max(1, round(reference.size / preset.rate))

        for method in METHODS:
            config = ProsodyConfig(
                method=method,
                fallback_methods=(),
                strict=True,
                clip=False,
            )

            started = perf_counter()
            rendered = apply_prosody(
                reference,
                sample_rate,
                volume=_gain_string(preset.gain_db),
                pitch=_pitch_string(preset.semitones),
                rate=_rate_string(preset.rate),
                config=config,
            )
            elapsed = perf_counter() - started

            output_path = preset_dir / f"{method}.wav"
            sf.write(output_path, rendered, sample_rate, subtype="PCM_16")

            record = {
                "preset": preset.name,
                "method": method,
                "rate": preset.rate,
                "semitones": preset.semitones,
                "gain_db": preset.gain_db,
                "file": str(output_path.relative_to(output_dir)),
                **_metrics(
                    rendered,
                    sample_rate=sample_rate,
                    expected_samples=expected_samples,
                    runtime_seconds=elapsed,
                ),
            }
            records.append(record)
            print(
                f"{preset.name:>12} / {method:>14}: "
                f"{record['duration_seconds']:.3f}s, "
                f"RTF={record['real_time_factor']:.3f}"
            )

    return records


def _write_metrics(records: list[dict[str, object]], output_dir: Path) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for record in records for key in record))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_blind_set(
    records: list[dict[str, object]],
    output_dir: Path,
    *,
    seed: int,
) -> None:
    blind_dir = output_dir / "blind"
    blind_dir.mkdir(parents=True, exist_ok=True)

    randomized = list(records)
    random.Random(seed).shuffle(randomized)

    key_rows: list[dict[str, object]] = []
    for index, record in enumerate(randomized, start=1):
        blind_name = f"sample_{index:03d}.wav"
        shutil.copyfile(output_dir / str(record["file"]), blind_dir / blind_name)
        key_rows.append(
            {
                "blind_file": blind_name,
                "preset": record["preset"],
                "method": record["method"],
                "rate": record["rate"],
                "semitones": record["semitones"],
                "gain_db": record["gain_db"],
            }
        )

    with (output_dir / "blind_key.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)

    (blind_dir / "LISTENING_INSTRUCTIONS.txt").write_text(
        "\n".join(
            [
                "Rate each file without opening ../blind_key.csv first.",
                "Suggested 1-5 ratings:",
                "- naturalness",
                "- intelligibility",
                "- speaker/timbre preservation",
                "- robotic or metallic character",
                "- buzziness/flutter",
                "- transient duplication",
                "- clicks or boundary artifacts",
                "- overall preference",
                "",
                "Objective metrics are diagnostic only and do not rank",
                "perceived speech quality.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input-wav",
        type=Path,
        help="use an existing WAV as the common reference",
    )
    source.add_argument(
        "--text",
        default=(
            "The package must arrive today. A careful listener should compare timing, "
            "pitch, clarity, plosives, fricatives, and the identity of the speaker."
        ),
        help="text synthesized once when --input-wav is omitted",
    )
    parser.add_argument("--voice", default="af_bella")
    parser.add_argument(
        "--lang",
        default="en-us",
        help="Document language for the reference utterance.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=artifact_dir() / "prosody_algorithm_comparison",
    )
    parser.add_argument("--blind-seed", type=int, default=20260731)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_wav is not None:
        reference, sample_rate = _load_reference(args.input_wav)
        source_description = str(args.input_wav)
    else:
        reference, sample_rate = _synthesize_reference(args.text, voice=args.voice, lang=args.lang)
        source_description = f"PyKokoro voice={args.voice!r}"

    sf.write(
        args.output_dir / "reference.wav",
        reference,
        sample_rate,
        subtype="PCM_16",
    )

    records = _render(reference, sample_rate, args.output_dir)
    _write_metrics(records, args.output_dir)
    _write_blind_set(records, args.output_dir, seed=args.blind_seed)

    manifest = {
        "source": source_description,
        "sample_rate": sample_rate,
        "reference_samples": int(reference.size),
        "methods": list(METHODS),
        "presets": [asdict(preset) for preset in PRESETS],
        "blind_seed": args.blind_seed,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Reference: {args.output_dir / 'reference.wav'}")
    print(f"Metrics:   {args.output_dir / 'metrics.csv'}")
    print(f"Blind set: {args.output_dir / 'blind'}")
    print(f"Key:       {args.output_dir / 'blind_key.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
