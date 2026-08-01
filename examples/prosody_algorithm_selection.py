#!/usr/bin/env python3
"""Compare PyKokoro prosody backends on one validated reference waveform.

The tool separates source synthesis, prosody processing, and WAV export so a
failure can be assigned to the correct stage. On Android/Termux, prefer
``--input-wav`` because the standard ONNX Runtime package reports Android as
an unsupported platform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import audiosig
import numpy as np
import soundfile as sf

import pykokoro
from pykokoro import ProsodyConfig
from pykokoro.prosody import apply_prosody, parse_rate

METHODS = ("wsola", "esola", "td_psola", "phase_vocoder")
DEFAULT_TEXT = (
    "The package must arrive today, without further delay. "
    "This longer reference avoids a one-word short-sentence extraction."
)


def _mono_float32(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio)

    if values.ndim == 2:
        # SoundFile convention is frames x channels.
        values = np.mean(values, axis=1)

    if values.ndim != 1:
        raise ValueError(
            f"expected mono or frames-by-channels audio, received shape {values.shape}"
        )

    values = values.astype(np.float32, copy=False)

    if values.size == 0:
        raise ValueError("audio is empty")
    if not np.isfinite(values).all():
        raise ValueError("audio contains NaN or infinity")

    return np.ascontiguousarray(values)


def _sha256(audio: np.ndarray) -> str:
    values = np.ascontiguousarray(audio)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _stats(
    audio: np.ndarray,
    *,
    sample_rate: int,
    expected_samples: int | None = None,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    values = np.asarray(audio, dtype=np.float64)
    differences = np.diff(values)
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(np.square(values))))

    result: dict[str, Any] = {
        "samples": int(values.size),
        "sample_rate": int(sample_rate),
        "duration_seconds": float(values.size / sample_rate),
        "dtype": str(audio.dtype),
        "finite": bool(np.isfinite(values).all()),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "peak": peak,
        "rms": rms,
        "dc_offset": float(np.mean(values)),
        "samples_at_or_above_full_scale": int(np.count_nonzero(np.abs(values) >= 1.0)),
        "max_adjacent_jump": (float(np.max(np.abs(differences))) if differences.size else 0.0),
        "rms_adjacent_jump": (
            float(np.sqrt(np.mean(np.square(differences)))) if differences.size else 0.0
        ),
        "sha256_float_bytes": _sha256(audio),
    }

    if expected_samples is not None:
        result["expected_samples"] = int(expected_samples)
        result["length_error"] = int(values.size - expected_samples)

    if runtime_seconds is not None:
        duration = float(values.size / sample_rate)
        result["runtime_seconds"] = float(runtime_seconds)
        result["real_time_factor"] = float(runtime_seconds / duration) if duration > 0.0 else 0.0

    return result


def _write_verified_pcm16(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    values = _mono_float32(audio)
    peak = float(np.max(np.abs(values)))

    # Do not let SoundFile silently hard-clip a comparison sample.
    if peak > 1.0:
        raise ValueError(
            f"{path.name}: peak {peak:.6f} exceeds full scale; "
            "remove positive gain, lower the source level, or explicitly "
            "add a normalization policy"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, values, sample_rate, subtype="PCM_16")

    info = sf.info(path)
    decoded, decoded_rate = sf.read(path, dtype="float32", always_2d=False)
    decoded = _mono_float32(decoded)

    if info.format != "WAV":
        raise RuntimeError(f"{path}: expected WAV, received {info.format}")
    if info.subtype != "PCM_16":
        raise RuntimeError(f"{path}: expected PCM_16, received {info.subtype}")
    if info.channels != 1:
        raise RuntimeError(f"{path}: expected mono, received {info.channels} channels")
    if decoded_rate != sample_rate:
        raise RuntimeError(f"{path}: sample-rate mismatch {decoded_rate} != {sample_rate}")
    if decoded.size != values.size:
        raise RuntimeError(f"{path}: frame-count mismatch {decoded.size} != {values.size}")
    if not np.isfinite(decoded).all():
        raise RuntimeError(f"{path}: decoded WAV contains non-finite samples")

    return {
        "wav_format": info.format,
        "wav_subtype": info.subtype,
        "wav_channels": int(info.channels),
        "wav_frames": int(info.frames),
        "wav_sample_rate": int(info.samplerate),
        "wav_bytes": int(path.stat().st_size),
        "decoded_peak": float(np.max(np.abs(decoded))),
        "decoded_rms": float(np.sqrt(np.mean(np.square(decoded, dtype=np.float64)))),
    }


def _load_reference(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return _mono_float32(audio), int(sample_rate)


def _synthesize_reference(*, text: str, voice: str) -> tuple[np.ndarray, int]:
    # Keep the input-WAV path free of ONNX imports.
    from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
    from pykokoro.short_sentence_handler import ShortSentenceConfig

    config = PipelineConfig(
        voice=voice,
        generation=GenerationConfig(speed=1.0, random_seed=0),
        # Algorithm comparison must not include the carrier-phrase cutter.
        short_sentence_config=ShortSentenceConfig(enabled=False),
    )

    with KokoroPipeline(config) as pipeline:
        result = pipeline.run(text)

    return _mono_float32(result.audio), int(result.sample_rate)


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "sys_platform": sys.platform,
        "android_root": os.environ.get("ANDROID_ROOT"),
        "android_data": os.environ.get("ANDROID_DATA"),
        "pykokoro_version": getattr(pykokoro, "__version__", "unknown"),
        "pykokoro_file": getattr(pykokoro, "__file__", "unknown"),
        "audiosig_version": getattr(audiosig, "__version__", "unknown"),
        "audiosig_file": getattr(audiosig, "__file__", "unknown"),
        "numpy_version": np.__version__,
        "soundfile_version": getattr(sf, "__version__", "unknown"),
        "libsndfile_version": getattr(sf, "__libsndfile_version__", "unknown"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input-wav", type=Path, help="use a known-good WAV and bypass ONNX synthesis"
    )
    source.add_argument(
        "--text", default=DEFAULT_TEXT, help="text synthesized once when --input-wav is omitted"
    )
    parser.add_argument("--voice", default="af_bella")
    parser.add_argument("--output-dir", type=Path, default=Path("prosody_algorithm_outputs"))
    parser.add_argument("--rate", default="87%")
    parser.add_argument("--pitch", default="+1.2st")
    parser.add_argument(
        "--volume",
        default=None,
        help="optional gain such as -3dB; positive gain is excluded from the default comparison",
    )
    args = parser.parse_args()

    environment = _environment()
    print(json.dumps(environment, indent=2))

    is_android = bool(environment["android_root"] or environment["android_data"])
    if is_android and args.input_wav is None:
        print(
            "WARNING: Android detected. The standard ONNX Runtime package reports "
            "Android as unsupported. Listen to reference.wav first, or rerun with "
            "--input-wav using a known-good source.",
            file=sys.stderr,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_wav is not None:
        reference, sample_rate = _load_reference(args.input_wav)
        source_description = str(args.input_wav)
    else:
        reference, sample_rate = _synthesize_reference(text=args.text, voice=args.voice)
        source_description = f"PyKokoro voice={args.voice!r}"

    if sample_rate <= 0:
        raise ValueError(f"sample rate must be positive, received {sample_rate}")

    records: list[dict[str, Any]] = []
    reference_path = args.output_dir / "reference.wav"
    reference_record = {
        "kind": "reference",
        "method": None,
        "source": source_description,
        "file": reference_path.name,
        **_stats(reference, sample_rate=sample_rate),
    }
    reference_record.update(_write_verified_pcm16(reference_path, reference, sample_rate))
    records.append(reference_record)

    print(
        f"{'reference':>14}: {reference_path} "
        f"peak={reference_record['peak']:.4f}, "
        f"rms={reference_record['rms']:.4f}, "
        f"duration={reference_record['duration_seconds']:.3f}s"
    )

    rate_number = parse_rate(args.rate)
    expected_samples = max(1, round(reference.size / rate_number))

    for method in METHODS:
        config = ProsodyConfig(method=method, fallback_methods=(), strict=True, clip=False)

        started = perf_counter()
        rendered = apply_prosody(
            reference,
            sample_rate,
            rate=args.rate,
            pitch=args.pitch,
            volume=args.volume,
            config=config,
        )
        elapsed = perf_counter() - started
        rendered = _mono_float32(rendered)

        output_path = args.output_dir / f"prosody_{method}.wav"
        record = {
            "kind": "processed",
            "method": method,
            "source_sha256_float_bytes": reference_record["sha256_float_bytes"],
            "file": output_path.name,
            "rate": args.rate,
            "pitch": args.pitch,
            "volume": args.volume,
            **_stats(
                rendered,
                sample_rate=sample_rate,
                expected_samples=expected_samples,
                runtime_seconds=elapsed,
            ),
        }
        record.update(_write_verified_pcm16(output_path, rendered, sample_rate))
        records.append(record)

        print(
            f"{method:>14}: {output_path} "
            f"peak={record['peak']:.4f}, "
            f"rms={record['rms']:.4f}, "
            f"duration={record['duration_seconds']:.3f}s, "
            f"RTF={record['real_time_factor']:.3f}"
        )

    report = {
        "environment": environment,
        "source": source_description,
        "sample_rate": sample_rate,
        "rate": args.rate,
        "pitch": args.pitch,
        "volume": args.volume,
        "records": records,
    }
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Metrics: {report_path}")
    print("Listen to reference.wav before evaluating processed files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
