#!/usr/bin/env python3
"""Benchmark neutral PyKokoro voice loudness and write reviewed candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from audiosig import measure_loudness  # noqa: E402

from pykokoro import LoudnessConfig, PipelineConfig, __version__  # noqa: E402
from pykokoro.discovery import ModelCapabilities, discover_models  # noqa: E402
from pykokoro.generation_config import GenerationConfig  # noqa: E402
from pykokoro.voice_level import VoiceCalibrationKey  # noqa: E402

CORPUS_PATH = Path(__file__).with_name("data") / "voice_loudness_corpus.json"
DEFAULT_OUTPUT = Path("artifacts/voice_loudness")


def _load_corpus(path: Path = CORPUS_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or not isinstance(data.get("name"), str):
        raise ValueError("voice loudness corpus must use schema 1 and have a name")
    if not isinstance(data.get("locales"), dict):
        raise ValueError("voice loudness corpus locales must be an object")
    return data


def _quality(model: ModelCapabilities, requested: str | None) -> str:
    if requested is not None:
        if requested not in model.qualities:
            raise ValueError(f"quality {requested!r} is unavailable for model {model.model_id!r}")
        return requested
    if "fp32" in model.qualities:
        return "fp32"
    if not model.qualities:
        raise ValueError(f"model {model.model_id!r} has no runnable quality")
    return model.qualities[0]


def _rms_dbfs(audio: np.ndarray) -> float:
    rms = (
        float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64)))))
        if audio.size
        else 0.0
    )
    return -math.inf if rms == 0.0 else 20.0 * math.log10(rms)


def _selected_models(args: argparse.Namespace) -> list[tuple[ModelCapabilities, str]]:
    discovery = discover_models(offline=args.offline, preference=args.preference)
    selected: list[tuple[ModelCapabilities, str]] = []
    for model in discovery.models:
        if args.model_source and model.source != args.model_source:
            continue
        if args.model and model.model_id != args.model:
            continue
        if model.status not in {"ready", "experimental"} or not model.runtime_available:
            continue
        selected.append((model, _quality(model, args.quality)))
    return selected


def _entries(args: argparse.Namespace, corpus: dict[str, Any]) -> list[dict[str, Any]]:
    locales = corpus["locales"]
    result: list[dict[str, Any]] = []
    for model, quality in _selected_models(args):
        details = {detail.name: detail for detail in model.voice_details}
        for voice in model.voices:
            detail = details.get(voice)
            locale = (
                detail.locale
                if detail is not None
                else (model.languages[0] if model.languages else "")
            )
            if args.voice and voice != args.voice:
                continue
            if args.locale and locale != args.locale:
                continue
            if locale not in locales:
                continue
            result.append(
                {
                    "model_source": model.source,
                    "model_id": model.model_id,
                    "quality": quality,
                    "voice": voice,
                    "locale": locale,
                    "gender": detail.gender if detail is not None else "unknown",
                    "sentences": locales[locale],
                    "sample_rate": model.sample_rate or 24000,
                }
            )
    return result[: args.max_voices] if args.max_voices else result


def _measure_entry(
    entry: dict[str, Any], pipeline: Any, corpus_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    measurements: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sentence in entry["sentences"]:
        try:
            result = pipeline.run(
                sentence["text"],
                model_source=entry["model_source"],
                model_variant=entry["model_id"],
                model_quality=entry["quality"],
                voice=entry["voice"],
                lang=entry["locale"],
                loudness=LoudnessConfig(),
            )
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            metrics = measure_loudness(audio, sample_rate=result.sample_rate)
            measurements.append(
                {
                    "model_source": entry["model_source"],
                    "model_id": entry["model_id"],
                    "quality": entry["quality"],
                    "voice": entry["voice"],
                    "locale": entry["locale"],
                    "gender": entry["gender"],
                    "corpus": corpus_name,
                    "sentence_id": sentence["id"],
                    "duration_seconds": len(audio) / result.sample_rate,
                    "sample_rate": result.sample_rate,
                    "integrated_lufs": metrics.integrated_lufs,
                    "sample_peak_dbfs": metrics.sample_peak_dbfs,
                    "true_peak_dbtp": metrics.true_peak_dbtp,
                    "rms_dbfs": _rms_dbfs(audio),
                    "random_seed": 0,
                    "pykokoro_version": __version__,
                    "audiosig_version": "0.1.4",
                }
            )
        except Exception as exc:
            failures.append(
                {
                    **{
                        key: entry[key]
                        for key in ("model_source", "model_id", "quality", "voice", "locale")
                    },
                    "sentence_id": sentence["id"],
                    "error": str(exc),
                }
            )
    return measurements, failures


def _aggregate(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for item in measurements:
        value = float(item["integrated_lufs"])
        if math.isfinite(value):
            grouped.setdefault(
                (item["model_source"], item["model_id"], item["quality"], item["voice"]), []
            ).append(value)
    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        rows.append(
            {
                "model_source": key[0],
                "model_id": key[1],
                "quality": key[2],
                "voice": key[3],
                "samples": len(values),
                "median_lufs": median,
                "mad_lu": mad,
                "min_lufs": min(values),
                "max_lufs": max(values),
                "status": "eligible" if len(values) >= 4 else "insufficient_samples",
            }
        )
    return rows


def _write_calibration(path: Path, corpus_name: str, aggregates: list[dict[str, Any]]) -> None:
    eligible = [row for row in aggregates if row["samples"] >= 4]
    reference = -18.0 if not eligible else statistics.median(row["median_lufs"] for row in eligible)
    eligible = [row for row in eligible if abs(reference - row["median_lufs"]) <= 6.0]
    voices = {
        str(
            VoiceCalibrationKey(row["model_source"], row["model_id"], row["quality"], row["voice"])
        ): {
            "measured_lufs": row["median_lufs"],
            "reference_lufs": reference,
            "mad_lu": row["mad_lu"],
            "gain_db": reference - row["median_lufs"],
            "samples": row["samples"],
            "method": "bs1770",
            "corpus_version": corpus_name,
        }
        for row in eligible
    }
    if (
        path.resolve()
        == (
            Path(__file__).parents[1] / "pykokoro" / "data" / "voice_level_calibration.json"
        ).resolve()
    ):
        raise ValueError("refusing to overwrite packaged production calibration data")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "method": "bs1770",
                "corpus": corpus_name,
                "reference_lufs": reference,
                "generated_with": {"pykokoro": __version__, "audiosig": "0.1.4"},
                "voices": voices,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _spread_rows(measurements: list[dict[str, Any]], field: str) -> list[tuple[str, float]]:
    grouped: dict[str, list[float]] = {}
    for item in measurements:
        value = float(item["integrated_lufs"])
        if math.isfinite(value):
            grouped.setdefault(str(item[field]), []).append(value)
    return sorted(
        (name, max(values) - min(values)) for name, values in grouped.items() if len(values) > 1
    )


def _write_outputs(
    output: Path,
    measurements: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    corpus_name: str,
    calibration_path: Path | None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "measurements.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "corpus": corpus_name,
                "measurements": measurements,
                "failures": failures,
                "aggregates": aggregates,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fields = (
        list(measurements[0])
        if measurements
        else [
            "model_source",
            "model_id",
            "quality",
            "voice",
            "locale",
            "sentence_id",
            "integrated_lufs",
        ]
    )
    with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(measurements)
    values = [row["median_lufs"] for row in aggregates]
    lines = [
        "# Voice loudness benchmark",
        "",
        f"Corpus: `{corpus_name}`",
        f"Models: {len({(row['model_source'], row['model_id']) for row in aggregates})}",
        f"Voices: {len(aggregates)}",
        f"Successful utterances: {len(measurements)}",
        f"Failures: {len(failures)}",
        "",
    ]
    if values:
        reference_rows = [row for row in aggregates if row["samples"] >= 4]
        reference = (
            statistics.median(row["median_lufs"] for row in reference_rows)
            if reference_rows
            else -18.0
        )
        insufficient = [row for row in aggregates if row["status"] == "insufficient_samples"]
        extreme = [row for row in aggregates if abs(reference - row["median_lufs"]) > 6.0]
        variable = sorted(aggregates, key=lambda row: row["mad_lu"], reverse=True)[:3]
        peak_keys = sorted(
            {
                str(
                    VoiceCalibrationKey(
                        item["model_source"], item["model_id"], item["quality"], item["voice"]
                    )
                )
                for item in measurements
                if float(item["true_peak_dbtp"]) > -3.0
            }
        )
        lines.extend(
            [
                f"Voice median LUFS: min={min(values):.2f}, median={statistics.median(values):.2f}, max={max(values):.2f}",
                f"Spread: {max(values) - min(values):.2f} LU",
                f"Population reference: {reference:.2f} LUFS",
                "",
                "## Eligibility and review flags",
                f"Insufficient samples (<4): {len(insufficient)}",
                f"Extreme correction (>6 dB): {len(extreme)}",
                "High sentence-to-sentence variability: "
                + (
                    ", ".join(f"{row['voice']} (MAD {row['mad_lu']:.2f})" for row in variable)
                    or "none"
                ),
                "Near peak/headroom threshold (> -3 dBTP): " + (", ".join(peak_keys) or "none"),
                "",
                "## Per-model spread",
            ]
        )
        lines.extend(
            f"- {name}: {spread:.2f} LU" for name, spread in _spread_rows(measurements, "model_id")
        )
        lines.extend(["", "## Per-locale spread"])
        lines.extend(
            f"- {name}: {spread:.2f} LU" for name, spread in _spread_rows(measurements, "locale")
        )
        lines.extend(
            [
                "",
                "## Voice aggregates",
                "",
                "| Key | Samples | Median LUFS | MAD | Range | Status |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        lines.extend(
            f"| `{row['model_source']}:{row['model_id']}:{row['quality']}:{row['voice']}` | {row['samples']} | {row['median_lufs']:.2f} | {row['mad_lu']:.2f} | {row['min_lufs']:.2f} to {row['max_lufs']:.2f} | {row['status']} |"
            for row in aggregates
        )
    else:
        lines.append("No successful measurements were recorded.")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if calibration_path is not None:
        _write_calibration(calibration_path, corpus_name, aggregates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", choices=("github", "huggingface"))
    parser.add_argument("--model")
    parser.add_argument("--voice")
    parser.add_argument("--locale")
    parser.add_argument("--quality")
    parser.add_argument("--max-voices", type=int)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--preference", choices=("auto", "github", "huggingface", "upstream"), default="auto"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-calibration", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = _load_corpus()
    entries = _entries(args, corpus)
    if not entries:
        raise SystemExit("No runnable voices matched the requested filters")
    from pykokoro import KokoroPipeline

    first = entries[0]
    config = PipelineConfig(
        model_source=first["model_source"],
        model_variant=first["model_id"],
        model_quality=first["quality"],
        voice=first["voice"],
        generation=GenerationConfig(lang=first["locale"], speed=1.0, random_seed=0),
    )
    measurements: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with KokoroPipeline(config) as pipeline:
        for index, entry in enumerate(entries, 1):
            print(
                f"[{index:03d}/{len(entries)}] {entry['model_id']} / {entry['voice']} / {entry['locale']}"
            )
            current, current_failures = _measure_entry(entry, pipeline, corpus["name"])
            measurements.extend(current)
            failures.extend(current_failures)
    aggregates = _aggregate(measurements)
    _write_outputs(
        args.output, measurements, failures, aggregates, corpus["name"], args.write_calibration
    )
    print(f"Successful utterances: {len(measurements)}; failures: {len(failures)}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
