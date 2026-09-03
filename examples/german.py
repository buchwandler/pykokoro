#!/usr/bin/env python3
"""Generate German Gold-versus-Crane speech with the Martin v1.2 model."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.spacy_models import SpacyModelSize
from pykokoro.tokenizer import TokenizerConfig

TEXT = (
    "Zum 14.05.2026 um 18:20 Uhr ist das Abendessen geplant. "
    "Für den Auflauf brauchen wir 1,5 kg Kartoffeln, 500 g Quark, "
    "2 Eier, 1 ltr. Milch und ggf. 3 cm mehr Backpapier. "
    'Prof. Klein sagt: "Bitte stelle die Form auf die 2. Schiene, '
    "backe alles für 45 Min. und lass es danach 1 Min. oder auch "
    '2 Min. ruhen." Die Kosten liegen bei ca. 12,80 EUR zzgl. Pfand.'
)

OUTPUT_FILE = "german_martin_v1_2.wav"
GOLD_LEXICONS = ("gold",)
CRANE_LEXICONS = ("crane",)
ESPEAK_LEXICONS = ("espeak",)
OLAPH_LEXICONS = ("olaph",)
LEXICON_SEPARATOR_SECONDS = 1.0
LEXICON_SOURCES = (
    ("gold", GOLD_LEXICONS),
    ("crane", CRANE_LEXICONS),
    ("espeak", ESPEAK_LEXICONS),
    ("olaph", OLAPH_LEXICONS),
)


SPACY_MODES = ("off", "auto", "sm", "md", "lg", "trf")
STAGE_TIMING_KEYS = (
    ("doc", "parse"),
    ("language_plan", "source"),
    ("linguistics", "pass_a"),
    ("text_preparation", "prepare"),
    ("language_plan", "prepared"),
    ("linguistics", "pass_b"),
    ("segmentation", "split"),
    ("g2p", "phonemize"),
    ("runtime", "resolve_stages"),
    ("phoneme_processing", "preprocess"),
    ("audio_generation", "generate"),
    ("audio_postprocessing", "postprocess"),
)


def make_config(
    *,
    lexicons: str | tuple[str, ...] | None = None,
    spacy_mode: str = "auto",
) -> PipelineConfig:
    """Return the deterministic German Martin benchmark configuration."""
    if spacy_mode not in SPACY_MODES:
        raise ValueError(f"spacy_mode must be one of {SPACY_MODES}, got {spacy_mode!r}")
    if spacy_mode == "off":
        tokenizer_config = TokenizerConfig(lexicons=lexicons, use_spacy=False)
    elif spacy_mode == "auto":
        tokenizer_config = TokenizerConfig(lexicons=lexicons, use_spacy=None)
    else:
        tokenizer_config = TokenizerConfig(
            lexicons=lexicons,
            use_spacy=True,
            spacy_model_size=cast(SpacyModelSize, spacy_mode),
        )
    return PipelineConfig(
        generation=GenerationConfig(lang="de", speed=1.125),
        tokenizer_config=tokenizer_config,
        return_trace=True,
    )


def combine_lexicon_audio(
    gold_result: Any,
    crane_result: Any,
    espeak_result: Any | None = None,
    olaph_result: Any | None = None,
) -> np.ndarray:
    """Combine lexicon audio with one-second silence separators."""
    results = [gold_result, crane_result]
    if espeak_result is not None:
        results.append(espeak_result)
    if olaph_result is not None:
        results.append(olaph_result)
    for result in results[1:]:
        if gold_result.sample_rate != result.sample_rate:
            raise RuntimeError(
                "German runs returned different sample rates: "
                f"{gold_result.sample_rate} != {result.sample_rate}"
            )
        if gold_result.audio.ndim != result.audio.ndim:
            raise RuntimeError("German runs returned incompatible audio dimensions")

    gap_samples = round(gold_result.sample_rate * LEXICON_SEPARATOR_SECONDS)
    silence_shape = (gap_samples, *gold_result.audio.shape[1:])
    silence = np.zeros(silence_shape, dtype=gold_result.audio.dtype)
    audio_parts: list[np.ndarray] = []
    for index, result in enumerate(results):
        if index:
            audio_parts.append(silence)
        audio_parts.append(result.audio.astype(gold_result.audio.dtype, copy=False))
    return np.concatenate(audio_parts, axis=0)


def format_lexicon_layout() -> str:
    """Return the ordered audio layout shown by the comparison example."""
    return f" -> {LEXICON_SEPARATOR_SECONDS:.1f} s silence -> ".join(
        label for label, _lexicons in LEXICON_SOURCES
    )


def print_result(label: str, result: Any) -> None:
    """Print phonemes and warnings for one lexicon run."""
    print(f"\n[{label}] phonemes:")
    for index, segment in enumerate(result.phoneme_segments, start=1):
        print(f"  [{index}] {segment.text!r}")
        print(f"      {segment.phonemes}")
    if result.trace and result.trace.warnings:
        print(f"[{label}] warnings:")
        for warning in result.trace.warnings:
            print(f"  - {warning}")


@dataclass(frozen=True)
class BenchmarkRun:
    result: Any
    wall_seconds: float
    run_index: int


def effective_spacy_policy(result: Any, requested_mode: str) -> str:
    """Return the requested mode and concrete model selected by the pipeline."""
    if requested_mode == "off":
        return "off"
    metadata = getattr(result, "document_metadata", {})
    spacy_models = metadata.get("spacy_models", {}) if isinstance(metadata, dict) else {}
    selected: str | None = None
    if isinstance(spacy_models, dict):
        for selection in spacy_models.values():
            if isinstance(selection, dict) and selection.get("selected_model"):
                selected = str(selection["selected_model"])
                break
    return f"{requested_mode} -> {selected or 'fallback'}"


def stage_times(result: Any) -> dict[str, float]:
    """Return documented top-level stage durations in seconds."""
    if result.trace is None:
        return {}
    summary = result.trace.event_summary()
    return {
        f"{stage}/{name}": summary.get((stage, name), 0.0) / 1000.0
        for stage, name in STAGE_TIMING_KEYS
    }


def print_performance(
    label: str,
    run: BenchmarkRun,
    requested_spacy_mode: str,
    *,
    show_stage_times: bool,
    accounted_stage_seconds: float,
) -> None:
    """Print measured pipeline, stage, acoustic, and cold/warm metrics."""
    result = run.result
    audio_seconds = len(result.audio) / result.sample_rate if result.sample_rate else 0.0
    summary = result.trace.inference_summary() if result.trace is not None else {}
    onnx_runtime_ms = float(summary.get("onnx_runtime_ms", 0.0))
    onnx_audio_seconds = float(summary.get("onnx_audio_seconds", audio_seconds))
    onnx_calls = int(summary.get("onnx_calls", 0))
    cache_hits = int(summary.get("onnx_cache_hits", 0))
    cache_misses = int(summary.get("onnx_cache_misses", 0))
    retry_calls = int(summary.get("short_sentence_retry_calls", 0))
    fallback_calls = int(summary.get("fallback_onnx_calls", 0))
    onnx_seconds = onnx_runtime_ms / 1000.0
    non_onnx_seconds = max(0.0, run.wall_seconds - onnx_seconds)
    state = "cold" if run.run_index == 1 else "warm"
    print(f"\n[{label}] performance ({state}, run {run.run_index}):")
    print(f"  spaCy:                {effective_spacy_policy(result, requested_spacy_mode)}")
    print(f"  wall:                 {run.wall_seconds:.3f} s")
    print(f"  accounted stages:     {accounted_stage_seconds:.3f} s")
    print(f"  unaccounted wall:     {run.wall_seconds - accounted_stage_seconds:.3f} s")
    print(f"  audio:                {audio_seconds:.3f} s")
    print(
        f"  end-to-end RTF:       {run.wall_seconds / audio_seconds:.3f}"
        if audio_seconds
        else "  end-to-end RTF:       n/a"
    )
    print(f"  ONNX calls:           {onnx_calls - cache_hits}")
    print(f"  ONNX time:            {onnx_seconds:.3f} s")
    print(f"  non-ONNX time:        {non_onnx_seconds:.3f} s")
    print(
        f"  model-only RTF:       {onnx_seconds / onnx_audio_seconds:.3f}"
        if onnx_audio_seconds
        else "  model-only RTF:       n/a"
    )
    print(f"  cache hits/misses:    {cache_hits}/{cache_misses}")
    print(f"  short retries/fallbacks: {retry_calls}/{fallback_calls}")
    if show_stage_times:
        print("  stage timing:")
        for name, seconds in stage_times(result).items():
            print(f"    {name:32s} {seconds:.3f} s")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spacy",
        choices=SPACY_MODES,
        default="off",
        help="spaCy policy: off, auto, or an explicit model size (default: off)",
    )
    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=1,
        help="measured runs per lexicon (default: 1)",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="warm every lexicon once before measured runs",
    )
    parser.add_argument(
        "--show-stage-times",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show aggregate stage timing (default: enabled)",
    )
    return parser


def measure_runs(
    pipeline: KokoroPipeline,
    frontend: Any,
    configs: dict[str, TokenizerConfig],
    *,
    warmup: bool,
    runs: int,
) -> dict[str, list[BenchmarkRun]]:
    """Render every lexicon from one prepared frontend."""
    if warmup:
        for config in configs.values():
            pipeline.render_frontend(frontend, tokenizer_config=config)
    runs_by_label: dict[str, list[BenchmarkRun]] = {}
    for label, _lexicons in LEXICON_SOURCES:
        measured: list[BenchmarkRun] = []
        for run_index in range(1, runs + 1):
            started = time.perf_counter()
            result = pipeline.render_frontend(frontend, tokenizer_config=configs[label])
            measured.append(BenchmarkRun(result, time.perf_counter() - started, run_index))
        runs_by_label[label] = measured
    return runs_by_label


def main(argv: list[str] | None = None) -> None:
    """Generate the German Gold-versus-Crane comparison."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    print("Generating German comparison:")
    print("  acoustic model: Martin v1.2")
    print(f"  spaCy policy: {args.spacy}")
    print(f"  measured runs: {args.runs}")
    print(f"  warmup: {'enabled' if args.warmup else 'disabled'}")
    print(f"  separator: {LEXICON_SEPARATOR_SECONDS:.1f} s silence")

    base_config = make_config(lexicons=None, spacy_mode=args.spacy)
    runs_by_label: dict[str, list[BenchmarkRun]]
    with KokoroPipeline(base_config) as pipeline:
        assert base_config.tokenizer_config is not None
        configs = {
            label: replace(base_config.tokenizer_config, lexicons=lexicons)
            for label, lexicons in LEXICON_SOURCES
        }
        with pipeline.prepare_frontend(
            TEXT, tokenizer_config=base_config.tokenizer_config
        ) as frontend:
            runs_by_label = measure_runs(
                pipeline,
                frontend,
                configs,
                warmup=args.warmup,
                runs=args.runs,
            )

    for label, _lexicons in LEXICON_SOURCES:
        runs = runs_by_label[label]
        final_run = runs[-1]
        print_result(label, final_run.result)
        for run in runs:
            accounted = sum(stage_times(run.result).values())
            print_performance(
                label,
                run,
                args.spacy,
                show_stage_times=args.show_stage_times,
                accounted_stage_seconds=accounted,
            )
    final_results = [runs_by_label[label][-1].result for label, _lexicons in LEXICON_SOURCES]
    audio = combine_lexicon_audio(*final_results)
    sf.write(artifact_path(OUTPUT_FILE), audio, final_results[0].sample_rate)
    print(f"\nWrote {OUTPUT_FILE}")
    print(f"Layout: {format_lexicon_layout()}")


if __name__ == "__main__":
    main()
