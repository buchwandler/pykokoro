#!/usr/bin/env python3
"""Generate German Gold-versus-Crane speech with the Martin v1.2 model."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

import numpy as np
import soundfile as sf

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
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


def make_config(*, lexicons: str | tuple[str, ...] | None = None) -> PipelineConfig:
    """Return the automatic German Martin configuration."""
    return PipelineConfig(
        generation=GenerationConfig(lang="de", speed=1.125),
        tokenizer_config=TokenizerConfig(lexicons=lexicons),
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


def print_performance(label: str, result: Any, wall_seconds: float) -> None:
    """Print measured pipeline and acoustic metrics for one lexicon run."""
    audio_seconds = len(result.audio) / result.sample_rate if result.sample_rate else 0.0
    summary = result.trace.inference_summary() if result.trace is not None else {}
    onnx_runtime_ms = float(summary.get("onnx_runtime_ms", 0.0))
    onnx_audio_seconds = float(summary.get("onnx_audio_seconds", audio_seconds))
    onnx_calls = int(summary.get("onnx_calls", 0))
    cache_hits = int(summary.get("onnx_cache_hits", 0))
    cache_misses = int(summary.get("onnx_cache_misses", 0))
    retry_calls = int(summary.get("short_sentence_retry_calls", 0))
    fallback_calls = int(summary.get("fallback_onnx_calls", 0))
    print(f"\n[{label}] performance:")
    print(f"  wall:                 {wall_seconds:.3f} s")
    print(f"  audio:                {audio_seconds:.3f} s")
    print(
        f"  end-to-end RTF:       {wall_seconds / audio_seconds:.3f}"
        if audio_seconds
        else "  end-to-end RTF:       n/a"
    )
    print(f"  ONNX calls:           {onnx_calls - cache_hits}")
    print(f"  ONNX time:            {onnx_runtime_ms / 1000.0:.3f} s")
    print(
        f"  model-only RTF:       {onnx_runtime_ms / 1000.0 / onnx_audio_seconds:.3f}"
        if onnx_audio_seconds
        else "  model-only RTF:       n/a"
    )
    print(f"  cache hits/misses:    {cache_hits}/{cache_misses}")
    print(f"  short retries/fallbacks: {retry_calls}/{fallback_calls}")


def main() -> None:
    """Generate the German Gold-versus-Crane comparison."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    print("Generating German comparison:")

    print("  acoustic model: Martin v1.2")
    print("  first:  gold lexicon")
    print("  second: crane lexicon")
    print("  third:  espeak lexicon")
    print("  fourth: olaph lexicon")
    print(f"  separator: {LEXICON_SEPARATOR_SECONDS:.1f} s silence")

    base_config = make_config(lexicons=None)
    results: dict[str, Any] = {}
    timings: dict[str, float] = {}
    with KokoroPipeline(base_config) as pipeline:
        assert base_config.tokenizer_config is not None
        for label, lexicons in LEXICON_SOURCES:
            tokenizer_config = replace(
                base_config.tokenizer_config,
                lexicons=lexicons,
            )
            started = time.perf_counter()
            results[label] = pipeline.run(
                TEXT,
                tokenizer_config=tokenizer_config,
            )
            timings[label] = time.perf_counter() - started

    for label, _lexicons in LEXICON_SOURCES:
        print_result(label, results[label])
        print_performance(label, results[label], timings[label])
    audio = combine_lexicon_audio(*(results[label] for label, _lexicons in LEXICON_SOURCES))
    sf.write(OUTPUT_FILE, audio, results[LEXICON_SOURCES[0][0]].sample_rate)
    print(f"\nWrote {OUTPUT_FILE}")
    print(f"Layout: {format_lexicon_layout()}")


if __name__ == "__main__":
    main()
