#!/usr/bin/env python3
"""Generate German Gold-versus-Crane speech with Spokenform before sentence splitting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

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

OUTPUT_FILE = "german_thorsten.wav"
GOLD_LEXICONS = ("gold",)
CRANE_LEXICONS = ("crane",)
LEXICON_SEPARATOR_SECONDS = 1.0


def make_config(
    *,
    short_sentence: bool = True,
    lexicons: str | tuple[str, ...] | None = None,
) -> PipelineConfig:
    """Return the explicitly selected ready Thorsten configuration."""
    from pykokoro.short_sentence_handler import ShortSentenceConfig

    return PipelineConfig(
        voice="thorsten",
        model_source="github",
        model_variant="de-thorsten",
        model_quality="fp32",
        generation=GenerationConfig(lang="de", speed=1.0),
        tokenizer_config=TokenizerConfig(lexicons=lexicons),
        short_sentence_config=ShortSentenceConfig(enabled=short_sentence),
        return_trace=True,
    )


def combine_lexicon_audio(gold_result: Any, crane_result: Any) -> np.ndarray:
    """Combine Gold and Crane audio with a one-second silence separator."""
    if gold_result.sample_rate != crane_result.sample_rate:
        raise RuntimeError(
            "Gold and Crane runs returned different sample rates: "
            f"{gold_result.sample_rate} != {crane_result.sample_rate}"
        )
    if gold_result.audio.ndim != crane_result.audio.ndim:
        raise RuntimeError("Gold and Crane runs returned incompatible audio dimensions")

    gap_samples = round(gold_result.sample_rate * LEXICON_SEPARATOR_SECONDS)
    silence_shape = (gap_samples, *gold_result.audio.shape[1:])
    silence = np.zeros(silence_shape, dtype=gold_result.audio.dtype)
    crane_audio = crane_result.audio.astype(gold_result.audio.dtype, copy=False)
    return np.concatenate([gold_result.audio, silence, crane_audio], axis=0)


def print_result(label: str, result: Any, raw_segments_dir: str | None) -> None:
    """Print phonemes and warnings and optionally write labeled raw segments."""
    print(f"\n[{label}] phonemes:")
    for index, segment in enumerate(result.phoneme_segments, start=1):
        print(f"  [{index}] {segment.text!r}")
        print(f"      {segment.phonemes}")
        if raw_segments_dir and segment.raw_audio is not None:
            raw_path = Path(raw_segments_dir) / label / f"segment-{index:03d}.wav"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(raw_path, segment.raw_audio, result.sample_rate)
            print(f"      raw: {raw_path}")
    if result.trace and result.trace.warnings:
        print(f"[{label}] warnings:")
        for warning in result.trace.warnings:
            print(f"  - {warning}")


def main() -> None:
    """Generate the German Gold-versus-Crane comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-short-sentence",
        action="store_true",
        help="Disable short-sentence handling for exporter diagnostics.",
    )
    parser.add_argument(
        "--raw-segments-dir",
        type=str,
        help="Write each raw ONNX segment to gold/ and crane/ subdirectories.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    short_sentence = not args.no_short_sentence
    print("Generating German comparison:")
    print("  acoustic model: de-thorsten (Thorsten)")
    print("  first: gold lexicon")
    print("  second: crane lexicon")
    print(f"  separator: {LEXICON_SEPARATOR_SECONDS:.1f} s silence")

    with KokoroPipeline(
        make_config(short_sentence=short_sentence, lexicons=GOLD_LEXICONS)
    ) as pipeline:
        gold_result = pipeline.run(TEXT)
    with KokoroPipeline(
        make_config(short_sentence=short_sentence, lexicons=CRANE_LEXICONS)
    ) as pipeline:
        crane_result = pipeline.run(TEXT)

    print_result("gold", gold_result, args.raw_segments_dir)
    print_result("crane", crane_result, args.raw_segments_dir)
    audio = combine_lexicon_audio(gold_result, crane_result)
    sf.write(artifact_path(OUTPUT_FILE), audio, gold_result.sample_rate)
    print(f"\nWrote {OUTPUT_FILE}")
    print(f"Layout: gold -> {LEXICON_SEPARATOR_SECONDS:.1f} s silence -> crane")


if __name__ == "__main__":
    main()
