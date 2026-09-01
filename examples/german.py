#!/usr/bin/env python3
"""Generate German Gold-versus-Crane speech with the Martin v1.2 model."""

from __future__ import annotations

import logging
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


def make_config(*, lexicons: str | tuple[str, ...] | None = None) -> PipelineConfig:
    """Return the automatic German Martin configuration."""
    return PipelineConfig(
        generation=GenerationConfig(lang="de", speed=1.125),
        tokenizer_config=TokenizerConfig(lexicons=lexicons),
        return_trace=True,
    )


def combine_lexicon_audio(
    gold_result: Any, crane_result: Any, espeak_result: Any, olaph_result: Any
) -> np.ndarray:
    """Combine Gold and Crane audio with a one-second silence separator."""
    if gold_result.sample_rate != crane_result.sample_rate:
        raise RuntimeError(
            "Gold and Crane runs returned different sample rates: "
            f"{gold_result.sample_rate} != {crane_result.sample_rate}"
        )
    if gold_result.audio.ndim != crane_result.audio.ndim:
        raise RuntimeError("Gold and Crane runs returned incompatible audio dimensions")
    if gold_result.sample_rate != espeak_result.sample_rate:
        raise RuntimeError(
            "Gold and Espeak runs returned different sample rates: "
            f"{gold_result.sample_rate} != {espeak_result.sample_rate}"
        )
    if gold_result.audio.ndim != espeak_result.audio.ndim:
        raise RuntimeError("Gold and Espeak runs returned incompatible audio dimensions")
    if gold_result.sample_rate != olaph_result.sample_rate:
        raise RuntimeError(
            "Gold and Olaph runs returned different sample rates: "
            f"{gold_result.sample_rate} != {olaph_result.sample_rate}"
        )
    if gold_result.audio.ndim != olaph_result.audio.ndim:
        raise RuntimeError("Gold and Olaph runs returned incompatible audio dimensions")

    gap_samples = round(gold_result.sample_rate * LEXICON_SEPARATOR_SECONDS)
    silence_shape = (gap_samples, *gold_result.audio.shape[1:])
    silence = np.zeros(silence_shape, dtype=gold_result.audio.dtype)
    crane_audio = crane_result.audio.astype(gold_result.audio.dtype, copy=False)
    espeak_audio = espeak_result.audio.astype(gold_result.audio.dtype, copy=False)
    olaph_audio = olaph_result.audio.astype(gold_result.audio.dtype, copy=False)
    return np.concatenate(
        [gold_result.audio, silence, crane_audio, silence, espeak_audio, silence, olaph_audio],
        axis=0,
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


def main() -> None:
    """Generate the German Gold-versus-Crane comparison."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    print("Generating German comparison:")
    print("  acoustic model: Martin v1.2")
    print("  first: gold lexicon")
    print("  second: crane lexicon")
    print("  third: espeak lexicon")
    print("  forth: olaph lexicon")
    print(f"  separator: {LEXICON_SEPARATOR_SECONDS:.1f} s silence")

    with KokoroPipeline(make_config(lexicons=GOLD_LEXICONS)) as pipeline:
        gold_result = pipeline.run(TEXT)
    with KokoroPipeline(make_config(lexicons=CRANE_LEXICONS)) as pipeline:
        crane_result = pipeline.run(TEXT)
    with KokoroPipeline(make_config(lexicons=ESPEAK_LEXICONS)) as pipeline:
        espeak_result = pipeline.run(TEXT)
    with KokoroPipeline(make_config(lexicons=OLAPH_LEXICONS)) as pipeline:
        olaph_result = pipeline.run(TEXT)

    print_result("gold", gold_result)
    print_result("crane", crane_result)
    print_result("espeak", espeak_result)
    print_result("olaph", olaph_result)
    audio = combine_lexicon_audio(gold_result, crane_result, espeak_result, olaph_result)
    sf.write(OUTPUT_FILE, audio, gold_result.sample_rate)
    print(f"\nWrote {OUTPUT_FILE}")
    print(f"Layout: gold -> {LEXICON_SEPARATOR_SECONDS:.1f} s silence -> crane")


if __name__ == "__main__":
    main()
