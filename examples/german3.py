#!/usr/bin/env python3
"""Generate German speech with the Thorsten model."""

from __future__ import annotations

import logging

import soundfile as sf

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

TEXT = (
    "Zum 14.05.2026 um 18:20 Uhr ist das Abendessen geplant. "
    "Für den Auflauf brauchen wir 1,5 kg Kartoffeln, 500 g Quark, "
    "2 Eier, 1 ltr. Milch und ggf. 3 cm mehr Backpapier. "
    'Prof. Klein sagt: "Bitte stelle die Form auf die 2. Schiene, '
    "backe alles für 45 Min. und lass es danach 1 Min. oder auch "
    '2 Min. ruhen." Die Kosten liegen bei ca. 12,80 EUR zzgl. Pfand.'
)

OUTPUT_FILE = "german_thorsten.wav"


def make_config() -> PipelineConfig:
    """Return the explicitly selected ready Thorsten configuration."""
    return PipelineConfig(
        voice="thorsten",
        model_source="github",
        model_variant="de-thorsten",
        model_quality="fp32",
        generation=GenerationConfig(lang="de", speed=1.0),
        return_trace=True,
    )


def main() -> None:
    """Generate the normalization-heavy German demonstration."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    with KokoroPipeline(make_config()) as pipeline:
        result = pipeline.run(TEXT)

    sf.write(OUTPUT_FILE, result.audio, result.sample_rate)
    duration = len(result.audio) / result.sample_rate
    print(f"Created {OUTPUT_FILE}")
    print("Model: de-thorsten")
    print(f"Sample rate: {result.sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print("\nPhoneme segments:")
    for index, segment in enumerate(result.phoneme_segments, start=1):
        print(f"  [{index}] {segment.text!r}")
        print(f"      {segment.phonemes}")
    if result.trace and result.trace.warnings:
        print("Warnings:")
        for warning in result.trace.warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
