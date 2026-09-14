#!/usr/bin/env python3
"""English TTS example using the Kokoro 7M Distill model.

Usage:
    python examples/english_7m.py

Output:
    english_7m_demo.wav - Generated English speech audio
"""

from __future__ import annotations

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import ConsoleAssetProgress, GenerationConfig, KokoroPipeline, PipelineConfig

TEXT = (
    "This example uses the compact Kokoro seven million parameter Distill model. "
    "It speaks English with the af MSA voice while keeping the full PyKokoro pipeline."
)
MODEL_SOURCE = "github"
MODEL_VARIANT = "en-oddadmix-7m-distill"
MODEL_QUALITY = "fp32"
VOICE = "af_msa"
LANG = "en-us"


def main() -> None:
    """Generate English speech with the 7M Distill model."""
    print("Preparing the Kokoro 7M Distill TTS pipeline...")
    progress = ConsoleAssetProgress()
    config = PipelineConfig(
        model_source=MODEL_SOURCE,
        model_variant=MODEL_VARIANT,
        model_quality=MODEL_QUALITY,
        voice=VOICE,
        generation=GenerationConfig(lang=LANG, speed=1.0),
        asset_progress=progress,
    )

    print(f"Text: {TEXT}")
    print(f"Model: {MODEL_VARIANT}")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")
    print("\nGenerating audio...")

    with KokoroPipeline(config) as pipe:
        result = pipe.run(TEXT)

    output_file = artifact_path("english_7m_demo.wav")
    sf.write(output_file, result.audio, result.sample_rate)

    duration = len(result.audio) / result.sample_rate
    print(f"Created {output_file}")
    print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
