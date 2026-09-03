#!/usr/bin/env python3
"""
French TTS example using pykokoro.

This example demonstrates text-to-speech synthesis in French
using the Kokoro model with French voices.

Usage:
    python examples/french.py

Output:
    french_demo.wav - Generated French speech audio

Available French voices:
    - ff_siwis (female)
"""

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# French quote about life and dreams (Victor Hugo inspired)
TEXT = (
    "La vie est un voyage, pas une destination. "
    "Chaque jour est une nouvelle page dans le livre de notre existence. "
    "Il faut rever sa vie et vivre son reve."
)

VOICE = "ff_siwis"  # French Female voice
LANG = "fr-fr"  # French


def main():
    """Generate French speech audio."""
    print("Initializing TTS engine...")
    pipe = KokoroPipeline(
        PipelineConfig(voice=VOICE, generation=GenerationConfig(lang=LANG, speed=1.0))
    )

    print(f"Text: {TEXT}")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")

    print("\nGenerating audio...")
    res = pipe.run(TEXT)
    samples, sample_rate = res.audio, res.sample_rate

    output_file = artifact_path("french_demo.wav")
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
