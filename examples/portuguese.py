#!/usr/bin/env python3
"""
Brazilian Portuguese TTS example using pykokoro.

This example demonstrates text-to-speech synthesis in Brazilian Portuguese
using the Kokoro model with Portuguese voices.

Usage:
    python examples/portuguese.py

Output:
    portuguese_demo.wav - Generated Portuguese speech audio

Available Portuguese voices:
    - pf_dora (female)
    - pm_alex, pm_santa (male)
"""

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Brazilian Portuguese quote about dreams and life
TEXT = (
    "A vida e feita de escolhas. Cada passo que damos nos leva a um novo caminho. "
    "Sonhe grande, trabalhe duro, e nunca desista dos seus objetivos. "
    "O sucesso e a soma de pequenos esforcos repetidos dia apos dia."
)

VOICE = "pf_dora"  # Portuguese Female voice
LANG = "pt-br"  # Brazilian Portuguese


def main():
    """Generate Portuguese speech audio."""
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

    output_file = artifact_path("portuguese_demo.wav")
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
