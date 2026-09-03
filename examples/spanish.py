#!/usr/bin/env python3
"""
Spanish TTS example using pykokoro.

This example demonstrates text-to-speech synthesis in Spanish
using the Kokoro model with Spanish voices.

Usage:
    python examples/spanish.py

Output:
    spanish_demo.wav - Generated Spanish speech audio

Available Spanish voices:
    - ef_dora (female)
    - em_alex, em_santa (male)
"""

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Spanish quote about life and dreams (inspired by Don Quixote)
TEXT = (
    "La vida es un viaje maravilloso lleno de aventuras. "
    "Quien no se atreve a sonar, nunca vera sus suenos hacerse realidad. "
    "Cada dia es una nueva oportunidad para ser mejor que ayer."
)

VOICE = "ef_dora"  # Spanish Female voice
LANG = "es"  # Spanish


def main():
    """Generate Spanish speech audio."""
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

    output_file = artifact_path("spanish_demo.wav")
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
