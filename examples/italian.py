#!/usr/bin/env python3
"""
Italian TTS example using pykokoro.

This example demonstrates text-to-speech synthesis in Italian
using the Kokoro model with Italian voices.

Usage:
    python examples/italian.py

Output:
    italian_demo.wav - Generated Italian speech audio

Available Italian voices:
    - if_sara (female)
    - im_nicola (male)
"""

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Italian quote about life, art, and beauty
TEXT = (
    "La vita e come una melodia: ogni nota ha il suo significato. "
    "L'arte non riproduce cio che e visibile, "
    "ma rende visibile cio che non sempre lo e. "
    "La bellezza salvera il mondo."
)

VOICE = "if_sara"  # Italian Female voice
LANG = "it"  # Italian


def main():
    """Generate Italian speech audio."""
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

    output_file = artifact_path("italian_demo.wav")
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
