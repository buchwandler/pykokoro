#!/usr/bin/env python3
"""
Japanese TTS example using pykokoro.

This example demonstrates text-to-speech synthesis in Japanese
using the Kokoro model with Japanese voices.

Usage:
    python examples/japanese.py

Output:
    japanese_demo.wav - Generated Japanese speech audio

Available Japanese voices:
    - jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro (female)
    - jm_kumo (male)
"""

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Japanese proverb about perseverance and continuous improvement
TEXT = "七転び八起き。失敗を恐れず、何度でも立ち上がれ。一歩一歩、着実に前へ進むことが大切です。"

VOICE = "jf_alpha"  # Japanese Female voice
LANG = "ja"  # Japanese


def main():
    """Generate Japanese speech audio."""
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

    output_file = "japanese_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
