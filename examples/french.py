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

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

# French quote about life and dreams (Victor Hugo inspired)
TEXT = (
    "La vie est un voyage, pas une destination. "
    "Chaque jour est une nouvelle page dans le livre de notre existence. "
    "Il faut rêver sa vie et vivre son rêve."
)
VOICE = "ff_siwis"  # French female voice
LANG = "fr-fr"  # French


def main() -> None:
    """Generate French speech audio."""
    config = SynthesisConfig(
        voice=VOICE,
        generation=GenerationConfig(lang=LANG, speed=1.0),
    )
    with KokoroSynthesizer(config) as synthesizer:
        print(f"Text: {TEXT}")
        print(f"Voice: {VOICE}")
        print(f"Language: {LANG}")
        rendered = synthesizer.synthesize_text(TEXT, language=LANG, voice=VOICE)

    output_file = artifact_path("french_demo.wav")
    rendered.save_wav(output_file)
    duration = len(rendered.audio) / rendered.sample_rate
    print(f"Created {output_file} ({duration:.2f} seconds)")


if __name__ == "__main__":
    main()
