#!/usr/bin/env python3
"""Synthesize a short Spanish sample with PyKokoro."""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

TEXT = (
    "La vida es un viaje maravilloso lleno de aventuras. "
    "Quien no se atreve a soñar nunca verá sus sueños hacerse realidad. "
    "Cada día es una nueva oportunidad para ser mejor que ayer."
)
VOICE = "ef_dora"
LANG = "es"


def main() -> None:
    """Generate Spanish speech audio."""
    config = SynthesisConfig(
        voice=VOICE,
        generation=GenerationConfig(lang=LANG, speed=1.0),
    )
    with KokoroSynthesizer(config) as synthesizer:
        print(f"Text: {TEXT}")
        print(f"Voice: {VOICE}")
        print(f"Language: {LANG}")
        rendered = synthesizer.synthesize_text(TEXT, language=LANG, voice=VOICE)

    output_file = artifact_path("spanish_demo.wav")
    rendered.save_wav(output_file)
    duration = len(rendered.audio) / rendered.sample_rate
    print(f"Created {output_file} ({duration:.2f} seconds)")


if __name__ == "__main__":
    main()
