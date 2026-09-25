#!/usr/bin/env python3
"""Synthesize a short Italian sample with PyKokoro."""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

TEXT = (
    "La vita è come una melodia: ogni nota ha il suo significato. "
    "L'arte non riproduce ciò che è visibile, ma rende visibile ciò che non sempre lo è. "
    "La bellezza salverà il mondo."
)
VOICE = "if_sara"
LANG = "it"


def main() -> None:
    """Generate Italian speech audio."""
    config = SynthesisConfig(
        voice=VOICE,
        generation=GenerationConfig(lang=LANG, speed=1.0),
    )
    with KokoroSynthesizer(config) as synthesizer:
        print(f"Text: {TEXT}")
        print(f"Voice: {VOICE}")
        print(f"Language: {LANG}")
        rendered = synthesizer.synthesize_text(TEXT, language=LANG, voice=VOICE)

    output_file = artifact_path("italian_demo.wav")
    rendered.save_wav(output_file)
    duration = len(rendered.audio) / rendered.sample_rate
    print(f"Created {output_file} ({duration:.2f} seconds)")


if __name__ == "__main__":
    main()
