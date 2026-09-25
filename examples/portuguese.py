#!/usr/bin/env python3
"""Synthesize a short Brazilian Portuguese sample with PyKokoro."""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

TEXT = (
    "A vida é feita de escolhas. Cada passo que damos nos leva a um novo caminho. "
    "Sonhe grande, trabalhe duro e nunca desista dos seus objetivos. "
    "O sucesso é a soma de pequenos esforços repetidos dia após dia."
)
VOICE = "pf_dora"
LANG = "pt-br"


def main() -> None:
    """Generate Brazilian Portuguese speech audio."""
    config = SynthesisConfig(
        voice=VOICE,
        generation=GenerationConfig(lang=LANG, speed=1.0),
    )
    with KokoroSynthesizer(config) as synthesizer:
        print(f"Text: {TEXT}")
        print(f"Voice: {VOICE}")
        print(f"Language: {LANG}")
        rendered = synthesizer.synthesize_text(TEXT, language=LANG, voice=VOICE)

    output_file = artifact_path("portuguese_demo.wav")
    rendered.save_wav(output_file)
    duration = len(rendered.audio) / rendered.sample_rate
    print(f"Created {output_file} ({duration:.2f} seconds)")


if __name__ == "__main__":
    main()
