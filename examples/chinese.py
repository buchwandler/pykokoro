#!/usr/bin/env python3
"""Synthesize a short Mandarin Chinese sample with the Kokoro v1.1-zh model."""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

TEXT = "学如逆水行舟，不进则退。知识就是力量，时间就是金钱。"
VOICE = "zf_001"
LANG = "zh"


def main() -> None:
    """Generate Mandarin Chinese speech audio."""
    config = SynthesisConfig(
        voice=VOICE,
        model_source="github",
        model_variant="v1.1-zh",
        model_quality="fp32",
        generation=GenerationConfig(lang=LANG, speed=1.0),
    )
    with KokoroSynthesizer(config) as synthesizer:
        print(f"Text: {TEXT}")
        print(f"Voice: {VOICE}")
        print(f"Language: {LANG}")
        rendered = synthesizer.synthesize_text(TEXT, language=LANG, voice=VOICE)

    output_file = artifact_path("chinese_demo.wav")
    rendered.save_wav(output_file)
    duration = len(rendered.audio) / rendered.sample_rate
    print(f"Created {output_file} ({duration:.2f} seconds)")


if __name__ == "__main__":
    main()
