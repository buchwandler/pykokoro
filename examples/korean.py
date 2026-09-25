#!/usr/bin/env python3
"""Experimental Korean phonemization example using PyKokoro."""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig, VoiceBlend

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

TEXT = "안녕하세요. 한국어 음성 합성 예제입니다. 오늘은 날씨가 좋아 공원에서 산책하기 좋습니다."
BLEND = "jf_alpha:50,jf_gongitsune:50"
LANG = "ko"


def main() -> None:
    """Generate an experimental Korean speech sample."""
    voice = VoiceBlend.parse(BLEND)
    config = SynthesisConfig(
        voice=voice,
        generation=GenerationConfig(lang=LANG, speed=1.0),
    )
    print("Kokoro was not trained specifically for Korean; pronunciation may be inaccurate.")
    print(f"Voice blend: {BLEND}")
    print(f"Language: {LANG}")
    with KokoroSynthesizer(config) as synthesizer:
        rendered = synthesizer.synthesize_text(TEXT, language=LANG, voice=voice)

    output_file = artifact_path("korean_demo.wav")
    rendered.save_wav(output_file)
    duration = len(rendered.audio) / rendered.sample_rate
    print(f"Created {output_file} ({duration:.2f} seconds)")


if __name__ == "__main__":
    main()
