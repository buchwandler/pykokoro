#!/usr/bin/env python3
"""Blend voices with either the structured or compact public constructor.

Synthesis may install model assets. SLERP is also available through
``VoiceBlend.parse(...@slerp)`` and requires exactly two voices.
"""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig, VoiceBlend

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path


def explicit_blend() -> VoiceBlend:
    return VoiceBlend(
        voices=[("af_sarah", 0.6), ("af_bella", 0.4)],
        interpolation="linear",
    )


def parsed_blend() -> VoiceBlend:
    return VoiceBlend.parse("af_sarah:60,af_bella:40")


def make_config(blend: VoiceBlend | None = None) -> SynthesisConfig:
    return SynthesisConfig(
        voice=blend if blend is not None else explicit_blend(),
        generation=GenerationConfig(lang="en-us"),
    )


def main() -> None:
    structured = explicit_blend()
    parsed = parsed_blend()
    print(f"Structured blend: {structured}")
    print(f"Parsed blend: {parsed}")
    with KokoroSynthesizer(make_config(parsed)) as synthesizer:
        result = synthesizer.synthesize_text("This voice combines two styles.", language="en-us")
    output = artifact_path("voice_blend.wav")
    result.save_wav(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
