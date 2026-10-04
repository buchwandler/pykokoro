#!/usr/bin/env python3
"""Compare linear and spherical interpolation of two voices.

Synthesis may install model assets. Linear blending supports one or more voices;
SLERP requires exactly two voices.
"""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig, VoiceBlend

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

LINEAR_BLEND = "af_sarah=60,af_bella=40"
SLERP_BLEND = f"{LINEAR_BLEND}@slerp"
SAMPLE_TEXT = "This voice combines two styles."


def explicit_blend() -> VoiceBlend:
    return VoiceBlend(
        voices=[("af_sarah", 0.6), ("af_bella", 0.4)],
        interpolation="linear",
    )


def parsed_blend() -> VoiceBlend:
    return VoiceBlend.parse(LINEAR_BLEND)


def slerp_blend() -> VoiceBlend:
    return VoiceBlend.parse(SLERP_BLEND)


def make_config(blend: VoiceBlend | None = None) -> SynthesisConfig:
    return SynthesisConfig(
        voice=blend if blend is not None else explicit_blend(),
        generation=GenerationConfig(lang="en-us"),
    )


def main() -> None:
    linear = parsed_blend()
    spherical = slerp_blend()
    print(f"Linear blend: {linear}")
    print(f"SLERP blend: {spherical}")

    with KokoroSynthesizer(make_config(linear)) as synthesizer:
        linear_result = synthesizer.synthesize_text(SAMPLE_TEXT, language="en-us")
        spherical_result = synthesizer.synthesize_text(
            SAMPLE_TEXT,
            language="en-us",
            voice=spherical,
        )

    linear_output = artifact_path("voice_blend.wav")
    linear_result.save_wav(linear_output)
    print(f"Wrote {linear_output}")

    slerp_output = artifact_path("voice_blend_slerp.wav")
    spherical_result.save_wav(slerp_output)
    print(f"Wrote {slerp_output}")


if __name__ == "__main__":
    main()
