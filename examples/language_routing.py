#!/usr/bin/env python3
"""Demonstrate prepared-text routing and deterministic source-span overrides.

Synthesis may install model assets. Pronunciation-language routing affects G2P
for prepared text only: it does not switch the acoustic model or voice. For a
known span, an explicit ``PronunciationOverride(language=...)`` is deterministic.
"""

from __future__ import annotations

from pykokoro import (
    GenerationConfig,
    KokoroSynthesizer,
    LanguageRoutingConfig,
    PronunciationOverride,
    SynthesisConfig,
    SynthesisSegment,
)

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path


def make_config() -> SynthesisConfig:
    return SynthesisConfig(
        voice="af_sarah",
        generation=GenerationConfig(lang="en-us"),
        language_routing=LanguageRoutingConfig(mode="auto", languages=("en", "de")),
    )


def make_requests() -> tuple[SynthesisSegment, SynthesisSegment]:
    return (
        SynthesisSegment(
            id="automatic-routing",
            text="Hello Welt.",
            language="en-us",
            voice="af_sarah",
        ),
        SynthesisSegment(
            id="explicit-german-span",
            text="Hello Welt.",
            language="en-us",
            voice="af_sarah",
            pronunciation_overrides=(PronunciationOverride(start=6, end=10, language="de"),),
        ),
    )


def main() -> None:
    with KokoroSynthesizer(make_config()) as synthesizer:
        for request in make_requests():
            result = synthesizer.synthesize(request)
            output = artifact_path(f"{request.id}.wav")
            result.save_wav(output)
            print(f"Wrote {request.id} to {output}")
    print("Routing changed pronunciation analysis, not the acoustic model or voice.")


if __name__ == "__main__":
    main()
