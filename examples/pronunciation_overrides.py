"""Apply an explicit source-aligned pronunciation-language override."""

from __future__ import annotations

from pykokoro import (
    GenerationConfig,
    KokoroSynthesizer,
    PronunciationOverride,
    SynthesisConfig,
    SynthesisSegment,
)

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path


def main() -> None:
    request = SynthesisSegment(
        id="mixed-language",
        text="Hello Welt.",
        language="en-us",
        voice="af_sarah",
        pronunciation_overrides=(PronunciationOverride(start=6, end=10, language="de"),),
    )
    config = SynthesisConfig(
        voice="af_sarah",
        generation=GenerationConfig(lang="en-us"),
    )
    with KokoroSynthesizer(config) as synthesizer:
        rendered = synthesizer.synthesize(request)
    output = artifact_path("mixed-language.wav")
    rendered.save_wav(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
