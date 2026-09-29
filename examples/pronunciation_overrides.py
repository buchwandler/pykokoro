"""Compare language and direct-phoneme pronunciation overrides on source spans."""

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
    requests = (
        SynthesisSegment(
            id="language-override",
            text="Hello Welt.",
            language="en-us",
            voice="af_sarah",
            pronunciation_overrides=(PronunciationOverride(start=6, end=10, language="de"),),
        ),
        SynthesisSegment(
            id="phoneme-span-override",
            text="Hello world.",
            language="en-us",
            voice="af_sarah",
            pronunciation_overrides=(PronunciationOverride(start=6, end=11, phonemes="wˈɜɹld"),),
        ),
    )
    config = SynthesisConfig(
        voice="af_sarah",
        generation=GenerationConfig(lang="en-us"),
    )
    with KokoroSynthesizer(config) as synthesizer:
        for request in requests:
            rendered = synthesizer.synthesize(request)
            output = artifact_path(f"{request.id}.wav")
            rendered.save_wav(output)
            print(f"Wrote {request.id} to {output}")


if __name__ == "__main__":
    main()
