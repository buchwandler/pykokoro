"""Pass source-aligned linguistic context without planner or spaCy objects."""

from __future__ import annotations

from pykokoro import (
    GenerationConfig,
    KokoroSynthesizer,
    LinguisticToken,
    SynthesisConfig,
    SynthesisSegment,
)

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path


def main() -> None:
    request = SynthesisSegment(
        id="contextual-pronunciation",
        text="I read the note yesterday.",
        language="en-us",
        voice="af_sarah",
        annotations=(LinguisticToken(2, 6, text="read", pos="VERB", tag="VBD", lemma="read"),),
    )
    config = SynthesisConfig(
        voice="af_sarah",
        generation=GenerationConfig(lang="en-us"),
    )
    with KokoroSynthesizer(config) as synthesizer:
        rendered = synthesizer.synthesize(request)
    output = artifact_path("contextual-pronunciation.wav")
    rendered.save_wav(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
