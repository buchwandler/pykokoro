"""Render several independent speech requests without joining their waveforms."""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig, SynthesisSegment

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path


def main() -> None:
    config = SynthesisConfig(generation=GenerationConfig(lang="en-us"))
    requests = (
        SynthesisSegment("greeting-a", "Hello there.", "en-us", voice="af_sarah"),
        SynthesisSegment("greeting-b", "Good morning.", "en-us", voice="af_bella"),
    )
    with KokoroSynthesizer(config) as synthesizer:
        for rendered in synthesizer.synthesize_segments(requests):
            output = artifact_path(f"{rendered.id}.wav")
            rendered.save_wav(output)
            print(f"Wrote independent request {rendered.id!r} to {output}")


if __name__ == "__main__":
    main()
