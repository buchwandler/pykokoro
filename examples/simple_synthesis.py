"""Synthesize one prepared sentence and write a standalone floating-point WAV."""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path


def main() -> None:
    config = SynthesisConfig(
        voice="af_sarah",
        generation=GenerationConfig(lang="en-us"),
    )
    with KokoroSynthesizer(config) as synthesizer:
        rendered = synthesizer.synthesize_text(
            "Hello from PyKokoro.",
            language="en-us",
            voice="af_sarah",
        )
    output = artifact_path("hello.wav")
    rendered.save_wav(output)
    print(f"Wrote {output} ({rendered.sample_rate} Hz)")


if __name__ == "__main__":
    main()
