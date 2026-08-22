#!/usr/bin/env python3
"""Generate speech and play it directly through the system audio device.

Install the optional playback dependency before running this example::

    pip install "pykokoro[cpu,playback]"

Run from the repository root::

    python examples/play_audio.py

This example keeps the generated waveform in memory and does not create a WAV file.
"""

from __future__ import annotations

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

TEXT = "This speech is played directly without writing an audio file."
VOICE = "af_heart"
LANGUAGE = "en-us"


def main() -> None:
    """Generate and play a short example sentence."""
    print("Initializing TTS engine...")
    config = PipelineConfig(
        voice=VOICE,
        generation=GenerationConfig(lang=LANGUAGE, speed=1.0),
    )

    with KokoroPipeline(config) as pipeline:
        result = pipeline.run(TEXT)
        print(f'Playing: "{TEXT}"')
        result.play()

    print("Playback finished.")


if __name__ == "__main__":
    main()
