#!/usr/bin/env python3
"""Generate paragraphs and play them through one persistent audio stream.

Install the optional playback dependency before running this example::

    pip install "pykokoro[cpu,playback]"

Run from the repository root::

    python examples/play_paragraphs.py

Audio is copied into a bounded playback queue before each rendered result is
released. No WAV files are created.
"""

from __future__ import annotations

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.playback import SoundDevicePlayer

TEXT = """
This is the first paragraph of the direct playback example.

This is the second paragraph. It is rendered while the player consumes audio.

This is the third paragraph, also sent directly to the audio device.
"""


def main() -> None:
    player: SoundDevicePlayer | None = None
    with KokoroPipeline(PipelineConfig(voice="af_sarah")) as pipeline, pipeline.prepare_units(
        TEXT, unit="paragraph"
    ) as prepared:
            try:
                for result in prepared.render():
                    try:
                        if player is None:
                            player = SoundDevicePlayer(result.sample_rate, queue_size=2)
                            player.start()
                        player.submit(result.audio)
                    finally:
                        result.release_audio()
                if player is not None:
                    player.drain()
            finally:
                if player is not None:
                    player.close()


if __name__ == "__main__":
    main()
