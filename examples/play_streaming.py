#!/usr/bin/env python3
"""Play long-form speech with low startup latency and one persistent stream.

Install the optional playback dependency before running this example::

    pip install "pykokoro[cpu,playback]"

Run from the repository root::

    python examples/play_streaming.py

Document preparation is global, but sentence audio generation and playback overlap.
No temporary WAV file or complete generated waveform is retained.
"""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

TEXT = """
The first sentence starts playing as soon as it is generated.
While it plays, PyKokoro generates the next sentence.
The same persistent output stream is reused for the whole passage.
"""


def main() -> None:
    """Generate and play the sentence stream."""
    with KokoroPipeline(
        PipelineConfig(
            voice="af_sarah",
            generation=GenerationConfig(lang="en-us"),
        )
    ) as pipeline:
        pipeline.play_streaming(TEXT, queue_size=2)


if __name__ == "__main__":
    main()
