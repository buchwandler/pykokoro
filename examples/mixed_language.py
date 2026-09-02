#!/usr/bin/env python3
"""Mixed-language TTS using explicit SSMD language spans."""

from __future__ import annotations

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

SOURCE = (
    "Guten Tag. "
    '[This is an English phrase.]{lang="en-US"} '
    "Weiter auf Deutsch."
)


def main() -> None:
    """Render a document with an explicit German default and English span."""
    pipe = KokoroPipeline(
        PipelineConfig(
            voice="ff_siwis",
            generation=GenerationConfig(lang="de-DE"),
        )
    )
    try:
        result = pipe.run(SOURCE)
        sf.write("mixed_language_demo.wav", result.audio, result.sample_rate)
        print("Created mixed_language_demo.wav")
    finally:
        pipe.close()


if __name__ == "__main__":
    main()
