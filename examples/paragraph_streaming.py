"""Render prepared PyKokoro paragraphs one at a time."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig


def main() -> None:
    script = """
# Chapter One

The first paragraph ends here.

The second paragraph follows.
"""
    pipeline = KokoroPipeline(PipelineConfig(voice="af_sarah"))
    with pipeline.prepare_units(script, unit="paragraph") as prepared:
        for descriptor in prepared.units:
            print(descriptor.index, descriptor.paragraph_idx, descriptor.text_hash)

        for result in prepared.render():
            try:
                output = Path(f"paragraph-{result.descriptor.index + 1:08d}.wav")
                sf.write(output, result.audio, result.sample_rate)
            finally:
                result.release_audio()


if __name__ == "__main__":
    main()
