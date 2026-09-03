"""Render prepared PyKokoro paragraphs one at a time."""

from __future__ import annotations

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

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
                output = artifact_path(f"paragraph-{result.descriptor.index + 1:08d}.wav")
                sf.write(output, result.audio, result.sample_rate)
            finally:
                result.release_audio()


if __name__ == "__main__":
    main()
