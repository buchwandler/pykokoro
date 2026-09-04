#!/usr/bin/env python3
"""Stream prepared paragraph units into one chapter WAV and a marker sidecar.

Unlike ``pipeline.run()``, this example never concatenates all unit arrays in memory.
The output WAV is opened after the first unit establishes the sample rate, and every
subsequent unit is appended directly to the file.
"""

from __future__ import annotations

import json
from typing import Any

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

TEXT = """The first paragraph starts the chapter and contains an @intro marker.

The second paragraph is generated only after the first waveform has been written.

The final paragraph closes the streamed chapter without building a chapter-sized NumPy
array in memory.
"""


def main() -> None:
    wav_path = artifact_path("streamed-chapter.wav")
    marker_path = wav_path.with_suffix(".markers.json")
    writer: sf.SoundFile | None = None
    aggregate_markers: list[dict[str, Any]] = []
    base_sample_offset = 0

    try:
        with (
            KokoroPipeline(
                PipelineConfig(
                    voice="af_sarah",
                    generation=GenerationConfig(lang="en-us"),
                )
            ) as pipeline,
            pipeline.prepare_units(TEXT, unit="paragraph") as prepared,
        ):
            for result in prepared.render():
                try:
                    if writer is None:
                        writer = sf.SoundFile(
                            wav_path,
                            mode="w",
                            samplerate=result.sample_rate,
                            channels=1,
                            subtype="PCM_16",
                        )
                    writer.write(result.audio)
                    aggregate_markers.extend(
                        {
                            **marker,
                            "sample_offset": marker["sample_offset"] + base_sample_offset,
                        }
                        for marker in result.markers
                    )
                    base_sample_offset += len(result.audio)
                    print(f"Appended unit {result.descriptor.index}: {len(result.audio)} samples")
                finally:
                    result.release_audio()
    finally:
        if writer is not None:
            writer.close()

    marker_path.write_text(
        json.dumps(
            {
                "wav": str(wav_path),
                "sample_count": base_sample_offset,
                "markers": aggregate_markers,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {wav_path} and {marker_path}")


if __name__ == "__main__":
    main()
