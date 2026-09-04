#!/usr/bin/env python3
"""Render portable SSMD voice roles and markers as one WAV per paragraph."""

from __future__ import annotations

import soundfile as sf

try:
    from ._output import artifact_dir
except ImportError:
    from _output import artifact_dir

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig, SSMDRenderConfig

SCRIPT = """---
title: Paragraph Streaming With Portable Voices
voice_bindings:
  kokoro:
    narrator: af_sarah
    analyst: am_michael
pause_defaults:
  enabled: true
  sentence: 280ms
  paragraph: 850ms
  voice_change: 350ms
---
@opening
<div voice="narrator">
This paragraph uses a logical narrator role. The concrete Kokoro voice lives only in
the portable header.
</div>

@analysis
<div voice="analyst">
This paragraph changes speaker, keeps the document-level pause policy, and is rendered
as an independent waveform.
</div>

@closing
<div voice="narrator">
Markers are reported with offsets local to their paragraph unit, which makes them easy
to store beside each WAV file.
</div>
"""


def main() -> None:
    output_dir = artifact_dir() / "paragraph-ssmd-voices"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        voice="af_sarah",
        generation=GenerationConfig(lang="en-us"),
        ssmd=SSMDRenderConfig(provider="kokoro", missing_voice="error"),
        retain_segment_audio=False,
    )

    with (
        KokoroPipeline(config) as pipeline,
        pipeline.prepare_units(SCRIPT, unit="paragraph") as prepared,
    ):
        print("Document metadata:", dict(prepared.document_metadata))
        for descriptor in prepared.units:
            print(
                f"unit={descriptor.index} paragraph={descriptor.paragraph_idx} "
                f"hash={descriptor.text_hash[:12]} markers={descriptor.marker_names}"
            )

        for result in prepared.render():
            output = output_dir / f"paragraph-{result.descriptor.index + 1:04d}.wav"
            try:
                sf.write(output, result.audio, result.sample_rate)
                print(f"Wrote {output}")
                for marker in result.markers:
                    seconds = marker["sample_offset"] / result.sample_rate
                    print(
                        f"  marker={marker['name']!r} "
                        f"sample={marker['sample_offset']} seconds={seconds:.3f}"
                    )
            finally:
                result.release_audio()


if __name__ == "__main__":
    main()
