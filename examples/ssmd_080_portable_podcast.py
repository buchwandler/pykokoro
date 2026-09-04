#!/usr/bin/env python3
"""Render a portable SSMD 0.8 podcast using logical speaker roles."""

from __future__ import annotations

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig, SSMDRenderConfig

PODCAST_SCRIPT = """---
title: Tech Talk
voice_bindings:
  kokoro:
    host: af_sarah
    cohost: am_michael
    guest: af_nicole
pause_defaults:
  enabled: true
  sentence: 250ms
  paragraph: 700ms
  voice_change: 350ms
---
<div voice="host">
Welcome to Tech Talk.
</div>

<div voice="cohost">
Today we are discussing portable SSMD documents.
</div>

<div voice="guest">
The body uses stable roles while the header selects concrete voices.
</div>
"""


def main() -> None:
    cfg = PipelineConfig(
        generation=GenerationConfig(lang="en-us"),
        ssmd=SSMDRenderConfig(),
    )
    result = KokoroPipeline(cfg).run(PODCAST_SCRIPT)
    sf.write(artifact_path("ssmd_080_portable_podcast.wav"), result.audio, result.sample_rate)
    print(f"Rendered {result.document_metadata.get('title')!r}")


if __name__ == "__main__":
    main()
