#!/usr/bin/env python3
"""Render the same SSMD prosody example with each AudioSig speech backend."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig, ProsodyConfig

TEXT = """
The package must arrive
[today]{rate="87%" pitch="+1.2st" volume="+2dB"}.
""".strip()

METHODS = ("wsola", "esola", "td_psola", "phase_vocoder")


def main() -> None:
    output_dir = Path("prosody_algorithm_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        config = PipelineConfig(
            voice="af_bella",
            prosody=ProsodyConfig(
                method=method,
                strict=True,
            ),
        )
        with KokoroPipeline(config) as pipeline:
            result = pipeline.run(TEXT)

        output = output_dir / f"prosody_{method}.wav"
        sf.write(output, result.audio, result.sample_rate)
        print(f"{method:>14}: {output} ({len(result.audio) / result.sample_rate:.3f}s)")


if __name__ == "__main__":
    main()
