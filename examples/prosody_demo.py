#!/usr/bin/env python3
"""
Prosody control demo using native SSMD annotations.

SSMD prosody uses explicit ``[text]{key="value"}`` annotations. PyKokoro
carries the resulting rate, pitch, and volume metadata through the pipeline and
applies it during audio post-processing.

Requirements:
    pip install pykokoro[prosody]

Usage:
    python examples/prosody_demo.py
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

VOICE = "af_sarah"
LANG = "en-us"

VOLUME_SCRIPT = """
[This is extra soft.]{volume="x-soft"} ...500ms
[This is soft.]{volume="soft"} ...500ms
[This is medium volume.]{volume="medium"} ...500ms
[This is loud.]{volume="loud"} ...500ms
[This is extra loud.]{volume="x-loud"} ...500ms
[This is six decibels louder.]{volume="+6dB"} ...500ms
[This is three decibels quieter.]{volume="-3dB"}
"""

PITCH_SCRIPT = """
[This is extra low pitch.]{pitch="x-low"} ...500ms
[This is low pitch.]{pitch="low"} ...500ms
[This is medium pitch.]{pitch="medium"} ...500ms
[This is high pitch.]{pitch="high"} ...500ms
[This is extra high pitch.]{pitch="x-high"} ...500ms
[This is ten percent higher.]{pitch="+10%"} ...500ms
[This is ten percent lower.]{pitch="-10%"}
"""

RATE_SCRIPT = """
[This is extra slow.]{rate="x-slow"} ...500ms
[This is slow.]{rate="slow"} ...500ms
[This is medium speed.]{rate="medium"} ...500ms
[This is fast.]{rate="fast"} ...500ms
[This is extra fast.]{rate="x-fast"} ...500ms
[This is twenty percent faster.]{rate="+20%"} ...500ms
[This is twenty percent slower.]{rate="-20%"}
"""

COMBINED_SCRIPT = """
[This is loud, high pitched, and fast.]{volume="loud" pitch="high" rate="fast"} ...800ms
[This is soft, low pitched, and slow.]{volume="soft" pitch="low" rate="slow"} ...800ms
[This combines relative prosody values.]{volume="+6dB" pitch="+10%" rate="+20%"}
"""


def render(pipe: KokoroPipeline, label: str, text: str, output: Path) -> None:
    """Render one SSMD prosody example and write it to a WAV file."""
    print(f"\n--- {label} ---")
    print(text.strip())
    result = pipe.run(text)
    sf.write(output, result.audio, result.sample_rate)
    duration = len(result.audio) / result.sample_rate
    print(f"Created: {output} ({duration:.2f}s)")


def main() -> None:
    """Generate volume, pitch, rate, and combined prosody examples."""
    print("=" * 70)
    print("SSMD Prosody Control Demo")
    print("=" * 70)
    print('Syntax: [text]{volume="loud" rate="fast" pitch="high"}')

    pipe = KokoroPipeline(
        PipelineConfig(
            voice=VOICE,
            generation=GenerationConfig(
                lang=LANG,
                pause_mode="manual",
            ),
        )
    )

    render(pipe, "Volume", VOLUME_SCRIPT, Path("prosody_volume_demo.wav"))
    render(pipe, "Pitch", PITCH_SCRIPT, Path("prosody_pitch_demo.wav"))
    render(pipe, "Rate", RATE_SCRIPT, Path("prosody_rate_demo.wav"))
    render(pipe, "Combined", COMBINED_SCRIPT, Path("prosody_combined_demo.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
