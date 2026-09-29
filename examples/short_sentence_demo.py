#!/usr/bin/env python3
"""Compare supported short-sentence modes using only the public configuration API.

Run with ``python examples/short_sentence_demo.py``. The combined WAV is
caller-side example composition, not a PyKokoro composition API; synthesis may
install model assets.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import (
    GenerationConfig,
    KokoroSynthesizer,
    ShortSentenceConfig,
    SynthesisConfig,
)

VOICE = "bm_fable"
LANG = "en-us"
SENTENCES = ("Hi!", "Why?", "No.", "Yes!", "Help!", "One step at a time.")


def render_section(label: str, config: ShortSentenceConfig) -> tuple[np.ndarray, int]:
    synthesis_config = SynthesisConfig(
        voice=VOICE,
        generation=GenerationConfig(lang=LANG),
        short_sentence_config=config,
    )
    samples: list[np.ndarray] = []
    sample_rate: int | None = None
    with KokoroSynthesizer(synthesis_config) as synthesizer:
        for text in (label, *SENTENCES):
            rendered = synthesizer.synthesize_text(text, language=LANG, voice=VOICE)
            if sample_rate is not None and rendered.sample_rate != sample_rate:
                raise ValueError("Sample rate changed within one comparison section")
            sample_rate = rendered.sample_rate
            samples.append(np.asarray(rendered.audio, dtype=np.float32).reshape(-1))
            samples.append(np.zeros(round(sample_rate * 0.1), dtype=np.float32))
    if sample_rate is None:
        raise ValueError("No audio was rendered")
    return np.concatenate(samples), sample_rate


def main() -> None:
    configurations = (
        ("Disabled", ShortSentenceConfig(enabled=False)),
        ("Wrap", ShortSentenceConfig(resolve_mode="wrap")),
        ("Phrase", ShortSentenceConfig(resolve_mode="phrase")),
        ("Randomized phrase", ShortSentenceConfig(resolve_mode="randomized-phrase")),
    )
    sections: list[np.ndarray] = []
    sample_rate: int | None = None
    for label, config in configurations:
        print(f"Rendering short-sentence mode: {label}")
        audio, current_rate = render_section(label, config)
        if sample_rate is not None and current_rate != sample_rate:
            raise ValueError("Sample rate changed between comparison sections")
        sample_rate = current_rate
        sections.extend((audio, np.zeros(round(current_rate * 0.5), dtype=np.float32)))

    if sample_rate is None:
        raise ValueError("No sections were rendered")
    output = artifact_path("short_sentence_demo.wav")
    sf.write(output, np.concatenate(sections), sample_rate, subtype="FLOAT")
    print(f"Wrote caller-composed comparison recording to {output}")


if __name__ == "__main__":
    main()
