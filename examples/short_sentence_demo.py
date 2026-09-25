#!/usr/bin/env python3
"""Compare short-sentence handling modes with the request-centric synthesis API.

Run with ``python examples/short_sentence_demo.py``. The combined recording is
written to ``example-artifacts/short_sentence_demo.wav`` by default.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroSynthesizer, SynthesisConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.short_sentence_handler import (
    PhraseResolveMode,
    ShortSentenceConfig,
)

VOICE = "bm_fable"
LANG = "en-us"
TEST_SENTENCES = [
    "Hi!",
    "Why?",
    "No.",
    "No!",
    "Yes!",
    "Help!",
    "Oh!",
    "Stop!",
    "What?",
    "Don't!",
    "One … step. In front.",
    "Of.",
    "The other.",
    "Ah.",
    "Hush.",
    "Go on.",
    "Not yet.",
    "I know.",
    "Mm-hm.",
    "Mr. Vale.",
    "'Tis.",
    "Chapter 4",
    "Hermione.",
]


def neutral_phrase_short_sentence_config() -> ShortSentenceConfig:
    """Use phrase generation and timestamp-based cutting for short sentences."""
    return ShortSentenceConfig(
        resolve_modes={
            "phrase": PhraseResolveMode(
                neutral_phrase="The conversation stopped, {segment}, before someone answered.",
                end_phrase="The conversation stopped after one last reply: {segment}",
            )
        },
        min_phoneme_length=20,
        resolve_mode="phrase",
    )


def wrap_short_sentence_config() -> ShortSentenceConfig:
    """Use phoneme wrapping for short sentences."""
    return ShortSentenceConfig(min_phoneme_length=20, resolve_mode="wrap")


def render_section(
    label: str,
    config: ShortSentenceConfig,
) -> tuple[np.ndarray, int]:
    """Render one labeled set of examples using a single synthesizer instance."""
    synthesis_config = SynthesisConfig(
        voice=VOICE,
        generation=GenerationConfig(lang=LANG, speed=1.0),
        short_sentence_config=config,
    )
    samples: list[np.ndarray] = []
    sample_rate: int | None = None

    with KokoroSynthesizer(synthesis_config) as synthesizer:
        for text in [label, *TEST_SENTENCES]:
            rendered = synthesizer.synthesize_text(text, language=LANG, voice=VOICE)
            current_rate = rendered.sample_rate
            if sample_rate is not None and current_rate != sample_rate:
                raise ValueError(
                    f"Sample rate changed within section: {sample_rate} to {current_rate}"
                )
            sample_rate = current_rate
            audio = np.asarray(rendered.audio, dtype=np.float32).reshape(-1)
            samples.append(audio)
            samples.append(np.zeros(round(current_rate * 0.1), dtype=np.float32))
            if text != label:
                print(f"  {text!r}: {len(audio) / current_rate:.3f}s")

    if sample_rate is None:
        raise ValueError("No audio was rendered")
    print(f"Rendered {label!r} at {sample_rate} Hz")
    return np.concatenate(samples), sample_rate


def main() -> None:
    """Generate a recording comparing short-sentence handling modes."""
    configurations = [
        ("Without short sentence handling", ShortSentenceConfig(enabled=False)),
        ("With wrap mode", wrap_short_sentence_config()),
        ("With phrase mode", neutral_phrase_short_sentence_config()),
    ]
    sections: list[np.ndarray] = []
    sample_rate: int | None = None

    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")
    for label, config in configurations:
        print(f"\nRendering {label}...")
        audio, current_rate = render_section(label, config)
        if sample_rate is not None and current_rate != sample_rate:
            raise ValueError(
                f"Sample rate changed between sections: {sample_rate} to {current_rate}"
            )
        sample_rate = current_rate
        sections.extend((audio, np.zeros(round(current_rate * 0.5), dtype=np.float32)))

    if sample_rate is None:
        raise ValueError("No sections were rendered")

    combined = np.concatenate(sections)
    output_file = artifact_path("short_sentence_demo.wav")
    sf.write(output_file, combined, sample_rate, subtype="FLOAT")
    duration = len(combined) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Total duration: {duration:.2f}s ({duration / 60:.2f} minutes)")


if __name__ == "__main__":
    main()
