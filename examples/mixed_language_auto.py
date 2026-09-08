#!/usr/bin/env python3
"""Fixed-model multilingual pronunciation with KokoroG2P routing."""

from __future__ import annotations

from typing import Any

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import GenerationConfig, KokoroPipeline, LanguageDetectionConfig, PipelineConfig

SOURCE = "Die Manpowerdiskussion wird gecancelt, du kannst das File downloaden."


def print_phonemes(label: str, result: Any) -> None:
    """Print the text, detected language, and phonemes for each segment."""
    print(f"\n{label} phonemes:")
    for segment in result.phoneme_segments:
        print(f"  {segment.text!r} ({segment.lang}): {segment.phonemes}")


def main() -> None:
    """Render and play the sentence with language detection off and on."""
    detection_off = LanguageDetectionConfig(mode="off")
    detection_auto = LanguageDetectionConfig(mode="auto", languages=("de", "en"))
    config = PipelineConfig(
        voice="martin",
        generation=GenerationConfig(lang="de"),
        language_detection=detection_off,
    )
    with KokoroPipeline(config) as pipe:
        for label, detection, filename in (
            ("Language detection off", detection_off, "mixed_language_off.wav"),
            ("Language detection on", detection_auto, "mixed_language_auto.wav"),
        ):
            result = pipe.run(SOURCE, language_detection=detection)
            print_phonemes(label, result)
            output_path = artifact_path(filename)
            sf.write(output_path, result.audio, result.sample_rate)
            print(f"Created {output_path}")
            print(f"Playing {label.lower()}...")
            result.play()


if __name__ == "__main__":
    main()
