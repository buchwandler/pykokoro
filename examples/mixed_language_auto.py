#!/usr/bin/env python3
"""Compare fixed-model multilingual pronunciation across German lexica."""

from __future__ import annotations

from typing import Any

import numpy as np
import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import GenerationConfig, KokoroPipeline, LanguageDetectionConfig, PipelineConfig
from pykokoro.tokenizer import TokenizerConfig

SOURCE = "Die Manpowerdiskussion wird gecancelt, du kannst das File downloaden."
OUTPUT_FILE = "mixed_language_auto.wav"
LEXICON_SEPARATOR_SECONDS = 1.0
LEXICON_SOURCES = (
    ("de-de:gold", "gold"),
    ("de-de:crane", "crane"),
    ("de-de:espeak", "espeak"),
    ("de-de:lexhint", "lexhint"),
)


def print_phonemes(label: str, result: Any) -> None:
    """Print the text, detected language, and phonemes for each segment."""
    print(f"\n{label} phonemes:")
    for segment in result.phoneme_segments:
        print(f"  {segment.text!r} ({segment.lang}): {segment.phonemes}")


def print_language_routes(result: Any) -> None:
    """Print structured selected-lexicon routing diagnostics."""
    routes = result.document_metadata.get("g2p_language_routes", [])
    print("\nLanguage routing:")
    for route in routes:
        print(
            f"  {route['text']} [{route['char_start']}:{route['char_end']}] "
            f"default={route['default_language']} reason={route['reason']} "
            f"confidence={route['confidence']}"
        )
        for fragment in route.get("fragments", []):
            evidence = fragment.get("evidence_lexicon_id")
            if evidence is not None:
                evidence = (
                    f" evidence={evidence} kind={fragment.get('evidence_kind')} "
                    f"rating={fragment.get('evidence_rating')}"
                )
            else:
                evidence = ""
            print(f"    {fragment['text']} -> {fragment['language']} {fragment['kind']}{evidence}")


def combine_audio(results: list[Any]) -> np.ndarray:
    """Combine result audio in order with one-second silence separators."""
    first = results[0]
    sample_rate = first.sample_rate
    first_audio = np.asarray(first.audio)
    gap_samples = round(sample_rate * LEXICON_SEPARATOR_SECONDS)
    silence = np.zeros((gap_samples, *first_audio.shape[1:]), dtype=first_audio.dtype)
    parts: list[np.ndarray] = []
    for index, result in enumerate(results):
        if result.sample_rate != sample_rate:
            raise RuntimeError("Mixed-language runs returned different sample rates")
        audio = np.asarray(result.audio)
        if audio.ndim != first_audio.ndim:
            raise RuntimeError("Mixed-language runs returned incompatible audio dimensions")
        if index:
            parts.append(silence)
        parts.append(audio.astype(first_audio.dtype, copy=False))
    return np.concatenate(parts, axis=0)


def main() -> None:
    """Render eight ordered runs and write one combined WAV without playback."""
    detection_off = LanguageDetectionConfig(mode="off")
    detection_auto = LanguageDetectionConfig(mode="auto", languages=("de", "en"))
    config = PipelineConfig(
        voice="martin",
        generation=GenerationConfig(lang="de"),
        language_detection=detection_off,
    )
    results: list[Any] = []
    with KokoroPipeline(config) as pipe:
        for mode_label, detection in (
            ("Without automatic language detection", detection_off),
            ("With automatic language detection", detection_auto),
        ):
            for lexicon_id, lexicon in LEXICON_SOURCES:
                result = pipe.run(
                    SOURCE,
                    language_detection=detection,
                    tokenizer_config=TokenizerConfig(lexicons=(lexicon,)),
                )
                results.append(result)
                print_phonemes(f"{mode_label}, {lexicon_id}", result)
                if detection.mode == "auto":
                    print_language_routes(result)

    output_path = artifact_path(OUTPUT_FILE)
    sf.write(output_path, combine_audio(results), results[0].sample_rate)
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
