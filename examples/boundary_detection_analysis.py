#!/usr/bin/env python3
"""Boundary Detection Analysis Example

This example demonstrates how to view the detailed output of the multi-feature
boundary detection algorithm used for short sentence handling in PyKokoro.

The algorithm analyzes "Two. {word}" audio to find the pause between context
and target word using:
1. Short-Time Energy (STE)
2. Zero Crossing Rate (ZCR)
3. Spectral Flux
4. Combined feature valleys

To see the full analysis, simply run PyKokoro with DEBUG logging enabled
on a short single-word sentence.
"""

from __future__ import annotations

import logging

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.short_sentence_handler import ShortSentenceConfig


def analyze_short_sentence(text: str = "Hi!", voice: str = "af_sarah"):
    """Analyze boundary detection for a short word with full debug output.

    Args:
        text: Short single-word text to analyze (default: "Hi!")
        voice: Voice to use for generation (default: "af_sarah")
    """
    print("=" * 80)
    print(f"BOUNDARY DETECTION ANALYSIS: '{text}'")
    print("=" * 80)
    print("\nThis will show the complete multi-feature boundary detection process")
    print(f"for the short single-word sentence: '{text}'")
    print("\nThe algorithm will:")
    print(f"  1. Generate audio for 'Two. {text}'")
    print("  2. Frame the signal (20ms frames, 10ms hop)")
    print("  3. Extract Short-Time Energy (STE), Zero Crossing Rate (ZCR), Spectral Flux")
    print("  4. Smooth features with median filter")
    print("  5. Combine features to find valleys")
    print("  6. Detect speech boundaries dynamically")
    print("  7. Find first valley after speech start (boundary candidate)")
    print("  8. Select deepest valley in search range")
    print("  9. Cut audio at detected boundary")
    print("\nWatch for these key log lines:")
    print("  - 'Speech boundaries detected' - shows where speech starts/ends")
    print("  - 'Found first valley' - shows detected boundary candidate after 'Two.'")
    print("  - 'Top 3 deepest valleys' - candidate boundaries")
    print("  - 'Selected deepest valley' - final choice")
    print("  - 'Cut point' - where audio will be trimmed")
    print("\n" + "=" * 80)
    print("DEBUG OUTPUT:")
    print("=" * 80 + "\n")

    # Create Kokoro with short sentence handling and DEBUG logging
    config = ShortSentenceConfig(
        enabled=True,
        min_phoneme_length=10,
        phoneme_pretext="—",
    )

    pipe = KokoroPipeline(
        PipelineConfig(
            voice=voice,
            generation=GenerationConfig(lang="en-us", speed=1.0),
            short_sentence_config=config,
        )
    )

    # Generate audio - this will trigger the boundary detection
    res = pipe.run(text)
    audio, sample_rate = res.audio, res.sample_rate

    duration = len(audio) / sample_rate

    print("\n" + "=" * 80)
    print("RESULT:")
    print("=" * 80)
    print(f"  Input text: '{text}'")
    print(f"  Output duration: {duration:.3f}s")
    print(f"  Output samples: {len(audio)}")
    print(f"  Sample rate: {sample_rate}Hz")
    print(f"\nThe output audio should contain ONLY '{text}' with no 'Two.' audible.")
    print("=" * 80 + "\n")

    return audio, sample_rate


if __name__ == "__main__":
    # Enable DEBUG logging to see all details
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)-8s [%(name)s] - %(message)s")

    # Example 1: Analyze "Hi!"
    print("\n\n")
    analyze_short_sentence("Hi!", voice="af_sarah")

    print("\n\n\n")

    # Example 2: Analyze a different word
    analyze_short_sentence("Stop!", voice="af_sarah")

    print("\n\nTo understand the values:")
    print("  - STE (Short-Time Energy): Higher = louder, Lower = quieter/silence")
    print("  - ZCR (Zero Crossing Rate): Higher = noisy/unvoiced, Lower = voiced/tonal")
    print("  - Spectral Flux: Higher = spectral change, Lower = stable spectrum")
    print("  - Combined: Lower value = better boundary candidate (valley)")
    print("  - Depth: How deep the valley is (lower = deeper = better boundary)")
