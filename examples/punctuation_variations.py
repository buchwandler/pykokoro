#!/usr/bin/env python3
"""
Punctuation variations example using pykokoro.

This example demonstrates how different punctuation marks affect the reading
of the same sentence. It shows how periods, quotation marks, ellipses,
exclamation marks, and question marks change the prosody and pacing.

Usage:
    python examples/punctuation_variations.py

Output:
    punctuation_variations_combined.wav - All variations in one file
"""

import numpy as np
import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.tokenizer import Tokenizer

# Four variations of the same sentence with different punctuation
VARIATIONS = [
    ("1_periods", "I'm so sorry. I said. That's that's terrible."),
    ("2_quotes", '"I\'m so sorry.", I said. "That\'s that\'s terrible."'),
    ("3_ellipses", '"I\'m so sorry ...", I said. "That\'s ... that\'s ... terrible."'),
    (
        "4_exclamation_question",
        '"I\'m so sorry!", I said. "That\'s ... that\'s ...terrible?"',
    ),
]

VOICE = "af_bella"  # American Female voice (good for expressive reading)
LANG = "en-us"  # American English


def main():
    """Generate speech with different punctuation variations."""
    print("Initializing TTS engine...")
    pipe = KokoroPipeline(
        PipelineConfig(voice=VOICE, generation=GenerationConfig(lang=LANG, speed=1.0))
    )
    tokenizer = Tokenizer()

    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")
    print("\nGenerating audio for punctuation variations...\n")

    all_samples = []
    sample_rate_value = 24000  # Default, will be set by first create call

    for i, (variation_name, text) in enumerate(VARIATIONS, 1):
        print(f"=== Variation {i}: {variation_name} ===")
        print(f"Text: {text}")

        # Generate filler announcement
        filler_text = f"Variation {i}, {variation_name.replace('_', ' ')}."
        print(f"Filler: {filler_text}")

        filler_res = pipe.run(filler_text)
        filler_samples, sample_rate_value = (
            filler_res.audio,
            filler_res.sample_rate,
        )

        # Convert text to phonemes
        print("Converting to phonemes...")
        phonemes = tokenizer.phonemize(text, lang=LANG)
        print(f"Phonemes: {phonemes}")

        print("Generating audio...")
        samples_res = pipe.run(text)
        samples, sample_rate_value = samples_res.audio, samples_res.sample_rate

        duration = len(samples) / sample_rate_value
        print(f"Duration: {duration:.2f} seconds\n")

        # Add filler, then the actual variation
        all_samples.append(filler_samples)
        all_samples.append(samples)

        # Add a pause between variations (0.5 seconds of silence)
        if i < len(VARIATIONS):
            pause = np.zeros(int(sample_rate_value * 0.5), dtype=np.float32)
            all_samples.append(pause)

    # Combine all samples
    print("Combining all variations into one file...")
    combined_samples = np.concatenate(all_samples)

    output_file = artifact_path("punctuation_variations_combined.wav")
    sf.write(output_file, combined_samples, sample_rate_value)

    total_duration = len(combined_samples) / sample_rate_value
    print(f"\nCreated {output_file}")
    print(f"Total duration: {total_duration:.2f} seconds")

    print("\nDone! Listen to the combined file to compare all punctuation variations.")


if __name__ == "__main__":
    main()
