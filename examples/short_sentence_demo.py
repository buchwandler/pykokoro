#!/usr/bin/env python3
"""
Short Sentence Handler Demonstration.

This example demonstrates the cross-correlation extraction technique used by PyKokoro
to improve audio quality for very short sentences.

The short sentence handler:
1. Detects sentences with fewer phonemes than a threshold (default: 30)
2. Generates the short sentence alone (poor quality, but needed for pattern)
3. Generates context + short sentence together (good quality with natural prosody)
4. Uses cross-correlation to find where the short sentence appears in combined audio
5. Extracts that portion from the combined audio (maintains high quality)

This produces higher-quality audio because neural TTS models typically need
more context to produce natural-sounding speech with proper prosody and intonation.
The cross-correlation approach is robust and doesn't depend on silence gap detection.

Usage:
    python examples/short_sentence_demo.py

Output:
    short_sentence_demo.wav - Audio demonstrating short sentence handling
    Detailed console output showing processing steps
"""

import numpy as np
import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.short_sentence_handler import (
    PhraseResolveMode,
    ShortSentenceConfig,
)
from pykokoro.tokenizer import Tokenizer

# Enable debug logging to see detailed processing information
# logging.basicConfig(
#    level=logging.DEBUG, format="%(levelname)s [%(name)s] - %(message)s"
# )

# Test sentences of varying lengths
TEST_SENTENCES = [
    # Very short (will trigger short sentence handling)
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


# Voice to use
# Note: Different voices may produce slightly different durations due to varying
# speaking rates. af_sarah is recommended for testing the short sentence handler.
VOICE = "bm_fable"  # Changed from af_bella for better short sentence results
LANG = "en-us"


def print_separator(title: str) -> None:
    """Print a visual separator with title."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_sentence_with_config(
    text: str,
    config: ShortSentenceConfig | None,
    config_name: str,
) -> tuple[np.ndarray, int]:
    """Generate audio for a sentence with a specific config.

    Args:
        kokoro: Kokoro instance
        text: Text to generate
        config: Short sentence configuration (or None to disable)
        config_name: Name for logging

    Returns:
        Tuple of (audio samples, sample rate)
    """
    kokoro_test = KokoroPipeline(
        PipelineConfig(
            voice=VOICE,
            generation=GenerationConfig(lang=LANG, speed=1.0),
            short_sentence_config=config,
        )
    )

    res = kokoro_test.run(text)
    samples, sr = res.audio, res.sample_rate

    print(f"  {config_name:25} -> {len(samples):6} samples ({len(samples) / sr:.3f}s)")

    return samples, sr


def main():
    """Generate audio demonstrating short sentence handling."""
    print_separator("SHORT SENTENCE HANDLER DEMONSTRATION")

    print("\nThis demo shows how PyKokoro improves audio quality for short sentences")
    print("using cross-correlation extraction with context.")
    print(f"\nVoice: {VOICE}")
    print(f"Language: {LANG}")
    print("\nNOTE: Audio duration will be similar, but QUALITY will be better")
    print("      with context-prepending. Listen to the generated files to compare!")

    # Initialize with default config
    print_separator("Testing Individual Sentences")

    kokoro = KokoroPipeline(
        PipelineConfig(voice=VOICE, generation=GenerationConfig(lang=LANG, speed=1.0))
    )
    tokenizer = Tokenizer()

    pretext_samples = []
    disabled_samples = []
    neutral_phrase_samples = []
    sample_rate = 24000

    pause = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    # Add announcement and samples to output
    announcement = "With wrap mode"
    intro = kokoro.run(announcement).audio
    pretext_samples.extend([intro, pause])
    # Add announcement and samples to output
    announcement = "Without short sentence handling"
    intro2 = kokoro.run(announcement).audio
    disabled_samples.extend([pause, intro2, pause])
    # Add announcement and samples to output
    announcement = "With phrase mode"
    intro3 = kokoro.run(announcement).audio
    neutral_phrase_samples.extend([pause, intro3, pause])

    # Test each sentence with different configurations
    for text in TEST_SENTENCES:
        phoneme_count = len(tokenizer.phonemize(text, lang=LANG))

        print(f"\nText: '{text}' ({phoneme_count} phonemes)")

        # Test with phoneme pretext wrapping only
        config_pretext = pretext_short_sentence_config()

        # Test with short sentence handling disabled
        config_disabled = ShortSentenceConfig(enabled=False)

        # Test with neutral phrase generation + cutting
        config_neutral_phrase = neutral_phrase_short_sentence_config()

        # Generate with all configs
        samples_pretext, sr = test_sentence_with_config(text, config_pretext, "With wrap mode")

        samples_disabled, sr = test_sentence_with_config(text, config_disabled, "Without handling")

        samples_neutral_phrase, sr = test_sentence_with_config(
            text, config_neutral_phrase, "With phrase mode"
        )

        pause = np.zeros(int(sr * 0.1), dtype=np.float32)

        disabled_samples.extend([samples_disabled, pause])
        pretext_samples.extend([samples_pretext, pause])
        neutral_phrase_samples.extend([samples_neutral_phrase, pause])

    # Configuration comparison
    print_separator("Configuration Comparison")

    test_text = "Why?"
    phonemes = tokenizer.phonemize(test_text, lang=LANG)
    phoneme_count = len(phonemes)

    print(f"\nTest sentence: '{test_text}' ({phoneme_count} phonemes)")
    print(f"Phonemes: {phonemes}")
    print()

    # Save combined audio
    print_separator("Saving Combined Audio")

    combined_samples = np.concatenate(disabled_samples + pretext_samples + neutral_phrase_samples)
    output_file = "short_sentence_demo.wav"
    sf.write(output_file, combined_samples, sample_rate)

    total_duration = len(combined_samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration / 60:.2f} minutes)")

    # Summary
    print_separator("SUMMARY")

    print("\nHow the Short Sentence Handler Works:")
    print("  1. Detects sentences with < min_phoneme_length phonemes")
    print("  2. Applies the configured resolve mode to the short segment")
    print("  3. Phrase modes synthesize a context phrase")
    print("  4. Cuts away the extra phrase context when boundaries are confident")

    print("\nBenefits:")
    print("  • Improved prosody and intonation for short sentences")
    print("  • More natural-sounding speech")
    print("  • Better handling of single-word sentences")

    print("\nConfiguration Options:")
    print("  • min_phoneme_length: Threshold for 'short' (default: 30)")
    print("  • resolve_mode: 'randomized-phrase' (default), 'phrase', or 'wrap'")
    print("  • enabled: Enable/disable the feature (default: True)")

    print("\nUsage:")
    print("  # Custom configuration")
    print("  config = ShortSentenceConfig(min_phoneme_length=15)")
    print("  pipe = KokoroPipeline(PipelineConfig(short_sentence_config=config))")
    print()
    print("  # Disable short sentence handling")
    print("  config = ShortSentenceConfig(enabled=False)")
    print("  pipe = KokoroPipeline(PipelineConfig(short_sentence_config=config))")

    print("\n" + "=" * 70)
    print("Listen to the WAV file to hear the difference!")
    print("=" * 70)


def neutral_phrase_short_sentence_config() -> ShortSentenceConfig:
    """Use phrase generation + cutting for short clauses in this demo."""
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


def pretext_short_sentence_config() -> ShortSentenceConfig:
    """Use phoneme pretext wrapping for the same demo range as phrase mode."""
    return ShortSentenceConfig(
        min_phoneme_length=20,
        resolve_mode="wrap",
    )


if __name__ == "__main__":
    main()
