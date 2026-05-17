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
    "Oh No.",
    "No!",
    "Yes!",
    "Help!",
    "Oh!",
    "Stop!",
    "What?",
    "Don't!",
    "One … step. In front.",
    "thing on her chest.",
    "Of.",
]
# Voice to use

VOICES = [
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
]
VOICE = "af_sarah"  # Changed from af_bella for better short sentence results

LANG = "en-us"


def print_separator(title: str) -> None:
    """Print a visual separator with title."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_sentence_with_config(
    voice: str,
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
    # Create a new Kokoro instance with the config
    kokoro_test = KokoroPipeline(
        PipelineConfig(
            voice=voice,
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
    print(f"\nTexts: {len(TEST_SENTENCES)} short-sentence samples")
    print(f"Language: {LANG}")
    print("\nNOTE: Audio duration will be similar, but QUALITY will be better")
    print("      with context-prepending. Listen to the generated files to compare!")

    # Initialize with default config
    print_separator("Testing Individual Sentences")

    kokoro = KokoroPipeline(
        PipelineConfig(voice=VOICE, generation=GenerationConfig(lang=LANG, speed=1.0))
    )
    tokenizer = Tokenizer()

    all_samples = []
    all_samples2 = []
    sample_rate = 24000

    pause = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    # Add announcement and samples to output
    announcement = "With pretexting"
    intro = kokoro.run(announcement).audio
    all_samples.extend([intro, pause])
    # Add announcement and samples to output
    announcement = "Without pretexting"
    intro2 = kokoro.run(announcement).audio
    all_samples2.extend([pause, intro2, pause])

    # Test each voice and sentence with different configurations
    for voice in VOICES:
        print(f"\nVoice: '{voice}'")
        for text in TEST_SENTENCES:
            phoneme_count = len(tokenizer.phonemize(text, lang=LANG))
            print(f"\nText: '{text}' ({phoneme_count} phonemes)")

            config_enabled = neutral_phrase_short_sentence_config()
            config_disabled = ShortSentenceConfig(enabled=False)

            samples_enabled, sr = test_sentence_with_config(
                voice, text, config_enabled, "With phrase cutting"
            )

            samples_disabled, sr = test_sentence_with_config(
                voice, text, config_disabled, "Without phrase cutting"
            )
            pause = np.zeros(int(sr * 0.1), dtype=np.float32)
            all_samples.extend([samples_enabled, pause])
            all_samples2.extend([samples_disabled, pause])

    # Save combined audio
    print_separator("Saving Combined Audio")

    combined_samples = np.concatenate(all_samples + all_samples2)
    output_file = "short_sentence_voices_demo.wav"
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


if __name__ == "__main__":
    main()
