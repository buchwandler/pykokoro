#!/usr/bin/env python3
"""
Token/Phoneme Length Effects Demonstration.

This example explores how ultra-short sentences behave differently depending
on how they're batched and processed. It demonstrates four processing strategies:

1. Individual Processing: Each sentence converted separately (very short context)
2. Paired Processing: Two sentences combined at a time (medium context)
3. Phoneme Threshold Batching: Accumulate until minimum phoneme count (optimal batching)
4. Complete Text: All text processed at once (full context)

The example helps understand the trade-offs between processing efficiency,
prosody quality, and context preservation.

Usage:
    python examples/token_length_effects.py

Output:
    token_length_effects_comparison.wav - All four versions in one file
"""

import numpy as np
import soundfile as sf

import pykokoro

# Ultra-short dialogue sentences
SENTENCES = [
    '"Why?"',
    '"I know."',
    "He sits.",
    '"Do?"',
    "She nods.",
    '"Go!"',
    "He runs.",
    '"Stop!"',
    "She calls.",
    '"Wait!"',
    "He turns.",
    '"Now?"',
    '"Yes."',
    "They walk.",
]

VOICE = "af_bella"  # American Female voice (good for expressive reading)
LANG = "en-us"
MIN_PHONEME_THRESHOLD = 50  # Minimum phonemes for Version 3


def print_separator(title: str) -> None:
    """Print a visual separator with title."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def get_batch_stats(batches: list[str], kokoro) -> dict:
    """Calculate statistics for a list of text batches."""
    phoneme_batches = [kokoro.phonemize(batch, lang=LANG) for batch in batches]
    token_batches = [kokoro.tokenizer.tokenize(p) for p in phoneme_batches]

    total_phonemes = sum(len(p) for p in phoneme_batches)
    total_tokens = sum(len(t) for t in token_batches)

    return {
        "num_batches": len(batches),
        "total_phonemes": total_phonemes,
        "total_tokens": total_tokens,
        "avg_phonemes": total_phonemes / len(batches) if batches else 0,
        "avg_tokens": total_tokens / len(batches) if batches else 0,
        "phoneme_batches": phoneme_batches,
        "token_batches": token_batches,
    }


def version1_individual(sentences: list[str]) -> list[str]:
    """Version 1: Process each sentence individually."""
    return sentences.copy()


def version2_paired(sentences: list[str]) -> list[str]:
    """Version 2: Combine sentences in pairs."""
    batches = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            batches.append(f"{sentences[i]} {sentences[i + 1]}")
        else:
            batches.append(sentences[i])
    return batches


def version3_threshold(sentences: list[str], kokoro, threshold: int) -> list[str]:
    """Version 3: Accumulate until reaching phoneme threshold."""
    batches = []
    current_batch = []
    current_phonemes = 0

    for sentence in sentences:
        # Check phoneme length of this sentence
        sentence_phonemes = kokoro.phonemize(sentence, lang=LANG)
        sentence_len = len(sentence_phonemes)

        # If adding this would exceed MAX or we've hit threshold, flush current batch
        if current_batch and (
            current_phonemes + sentence_len > 510 or current_phonemes >= threshold
        ):
            batches.append(" ".join(current_batch))
            current_batch = []
            current_phonemes = 0

        current_batch.append(sentence)
        current_phonemes += sentence_len

    # Add remaining sentences
    if current_batch:
        batches.append(" ".join(current_batch))

    return batches


def version4_complete(sentences: list[str]) -> list[str]:
    """Version 4: Process all text as one batch."""
    return [" ".join(sentences)]


def generate_version(
    version_name: str,
    batches: list[str],
    stats: dict,
    kokoro,
    all_samples: list,
    sample_rate: int,
) -> float:
    """Generate audio for one version and add to all_samples."""
    print_separator(f"{version_name}")

    print(f"Number of batches: {stats['num_batches']}")
    print(f"Total phonemes: {stats['total_phonemes']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Average phonemes per batch: {stats['avg_phonemes']:.1f}")
    print(f"Average tokens per batch: {stats['avg_tokens']:.1f}")
    print()

    print("Batches:")
    for i, (batch, phonemes, tokens) in enumerate(
        zip(batches, stats["phoneme_batches"], stats["token_batches"], strict=False), 1
    ):
        print(f"  Batch {i}:")
        print(f"    Text: {batch}")
        print(f"    Phonemes ({len(phonemes)} chars): {phonemes}")
        print(f"    Tokens: {len(tokens)}")
        print()

    # Generate filler announcement
    filler_samples, _ = kokoro.create(
        version_name,
        voice=VOICE,
        lang=LANG,
    )
    all_samples.append(filler_samples)

    # Generate audio for each batch
    version_samples = []
    for batch in batches:
        batch_samples, _ = kokoro.create(
            batch,
            voice=VOICE,
            lang=LANG,
        )
        version_samples.append(batch_samples)

    # Concatenate this version's audio
    version_audio = np.concatenate(version_samples)
    all_samples.append(version_audio)

    # Add pause between versions
    pause = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    all_samples.append(pause)

    duration = len(version_audio) / sample_rate
    print(f"Version audio duration: {duration:.2f}s")

    return duration


def main():
    """Generate token length effects comparison."""
    print("Initializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    print(f"\nVoice: {VOICE}")
    print(f"Language: {LANG}")
    print(f"Phoneme threshold for Version 3: {MIN_PHONEME_THRESHOLD}")

    # Show original sentences
    print_separator("Original Sentences")
    full_text = " ".join(SENTENCES)
    print(f"Total sentences: {len(SENTENCES)}")
    print(f"Full text: {full_text}")

    full_phonemes = kokoro.phonemize(full_text, lang=LANG)
    print(f"\nFull text phoneme length: {len(full_phonemes)} characters")
    print(f"Full text phonemes: {full_phonemes}")
    print()

    # Prepare all versions
    versions = {
        "First version: processing each sentence separately": version1_individual(SENTENCES),
        "Second version: combining sentences in pairs": version2_paired(SENTENCES),
        f"Third version: batching with {MIN_PHONEME_THRESHOLD} phoneme threshold": version3_threshold(
            SENTENCES, kokoro, MIN_PHONEME_THRESHOLD
        ),
        "Fourth version: processing all text at once": version4_complete(SENTENCES),
    }

    # Calculate statistics for all versions
    all_stats = {}
    for name, batches in versions.items():
        all_stats[name] = get_batch_stats(batches, kokoro)

    # Generate audio for all versions
    all_samples = []
    sample_rate = 24000
    version_durations = {}

    for version_name, batches in versions.items():
        duration = generate_version(
            version_name,
            batches,
            all_stats[version_name],
            kokoro,
            all_samples,
            sample_rate,
        )
        version_durations[version_name] = duration

    # Combine all audio
    print_separator("Combining All Versions")
    combined_samples = np.concatenate(all_samples)

    output_file = "token_length_effects_comparison.wav"
    sf.write(output_file, combined_samples, sample_rate)

    total_duration = len(combined_samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Total duration: {total_duration:.2f}s")

    # Summary comparison table
    print_separator("COMPARISON SUMMARY")
    print(f"{'Version':<50} {'Batches':<10} {'Avg Phonemes':<15} {'Duration':<10}")
    print("-" * 90)

    for version_name in versions:
        stats = all_stats[version_name]
        duration = version_durations[version_name]
        short_name = version_name.split(":")[0]
        print(
            f"{short_name:<50} {stats['num_batches']:<10} "
            f"{stats['avg_phonemes']:<15.1f} {duration:<10.2f}s"
        )

    print("\nKey Observations:")
    print("  • Version 1 (Individual): Maximum number of batches, shortest context")
    print("  • Version 2 (Paired): Reduced batches, better context than individual")
    print(
        f"  • Version 3 (Threshold): Optimized batching at {MIN_PHONEME_THRESHOLD} phoneme minimum"
    )
    print("  • Version 4 (Complete): Single batch, maximum context and prosody")
    print("\nListen to compare:")
    print("  - Prosody naturalness (how sentences flow together)")
    print("  - Intonation patterns (rising/falling pitch)")
    print("  - Processing efficiency (fewer batches = faster)")
    print()

    kokoro.close()
    print("Done! Listen to the file to hear the differences in prosody and naturalness.")


if __name__ == "__main__":
    main()
