#!/usr/bin/env python3
"""
Phoneme comparison between pykokoro and misaki G2P systems.

This script compares phoneme outputs from two different
G2P (grapheme-to-phoneme) systems:
1. pykokoro's kokorog2p (dictionary + espeak fallback)
2. misaki's G2P with espeak-ng fallback

When differences are found, it generates:
- A detailed text report of all differences
- An audio file narrating the differences
- A JSON file with structured comparison data

Usage:
    python examples/phoneme_diff_comparison.py [input_file.txt]

    If no input file is provided, a generated test text will be used.

Arguments:
    input_file.txt - Optional text file to analyze (default: use built-in test text)

Output:
    phoneme_diff_report.txt - Detailed text report
    phoneme_diff_report.wav - Audio narration of differences
    phoneme_diff_data.json - Structured JSON data

Examples:
    # Use default test text
    python examples/phoneme_diff_comparison.py

    # Analyze a specific text file
    python examples/phoneme_diff_comparison.py my_text.txt
"""

import json
import re
import sys

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

import pykokoro

# Import misaki if available
try:
    import misaki.en
    import misaki.espeak  # noqa: F401

    MISAKI_AVAILABLE = True
except ImportError:
    MISAKI_AVAILABLE = False
    print("WARNING: misaki not available. Install with: pip install misaki")


# Default test text with various challenging cases
DEFAULT_TEST_TEXT = """
Phoneme Comparison Test Text

Hello world! This is a comprehensive test of phoneme generation.

Common Contractions:
I don't think you'll understand what I'm saying.
She wouldn't've known about that issue.
We can't believe they're already here!
That's exactly what I thought you'd say.

Difficult Words:
The choir sang beautifully in the cathedral.
Colonel Smith scheduled a rendezvous at eight o'clock.
She received a receipt for the purchase.
The psychologist studied psychology thoroughly.

Homographs (same spelling, different pronunciation):
I read the book yesterday, and I will read another today.
The wind will wind around the mountain.
They were too close to the door to close it.
I shed a tear when I saw the tear in my jacket.

Informal Speech:
I'm gonna go to the store later.
You wanna come with me?
We gotta finish this project soon.
Let me help you with that.

Numbers and Special Cases:
It's 10 o'clock in the evening.
The 1990s were a great decade.
Dr. Johnson will see you at 3:30 PM.
She scored 95% on the test.

Mixed Challenges:
"Don't you think," she asked, "that we should've left earlier?"
The children's playground was closed for maintenance.
He couldn't've imagined a better outcome!
They're planning to celebrate their 25th anniversary.

This text includes various linguistic challenges to test phoneme generation accuracy
across different G2P systems, including contractions, homographs, informal speech,
and complex word structures.
"""


# Test text with various challenging cases
TEST_TEXT = """
Hello world! This is a test.
I don't think you'll understand.
She wouldn't've known about that.
The book is on the table.
I'm gonna go to the store.
Can't we just be friends?
That's what I thought you'd say.
"""


def split_into_words_with_positions(text):
    """Split text into words with line and position information.

    Returns:
        List of tuples: (word, line_num, word_position_in_line, char_position)
    """
    words_with_pos = []
    lines = text.split("\n")
    char_position = 0

    for line_num, line in enumerate(lines, 1):
        # Find all words in the line with their positions
        word_matches = re.finditer(r"\b\w+(?:\'[a-z]+)?\b", line, re.IGNORECASE)
        word_position = 0

        for word_position, match in enumerate(word_matches, start=1):
            word = match.group()
            words_with_pos.append((word, line_num, word_position, char_position + match.start()))

        char_position += len(line) + 1  # +1 for newline

    return words_with_pos


def get_pykokoro_phonemes(text, lang="en-us"):
    """Get phonemes from pykokoro's kokorog2p system.

    Returns:
        List of tuples: (word, phonemes)
    """
    from pykokoro.tokenizer import Tokenizer

    tokenizer = Tokenizer()

    # Extract words from text (same regex as misaki)
    words = re.findall(r"\b\w+(?:\'[a-z]+)?\b", text, re.IGNORECASE)

    word_phonemes = []
    for word in words:
        try:
            result = tokenizer.phonemize_detailed(word, lang=lang)
            phonemes = result.phonemes  # Use the full phonemes string
            word_phonemes.append((word, phonemes))
        except Exception as e:
            print(f"Warning: Error processing '{word}' with pykokoro: {e}")
            word_phonemes.append((word, ""))

    return word_phonemes


def get_misaki_phonemes(text, british=False):
    """Get phonemes from misaki G2P system.

    Returns:
        List of tuples: (word, phonemes)
    """
    if not MISAKI_AVAILABLE:
        return []

    # Import here to avoid issues when misaki is not available
    from misaki import en, espeak

    fallback = espeak.EspeakFallback(british=british)
    g2p = en.G2P(trf=False, british=british, fallback=fallback)

    # Extract words from text
    words = re.findall(r"\b\w+(?:\'[a-z]+)?\b", text, re.IGNORECASE)

    word_phonemes = []
    for word in words:
        try:
            phonemes, _ = g2p(word)
            word_phonemes.append((word, phonemes))
        except Exception as e:
            print(f"Warning: Error processing '{word}' with misaki: {e}")
            word_phonemes.append((word, ""))

    return word_phonemes


def normalize_phonemes(phonemes):
    """Normalize phoneme string for comparison.

    Removes spaces, converts to lowercase for comparison.
    """
    if not phonemes:
        return ""
    return phonemes.strip().lower()


def compare_phonemes(pykokoro_phonemes, misaki_phonemes, words_with_pos):
    """Compare phoneme outputs and find differences.

    Args:
        pykokoro_phonemes: List of (word, phonemes) from pykokoro
        misaki_phonemes: List of (word, phonemes) from misaki
        words_with_pos: List of (word, line_num, word_pos, char_pos)

    Returns:
        List of difference dictionaries
    """
    differences = []

    # Create lookup maps
    pykokoro_map = {word.lower(): phonemes for word, phonemes in pykokoro_phonemes}
    misaki_map = {word.lower(): phonemes for word, phonemes in misaki_phonemes}

    # Compare each word
    for word, line_num, word_pos, char_pos in words_with_pos:
        word_lower = word.lower()

        pykokoro_ph = pykokoro_map.get(word_lower, "")
        misaki_ph = misaki_map.get(word_lower, "")

        # Normalize for comparison
        pykokoro_norm = normalize_phonemes(pykokoro_ph)
        misaki_norm = normalize_phonemes(misaki_ph)

        # Check if different
        if pykokoro_norm != misaki_norm:
            differences.append(
                {
                    "word": word,
                    "line": line_num,
                    "position": word_pos,
                    "char_position": char_pos,
                    "pykokoro_phonemes": pykokoro_ph,
                    "misaki_phonemes": misaki_ph,
                    "pykokoro_normalized": pykokoro_norm,
                    "misaki_normalized": misaki_norm,
                }
            )

    return differences


def generate_diff_report(differences, text):
    """Generate a text report of phoneme differences.

    Returns:
        String containing the formatted report
    """
    if not differences:
        return "No phoneme differences found! Both systems produced identical results."

    report_lines = [
        "=" * 80,
        "PHONEME COMPARISON REPORT",
        "pykokoro (kokorog2p) vs misaki (G2P + espeak)",
        "=" * 80,
        "",
        f"Total differences found: {len(differences)}",
        "",
        "=" * 80,
        "DETAILED DIFFERENCES",
        "=" * 80,
        "",
    ]

    for i, diff in enumerate(differences, 1):
        report_lines.extend(
            [
                f"Difference {i}:",
                f"  Word: '{diff['word']}'",
                f"  Location: Line {diff['line']}, Word position {diff['position']}",
                f"  pykokoro phonemes: {diff['pykokoro_phonemes']}",
                f"  misaki phonemes:   {diff['misaki_phonemes']}",
                "",
            ]
        )

    report_lines.extend(
        [
            "=" * 80,
            "SUMMARY BY WORD",
            "=" * 80,
            "",
        ]
    )

    # Group by word
    word_diffs = {}
    for diff in differences:
        word = diff["word"].lower()
        if word not in word_diffs:
            word_diffs[word] = []
        word_diffs[word].append(diff)

    for word in sorted(word_diffs.keys()):
        diffs = word_diffs[word]
        report_lines.append(f"Word: '{word}' - {len(diffs)} occurrence(s)")
        first_diff = diffs[0]
        report_lines.append(f"  pykokoro: {first_diff['pykokoro_phonemes']}")
        report_lines.append(f"  misaki:   {first_diff['misaki_phonemes']}")
        report_lines.append("")

    return "\n".join(report_lines)


def generate_audio_narration(differences, output_file=None):
    """Generate audio narration of the phoneme differences.

    Creates audio where you can hear each word pronounced using the actual
    phoneme strings from both systems, so you can hear the difference.

    Args:
        differences: List of difference dictionaries
        output_file: Path to save audio file
    """
    if output_file is None:
        output_file = artifact_path("phoneme_diff_report.wav")
    import numpy as np

    if not differences:
        narration = """
        Phoneme comparison complete.
        Both systems produced identical phoneme outputs for all words in the test text.
        """
        kokoro = pykokoro.Kokoro()
        samples, sample_rate = kokoro.create(
            narration,
            voice="af_bella",
            speed=1.0,
            lang="en-us",
        )
        sf.write(output_file, samples, sample_rate)
        kokoro.close()
    else:
        # Generate audio that demonstrates the actual pronunciation differences
        print("\nGenerating audio narration with phoneme demonstrations...")
        kokoro = pykokoro.Kokoro()

        # Filter to only first occurrence of each unique word (case-insensitive)
        seen_words = set()
        unique_differences = []
        for diff in differences:
            word_lower = diff["word"].lower()
            if word_lower not in seen_words:
                seen_words.add(word_lower)
                unique_differences.append(diff)

        # Introduction
        word_count = len(unique_differences)
        intro_text = (
            f"Phoneme comparison report. Found {len(differences)} differences "
            f"between pykokoro and misaki across {word_count} unique words. "
            "You will hear each unique word once."
        )
        intro_samples, sample_rate = kokoro.create(
            intro_text, voice="af_bella", speed=1.0, lang="en-us"
        )

        all_samples = [intro_samples]
        silence = np.zeros(int(sample_rate * 0.5))  # 0.5 second pause

        for i, diff in enumerate(unique_differences, 1):
            word = diff["word"]
            line = diff["line"]
            pos = diff["position"]
            pykokoro_ph = diff["pykokoro_phonemes"]
            misaki_ph = diff["misaki_phonemes"]

            # Introduction for this difference
            intro = f"Difference {i}. Line {line}, word {pos}. The word: {word}."
            intro_s, _ = kokoro.create(intro, voice="af_bella", speed=1.0, lang="en-us")
            all_samples.append(silence)
            all_samples.append(intro_s)

            # Pronounce using pykokoro phonemes directly
            label1 = "Pykokoro:"
            label1_s, _ = kokoro.create(label1, voice="af_bella", speed=1.0, lang="en-us")
            all_samples.append(silence)
            all_samples.append(label1_s)

            # Pronounce using pykokoro phonemes directly
            word1_s, _ = kokoro.create(
                pykokoro_ph, voice="af_bella", speed=1.0, lang="en-us", is_phonemes=True
            )
            all_samples.append(word1_s)

            # Pronounce using misaki phonemes
            label2 = "Misaki:"
            label2_s, _ = kokoro.create(label2, voice="af_bella", speed=1.0, lang="en-us")
            all_samples.append(silence)
            all_samples.append(label2_s)

            # Pronounce using misaki phonemes directly
            word2_s, _ = kokoro.create(
                misaki_ph, voice="af_bella", speed=1.0, lang="en-us", is_phonemes=True
            )
            all_samples.append(word2_s)

        # Summary
        summary = (
            f"End of report. You heard {len(unique_differences)} unique words "
            "with phoneme differences."
        )
        summary_s, _ = kokoro.create(summary, voice="af_bella", speed=1.0, lang="en-us")
        all_samples.append(silence)
        all_samples.append(summary_s)

        # Concatenate all audio
        samples = np.concatenate(all_samples)
        sf.write(output_file, samples, sample_rate)
        kokoro.close()

    duration = len(samples) / sample_rate
    print(f"Audio narration saved to: {output_file}")
    print(f"Duration: {duration:.2f} seconds")


def load_text_from_file(filepath):
    """Load text from a file.

    Args:
        filepath: Path to text file

    Returns:
        String containing file contents
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"ERROR: Could not read file {filepath}: {e}")
        return None


def main():
    """Main function to compare phonemes and generate reports."""
    print("=" * 80)
    print("PHONEME COMPARISON: pykokoro vs misaki")
    print("=" * 80)

    if not MISAKI_AVAILABLE:
        print("\nERROR: misaki is not installed!")
        print("Please install it with: pip install misaki")
        return

    # Check for command-line argument (input file)
    input_file = None
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        print(f"\nInput file: {input_file}")
        text = load_text_from_file(input_file)
        if text is None:
            print("\nFalling back to default test text...")
            text = DEFAULT_TEST_TEXT.strip()
    else:
        print("\nNo input file provided. Using default test text.")
        print("(To analyze a custom file, run: python phoneme_diff_comparison.py your_file.txt)")
        text = DEFAULT_TEST_TEXT.strip()

    print("\nTest Text:")
    print("-" * 80)
    print(text)
    print("-" * 80)

    # Get word positions
    print("\nExtracting word positions...")
    words_with_pos = split_into_words_with_positions(text)
    print(f"Found {len(words_with_pos)} words")

    # Get phonemes from both systems
    print("\nGetting phonemes from pykokoro (kokorog2p)...")
    pykokoro_phonemes = get_pykokoro_phonemes(text, lang="en-us")
    print(f"pykokoro: {len(pykokoro_phonemes)} words processed")

    print("\nGetting phonemes from misaki (G2P + espeak)...")
    misaki_phonemes = get_misaki_phonemes(text, british=False)
    print(f"misaki: {len(misaki_phonemes)} words processed")

    # Compare
    print("\nComparing phonemes...")
    differences = compare_phonemes(pykokoro_phonemes, misaki_phonemes, words_with_pos)

    print(f"\nFound {len(differences)} differences")

    # Generate text report
    print("\nGenerating text report...")
    report = generate_diff_report(differences, text)

    # Save text report
    report_file = artifact_path("phoneme_diff_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Text report saved to: {report_file}")

    # Print report to console
    print("\n" + report)

    # Save JSON data
    json_file = artifact_path("phoneme_diff_data.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_text": text,
                "total_words": len(words_with_pos),
                "total_differences": len(differences),
                "differences": differences,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nJSON data saved to: {json_file}")

    # Generate audio narration
    generate_audio_narration(differences, "phoneme_diff_report.wav")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  1. {report_file} - Detailed text report")
    print(f"  2. {json_file} - Structured JSON data")
    print("  3. phoneme_diff_report.wav - Audio narration")


if __name__ == "__main__":
    main()
