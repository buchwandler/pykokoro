#!/usr/bin/env python3
"""
Short Sentence Handling Demonstration.

This example demonstrates how pykokoro automatically handles short sentences
using phrase-based context. Short phoneme sequences (like "Why?" = 3
phonemes) can produce poor audio quality when processed individually. The
short-sentence handler provides more context for natural prosody.

The technique works as follows:
1. Short sentences are detected based on phoneme length
2. The configured resolve mode adds context around the short segment
3. Phrase modes synthesize a context phrase
4. The extra context is cut away when boundaries are confident

This happens automatically during audio generation - no manual configuration
is needed. However, you can customize the behavior using ShortSentenceConfig.

Usage:
    python examples/optimal_phoneme_length_demo.py

Output:
    short_sentence_demo.wav - Audio demonstrating short sentence handling
    Detailed console output showing how short sentences are processed
"""

import soundfile as sf

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.types import Trace

# Dialogue with mix of very short and normal sentences
DIALOGUE_TEXT = """
"Why?" she asked.

"Do it!" he commanded.

"Go!" they shouted.

"I know." she whispered.

He sits quietly.

She nods slowly.

"Really?" he questioned.

"Yes." she confirmed.

"The quick brown fox jumps over the lazy dog." he said with a smile.

"That's wonderful news!" she exclaimed happily.
"""

VOICE = "af_bella"  # American Female voice
LANG = "en-us"


def print_separator(title: str) -> None:
    """Print a visual separator with title."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def analyze_segments(segments: list, title: str) -> dict:
    """Analyze and print segment statistics."""
    print_separator(title)

    phoneme_counts = []
    total_phonemes = 0
    short_count = 0

    # Default threshold for "short"
    min_threshold = 30

    for i, seg in enumerate(segments, 1):
        phoneme_length = len(seg.phonemes)
        phoneme_counts.append(phoneme_length)
        total_phonemes += phoneme_length

        is_short = phoneme_length < min_threshold
        if is_short:
            short_count += 1

        # Show first 5 and last 1 segments
        if i <= 5 or i == len(segments):
            short_marker = " [SHORT - will use short-sentence handling]" if is_short else ""
            print(f"\nSegment {i}:{short_marker}")
            print(f"  Text: '{seg.text[:60]}{'...' if len(seg.text) > 60 else ''}'")
            print(f"  Phonemes: {phoneme_length} chars")
            if i == 5 and len(segments) > 6:
                print(f"\n  ... ({len(segments) - 6} more segments) ...")

    avg_phonemes = total_phonemes / len(segments) if segments else 0
    min_phonemes = min(phoneme_counts) if phoneme_counts else 0
    max_phonemes = max(phoneme_counts) if phoneme_counts else 0

    print("\nStatistics:")
    print(f"  Total segments: {len(segments)}")
    print(f"  Short segments (<{min_threshold} phonemes): {short_count}")
    print(f"  Average phonemes per segment: {avg_phonemes:.1f}")
    print(f"  Min/Max phonemes: {min_phonemes}/{max_phonemes}")

    return {
        "segments": len(segments),
        "short_count": short_count,
        "total_phonemes": total_phonemes,
        "avg_phonemes": avg_phonemes,
        "min_phonemes": min_phonemes,
        "max_phonemes": max_phonemes,
        "phoneme_counts": phoneme_counts,
    }


def main():
    """Generate demo showcasing automatic short sentence handling."""
    print_separator("SHORT SENTENCE HANDLING DEMONSTRATION")
    print("\nThis demo shows how pykokoro automatically handles short sentences")
    print("using phrase-based context for improved audio quality.")
    print("\nHow it works:")
    print("  1. Short segments are detected based on phoneme length")
    print("  2. The configured resolve mode adds context")
    print("  3. Phrase modes synthesize a context phrase")
    print("  4. Extra context is cut away when boundaries are confident")
    print("\nThis happens automatically - no configuration needed!")

    # Create pipeline with default settings (automatic short sentence handling)
    print_separator("Creating Pipeline Instance")
    cfg = PipelineConfig(
        voice=VOICE,
        generation=GenerationConfig(lang=LANG, speed=1.0, pause_mode="tts"),
    )
    pipe = KokoroPipeline(cfg)
    print("Pipeline created with default ShortSentenceConfig:")
    print("  min_phoneme_length: 30 (segments below this use short-sentence handling)")
    print("  resolve_mode: randomized-phrase")
    print("  phrase_fallback_tries: 5")

    # Analyze the text to show which segments are short
    print_separator("Analyzing Text Segments")
    trace = Trace()
    doc = SsmdDocumentParser().parse(DIALOGUE_TEXT, cfg, trace)
    segments = KokoroG2PAdapter().phonemize(doc.segments, doc, cfg, trace)
    stats = analyze_segments(segments, "Segment Analysis (pause_mode='tts')")

    # Generate audio with automatic short sentence handling
    print_separator("Generating Audio")
    print("\nShort segments will automatically use short-sentence handling...")
    result = pipe.run(DIALOGUE_TEXT)
    samples = result.audio
    sample_rate = result.sample_rate

    duration = len(samples) / sample_rate
    print("\nAudio generated successfully!")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")

    # Save the audio
    output_file = "optimal_phoneme_length_demo.wav"
    sf.write(output_file, samples, sample_rate)
    print(f"\nSaved to: {output_file}")

    # Show how to customize the config
    print_separator("CUSTOMIZING SHORT SENTENCE HANDLING")
    print("\nYou can customize the behavior with ShortSentenceConfig:")
    print("""
    from pykokoro.short_sentence_handler import ShortSentenceConfig

    # More aggressive short sentence handling
    config = ShortSentenceConfig(
        min_phoneme_length=50,    # Treat segments <50 as short
        resolve_mode="randomized-phrase",
        phrase_fallback_tries=5,
    )

    pipe = KokoroPipeline(PipelineConfig(short_sentence_config=config))
    """)

    print("\nTo disable short sentence handling entirely:")
    print("""
    config = ShortSentenceConfig(
        min_phoneme_length=0,  # No segment is considered "short"
    )
    pipe = KokoroPipeline(PipelineConfig(short_sentence_config=config))
    """)

    # Summary
    print_separator("SUMMARY")
    print(f"\nProcessed {stats['segments']} segments total")
    print(f"  - {stats['short_count']} short segments (used short-sentence handling)")
    print(f"  - {stats['segments'] - stats['short_count']} normal segments")
    print("\nBenefits of short-sentence handling:")
    print("  - Better prosody for short utterances")
    print("  - More natural intonation")
    print("  - Consistent audio quality across all segment lengths")
    print("\nListen to the WAV file to hear the natural-sounding short sentences!")

    pipe.close()


if __name__ == "__main__":
    main()
