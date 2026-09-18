#!/usr/bin/env python3
"""Demonstrate span-based splitting + phonemization with the pipeline stages."""

from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser

from pykokoro import PipelineConfig
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.types import Trace


def print_separator(title: str) -> None:
    """Print a visual separator with title."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_segments(segments: list) -> None:
    """Print segment information in a readable format."""
    print_separator("Pipeline Segments")
    print(f"Total segments: {len(segments)}\n")

    for i, seg in enumerate(segments, 1):
        print(f"Segment {i}:")
        print(f"  Paragraph: {seg.paragraph_idx}")
        print(f"  Sentence:  {seg.sentence_idx}")
        print(f"  Text:      {seg.text!r}")
        print(f"  Phonemes:  {seg.phonemes}")
        print(f"  Tokens:    {len(seg.tokens)} tokens")
        print(f"  Language:  {seg.lang}")
        print(f"  Pause:     {seg.pause_after}s")
        print()


def main():
    """Run the span-based phonemization demonstration."""

    # Sample text with 3 paragraphs, each containing 3 short sentences
    text = """The sun rises in the east. Birds begin to sing. The day starts fresh.

Coffee brews in the kitchen. Toast pops from the toaster. Breakfast is almost ready.

People walk to work. Cars fill the streets. The city comes alive."""

    print("=" * 80)
    print(" SPLIT_AND_PHONEMIZE_TEXT DEMONSTRATION")
    print("=" * 80)
    print("\nOriginal Text:")
    print("-" * 80)
    print(text)
    print("-" * 80)

    cfg = PipelineConfig(voice="af_bella")
    trace = Trace()
    parser = SsmdDocumentParser()
    g2p = KokoroG2PAdapter()

    try:
        doc = parser.parse(text, cfg, trace)
        text_segments = doc.segments
        phoneme_segments = g2p.phonemize(text_segments, doc, cfg, trace)

        print_segments(phoneme_segments)

        total_chars = sum(len(seg.text) for seg in phoneme_segments)
        total_phonemes = sum(len(seg.phonemes) for seg in phoneme_segments)
        total_tokens = sum(len(seg.tokens) for seg in phoneme_segments)

        print("Summary Statistics:")
        print(f"  Total characters: {total_chars}")
        print(f"  Total phonemes:   {total_phonemes}")
        print(f"  Total tokens:     {total_tokens}")
        print(f"  Avg chars/seg:    {total_chars / len(phoneme_segments):.1f}")
        print(f"  Avg phonemes/seg: {total_phonemes / len(phoneme_segments):.1f}")

    except ImportError as e:
        print(f"\n⚠️  Error: {e}")
        print("   Sentence/clause splitting requires spaCy.")
        print("   Install with: pip install spacy")
        print("   Then: python -m spacy download en_core_web_sm")
    except Exception as e:
        print(f"\n❌ Error processing text: {e}")


if __name__ == "__main__":
    main()
