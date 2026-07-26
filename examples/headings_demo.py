#!/usr/bin/env python3
"""Heading Support Demo for PyKokoro.

This demo showcases PyKokoro's automatic heading detection and pause insertion
for markdown-style headings (# ## ###). Headings receive appropriate pauses
before and after the heading text to create natural document structure in audio.

Default heading configuration:
- Level 1 (#):   300ms pause before + 300ms after + strong emphasis
- Level 2 (##):  75ms pause before + 75ms after + moderate emphasis
- Level 3 (###): 50ms pause before + 50ms after + no emphasis

The SSMD library automatically detects headings and applies these pauses
when heading_emphasis capability is enabled (on by default in PyKokoro).
"""

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Sample document with multiple heading levels
DOCUMENT_TEXT = """
# Introduction to PyKokoro

PyKokoro is a Python text-to-speech library based on the Kokoro TTS model.
It provides high-quality voice synthesis with support for multiple languages
and expressive speech features.

## Key Features

PyKokoro offers several powerful capabilities for speech synthesis:
natural-sounding voices, markdown formatting support, and flexible audio control.

### Voice Selection

The library includes multiple voice styles to choose from.
You can select voices like Sarah, Bella, or Michael depending on your needs.

### Heading Support

Headings automatically receive appropriate pauses before and after the text.
This creates natural structure when reading documents aloud.

## Getting Started

To use PyKokoro, simply create a Kokoro instance and call the create method.
The library handles all the complex details of speech synthesis for you.

# Conclusion

PyKokoro makes text-to-speech simple and accessible for Python developers.
Try it out with your own documents today!
"""


def main():
    """Generate audio with heading pauses."""
    print("Heading Support Demo")
    print("=" * 50)
    print("\nThis demo generates speech from markdown text with headings.")
    print("Listen for pauses before and after each heading level:\n")
    print("  # Level 1: 300ms pauses (strong emphasis)")
    print("  ## Level 2: 75ms pauses (moderate emphasis)")
    print("  ### Level 3: 50ms pauses (no emphasis)")
    print("\n" + "=" * 50)

    pipe = KokoroPipeline(
        PipelineConfig(voice="af_sarah", generation=GenerationConfig(lang="en-us", speed=1.0))
    )

    # Generate audio with heading pauses
    print("\nGenerating audio...")
    res = pipe.run(DOCUMENT_TEXT)
    audio, sample_rate = res.audio, res.sample_rate

    # Save to file
    output_file = "headings_demo.wav"
    sf.write(output_file, audio, sample_rate)

    # Calculate duration
    duration_seconds = len(audio) / sample_rate

    print(f"\n✓ Created {output_file}")
    print(f"  Duration: {duration_seconds:.2f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Audio samples: {len(audio):,}")

    # Estimate pause contribution
    # 2 level-1 headings: 2 * (300+300) = 1200ms
    # 1 level-2 heading:   1 * (75+75)   = 150ms
    # 2 level-3 headings:  2 * (50+50)   = 200ms
    # Total pause time: ~1.55 seconds
    print("\n  Estimated heading pauses: ~1.55 seconds")
    print("    (2 × Level 1 = 1.20s, 1 × Level 2 = 0.15s, 2 × Level 3 = 0.20s)")

    print("\n" + "=" * 50)
    print("Play the generated audio file to hear the heading pauses!")
    print("=" * 50)


if __name__ == "__main__":
    main()
