#!/usr/bin/env python3
"""
Podcast-style multi-voice conversation using SSMD 0.8 voice annotations.

This example demonstrates creating a podcast with multiple speakers using
SSMD voice directives. The entire podcast is written as a single text string
with one ``<div voice="name">`` block per speaker turn.

Features demonstrated:
- Portable logical roles are shown in ``ssmd_080_portable_podcast.py``.
- Block-level voice switching with SSMD directives
- Automatic pause insertion between speakers
- Clean, readable podcast script format
- Single API call generates entire multi-voice conversation

Usage:
    python examples/podcast.py

Output:
    podcast_ssmd_demo.wav - Multi-voice podcast with automatic voice switching
"""

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Podcast script using SSMD block voice directives. For a portable header-first version,
# see examples/ssmd_080_portable_podcast.py.
# Inline voice changes can use: [short phrase]{voice="af_sarah"}
# fmt: off
PODCAST_SCRIPT = """
<div voice="af_sarah">
Welcome to Tech Talk! I'm Sarah, and today we're diving into the fascinating world of text-to-speech technology.
</div>

<div voice="am_michael">
And I'm Michael! We've got an amazing episode lined up. The advances in neural TTS have been incredible lately.
</div>

<div voice="af_sarah">
Absolutely! And we have a special guest with us today. Please welcome our AI researcher, Nicole!
</div>

<div voice="af_nicole">
Thanks for having me! I'm thrilled to be here. I've been working on voice synthesis for the past five years.
</div>

<div voice="am_michael">
Nicole, can you tell us about the latest breakthroughs in making synthetic voices sound more natural?
</div>

<div voice="af_nicole">
Of course! The key innovation has been in capturing prosody and emotional nuance. Modern models like Kokoro can generate speech that's nearly indistinguishable from human voices.
</div>

<div voice="af_sarah">
That's fascinating! What do you see as the main applications for this technology?
</div>

<div voice="af_nicole">
There are so many! Audiobook production, accessibility tools, language learning, and even preserving voices of people who might lose their ability to speak.
</div>

<div voice="am_michael">
The accessibility angle is really compelling. Imagine being able to give a voice to those who can't speak.
</div>

<div voice="af_sarah">
Exactly! And with open-source models, this technology is becoming available to everyone.
</div>

<div voice="af_nicole">
That's what excites me most. Democratizing access to high-quality speech synthesis opens up so many possibilities.
</div>

<div voice="am_michael">
Well, this has been an enlightening discussion! Any final thoughts, Nicole?
</div>

<div voice="af_nicole">
Just that we're at an inflection point. The next few years will bring even more amazing developments. Stay curious!
</div>

<div voice="af_sarah">
Thank you so much for joining us, Nicole! And thank you to our listeners for tuning in.
</div>

<div voice="am_michael">
Until next time, keep exploring the future of technology!
</div>
"""
# fmt: on


def main():
    print("=" * 70)
    print("SSMD MULTI-VOICE PODCAST DEMO")
    print("=" * 70)

    print("\nPodcast Script (SSMD format with voice directives):")
    print("-" * 70)
    # Show first few lines as preview
    lines = PODCAST_SCRIPT.strip().split("\n")
    for line in lines[:8]:
        if line.strip():
            print(line)
    print("...")
    speaker_count = sum(1 for line in lines if line.lstrip().startswith("<div voice="))
    print(f"({speaker_count} speaker segments)")
    print("-" * 70)

    print("\nInitializing TTS engine...")
    pipe = KokoroPipeline(
        PipelineConfig(
            voice="af_sarah",
            generation=GenerationConfig(
                lang="en-us",
                speed=1.0,
                pause_mode="manual",
            ),
        )
    )

    print("\nGenerating podcast with automatic voice switching...")
    print("Voice switching follows the SSMD <div voice=...> directives.")

    # A single pipeline run generates the multi-voice podcast.
    result = pipe.run(PODCAST_SCRIPT)
    samples, sample_rate = result.audio, result.sample_rate

    # Save to file
    output_file = "podcast_ssmd_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nSuccess! Created {output_file}")
    print(f"Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")

    print("\n" + "=" * 70)
    print("HOW IT WORKS")
    print("=" * 70)
    print("\nSSMD Voice Syntax:")
    print('  <div voice="name">...</div> - Voice block for a speaker turn')
    print('  [text]{voice="name"} - Inline voice change for a short phrase')
    print("\nExample:")
    print('  [Hello!]{voice="af_sarah"} ...s [Goodbye!]{voice="am_michael"}')
    print("\nAvailable voices:")
    print("  - af_sarah, af_nicole, af_sky (American Female)")
    print("  - am_adam, am_michael (American Male)")
    print("  - bf_emma, bf_isabella (British Female)")
    print("  - bm_george, bm_lewis (British Male)")
    print("\nPause markers:")
    print("  ...c - Comma pause (0.3s)")
    print("  ...s - Sentence pause (0.6s)")
    print("  ...p - Paragraph pause (1.0s)")
    print("  ...500ms - Custom duration")
    print("\nProcess:")
    print("  1. SSMD parser extracts voice metadata from directives and annotations")
    print("  2. Each segment is associated with its voice name")
    print("  3. AudioGenerator automatically switches voices per segment")
    print("  4. Single seamless audio output!")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
