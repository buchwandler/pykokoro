#!/usr/bin/env python3
"""
Podcast-style multi-voice conversation using SSMD voice annotations.

This example demonstrates creating a podcast with multiple speakers using
SSMD voice annotations. The entire podcast is written as a single text string
with inline voice switching using the syntax: [text](voice: name)

Features demonstrated:
- Inline voice switching with SSMD annotations
- Automatic pause insertion between speakers
- Clean, readable podcast script format
- Single API call generates entire multi-voice conversation

Usage:
    python examples/podcast.py

Output:
    podcast_ssmd_demo.wav - Multi-voice podcast with automatic voice switching
"""

import soundfile as sf

import pykokoro

# Podcast script using SSMD voice annotations
# Each speaker's dialogue is wrapped in [text](voice: name) annotation
# Pauses are added with SSMD break markers: ...s (sentence pause)
# fmt: off
PODCAST_SCRIPT = """
@voice: af_sarah
Welcome to Tech Talk! I'm Sarah, and today we're diving into the fascinating world of text-to-speech technology.

@voice: am_michael
And I'm Michael! We've got an amazing episode lined up. The advances in neural TTS have been incredible lately.

@voice: af_sarah
Absolutely! And we have a special guest with us today. Please welcome our AI researcher, Nicole!

@voice: af_nicole
Thanks for having me! I'm thrilled to be here. I've been working on voice synthesis for the past five years.

@voice: am_michael
Nicole, can you tell us about the latest breakthroughs in making synthetic voices sound more natural?

@voice: af_nicole
Of course! The key innovation has been in capturing prosody and emotional nuance. Modern models like Kokoro can generate speech that's nearly indistinguishable from human voices.

@voice: af_sarah
That's fascinating! What do you see as the main applications for this technology?

@voice: af_nicole
There are so many! Audiobook production, accessibility tools, language learning, and even preserving voices of people who might lose their ability to speak.

@voice: am_michael
The accessibility angle is really compelling. Imagine being able to give a voice to those who can't speak.

@voice: af_sarah
Exactly! And with open-source models, this technology is becoming available to everyone.

@voice: af_nicole
That's what excites me most. Democratizing access to high-quality speech synthesis opens up so many possibilities.

@voice: am_michael
Well, this has been an enlightening discussion! Any final thoughts, Nicole?

@voice: af_nicole
Just that we're at an inflection point. The next few years will bring even more amazing developments. Stay curious!

@voice: af_sarah
Thank you so much for joining us, Nicole! And thank you to our listeners for tuning in.

@voice: am_michael
Until next time, keep exploring the future of technology!
"""
# fmt: on


def main():
    print("=" * 70)
    print("SSMD MULTI-VOICE PODCAST DEMO")
    print("=" * 70)

    print("\nPodcast Script (SSMD format with voice annotations):")
    print("-" * 70)
    # Show first few lines as preview
    lines = PODCAST_SCRIPT.strip().split("\n")
    for line in lines[:8]:
        if line.strip():
            print(line)
    print("...")
    speaker_count = len(
        [line for line in lines if line.strip() and line.startswith("@voice:")]
    )
    print(f"({speaker_count} speaker segments)")
    print("-" * 70)

    print("\nInitializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    print("\nGenerating podcast with automatic voice switching...")
    print("Voice switching happens automatically based on SSMD annotations!")

    # Single API call generates entire multi-voice podcast
    samples, sample_rate = kokoro.create(
        PODCAST_SCRIPT,
        voice="af_sarah",  # Default voice (fallback if segment has no annotation)
        speed=1.0,
        lang="en-us",
        pause_mode="manual",  # PyKokoro controls pauses precisely
    )

    # Save to file
    output_file = "podcast_ssmd_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nSuccess! Created {output_file}")
    print(f"Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")

    kokoro.close()

    print("\n" + "=" * 70)
    print("HOW IT WORKS")
    print("=" * 70)
    print("\nSSMD Voice Annotation Syntax:")
    print("  [text](voice: name) - Speaks 'text' using voice 'name'")
    print("\nExample:")
    print("  [Hello!](voice: af_sarah) ...s [Goodbye!](voice: am_michael)")
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
    print("  1. SSMD parser extracts voice metadata from annotations")
    print("  2. Each segment is associated with its voice name")
    print("  3. AudioGenerator automatically switches voices per segment")
    print("  4. Single seamless audio output!")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
