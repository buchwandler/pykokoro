#!/usr/bin/env python3
"""
Voice SLERP (Spherical Linear Interpolation) Demonstration.

This example demonstrates the difference between linear and spherical
interpolation (SLERP) for blending voices. SLERP produces smoother,
more natural voice transitions by interpolating along the surface of
a hypersphere rather than through its interior.

The example compares:
1. Linear interpolation (weighted average)
2. SLERP interpolation (spherical interpolation)

Usage:
    python examples/voice_slerp_demo.py

Output:
    voice_slerp_demo.wav - Audio demonstrating different interpolation methods
    Detailed console output showing voice blending results
"""

import numpy as np
import soundfile as sf

import pykokoro
from pykokoro.voice_manager import VoiceBlend

# Test phrase
TEST_TEXT = "Hello, this is a demonstration of voice interpolation techniques."

# Voices to blend
VOICE_A = "af_bella"  # American Female
VOICE_B = "am_adam"  # American Male

# Language
LANG = "en-us"


def print_separator(title: str) -> None:
    """Print a visual separator with title."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Generate audio comparing linear and SLERP voice interpolation."""
    print_separator("VOICE SLERP INTERPOLATION DEMONSTRATION")
    print("\nThis demo compares two voice blending methods:")
    print(f"  Voice A: {VOICE_A} (American Female)")
    print(f"  Voice B: {VOICE_B} (American Male)")
    print("\nInterpolation methods:")
    print("  1. Linear: Weighted average (traditional method)")
    print("  2. SLERP: Spherical linear interpolation (smoother transitions)")

    print_separator("Initializing TTS Engine")
    kokoro = pykokoro.Kokoro()

    all_samples = []
    sample_rate = 24000

    # Generate with original voices
    print_separator("Generating Original Voices")

    print(f"\n1. Voice A ({VOICE_A})...")
    samples_a, sr = kokoro.create(TEST_TEXT, voice=VOICE_A, lang=LANG)
    print(f"   Generated {len(samples_a)} samples ({len(samples_a) / sr:.2f}s)")

    # Announcement
    announcement_a = f"Voice A: {VOICE_A}."
    intro_a, _ = kokoro.create(announcement_a, voice=VOICE_A, lang=LANG)
    all_samples.extend([intro_a, samples_a])

    # Pause
    pause = np.zeros(int(sr * 0.5), dtype=np.float32)
    all_samples.append(pause)

    print(f"\n2. Voice B ({VOICE_B})...")
    samples_b, sr = kokoro.create(TEST_TEXT, voice=VOICE_B, lang=LANG)
    print(f"   Generated {len(samples_b)} samples ({len(samples_b) / sr:.2f}s)")

    # Announcement
    announcement_b = f"Voice B: {VOICE_B}."
    intro_b, _ = kokoro.create(announcement_b, voice=VOICE_B, lang=LANG)
    all_samples.extend([intro_b, samples_b])
    all_samples.append(pause)

    # Generate blended voices at different interpolation points
    print_separator("Generating Blended Voices")

    interpolation_points = [0.25, 0.5, 0.75]

    for t in interpolation_points:
        weight_a = 1 - t
        weight_b = t

        print(f"\n--- Interpolation t={t} ({weight_a * 100:.0f}% A, {weight_b * 100:.0f}% B) ---")

        # LINEAR INTERPOLATION
        print("  Linear interpolation...")
        blend_linear = VoiceBlend(
            voices=[(VOICE_A, weight_a), (VOICE_B, weight_b)], interpolation="linear"
        )
        samples_linear, _ = kokoro.create(TEST_TEXT, voice=blend_linear, lang=LANG)
        print(f"    Generated {len(samples_linear)} samples ({len(samples_linear) / sr:.2f}s)")

        # Announcement
        announcement_linear = f"Linear interpolation at {int(t * 100)} percent."
        intro_linear, _ = kokoro.create(announcement_linear, voice=VOICE_A, lang=LANG)
        all_samples.extend([intro_linear, samples_linear])
        all_samples.append(pause)

        # SLERP INTERPOLATION
        print("  SLERP interpolation...")
        blend_slerp = VoiceBlend(
            voices=[(VOICE_A, weight_a), (VOICE_B, weight_b)], interpolation="slerp"
        )
        samples_slerp, _ = kokoro.create(TEST_TEXT, voice=blend_slerp, lang=LANG)
        print(f"    Generated {len(samples_slerp)} samples ({len(samples_slerp) / sr:.2f}s)")

        # Announcement
        announcement_slerp = f"SLERP interpolation at {int(t * 100)} percent."
        intro_slerp, _ = kokoro.create(announcement_slerp, voice=VOICE_A, lang=LANG)
        all_samples.extend([intro_slerp, samples_slerp])
        all_samples.append(pause)

    # Test string format parsing with @slerp suffix
    print_separator("Testing String Format Parsing")
    print("\nTesting '@slerp' suffix in voice string...")

    # Parse from string
    blend_str = f"{VOICE_A}:50,{VOICE_B}:50@slerp"
    print(f"  String: '{blend_str}'")

    samples_parsed, _ = kokoro.create(TEST_TEXT, voice=blend_str, lang=LANG)
    print(f"  Generated {len(samples_parsed)} samples ({len(samples_parsed) / sr:.2f}s)")

    announcement_parsed = "SLERP via string format."
    intro_parsed, _ = kokoro.create(announcement_parsed, voice=VOICE_A, lang=LANG)
    all_samples.extend([intro_parsed, samples_parsed])

    # Combine all audio
    print_separator("Saving Combined Audio")
    combined_samples = np.concatenate(all_samples)

    output_file = "voice_slerp_demo.wav"
    sf.write(output_file, combined_samples, sample_rate)

    total_duration = len(combined_samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration / 60:.2f} minutes)")

    # Summary
    print_separator("SUMMARY")
    print("\nVoice blending methods compared:")
    print("  • Original voices: af_bella and am_adam")
    print("  • Linear interpolation: Traditional weighted average")
    print("  • SLERP interpolation: Spherical linear interpolation")
    print("\nInterpolation points tested: 25%, 50%, 75%")
    print("\nKey observations:")
    print("  • SLERP preserves the 'direction' of voice embeddings")
    print("  • SLERP may produce smoother transitions between voices")
    print("  • Both methods create valid intermediate voices")
    print("\nListen to the WAV file to compare the methods!")
    print("\nUsage examples:")
    print("  # VoiceBlend object with SLERP")
    print("  blend = VoiceBlend([('af_bella', 0.5), ('am_adam', 0.5)], interpolation='slerp')")
    print("  samples, sr = kokoro.create(text, voice=blend)")
    print()
    print("  # String format with @slerp suffix")
    print("  samples, sr = kokoro.create(text, voice='af_bella:50,am_adam:50@slerp')")

    kokoro.close()


if __name__ == "__main__":
    main()
