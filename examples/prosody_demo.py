#!/usr/bin/env python3
"""
Prosody Control Demo for pykokoro.

This example demonstrates prosody control features (volume, pitch, rate)
in pykokoro using manually set metadata. Once SSMD library supports prosody
shorthand parsing (e.g., +loud+, >fast>, ^high^), these will work automatically.

For now, this demo shows how prosody processing works by manually setting
the prosody metadata on segments.

Requirements:
    pip install pykokoro[prosody]  # Installs audiomentations + librosa

Usage:
    python examples/prosody_demo.py

Output:
    prosody_volume_demo.wav - Volume variations
    prosody_pitch_demo.wav - Pitch variations
    prosody_rate_demo.wav - Rate/speed variations
    prosody_combined_demo.wav - All prosody features combined

Features demonstrated:
    - Volume control: silent, soft, medium, loud, x-loud, +6dB, -3dB, +20%
    - Pitch control: x-low, low, medium, high, x-high, +2st, -1st, +10%
    - Rate control: x-slow, slow, medium, fast, x-fast, 120%, 80%
    - Combined prosody: multiple effects applied together
"""

import soundfile as sf

from pykokoro import PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.ssmd_parser import SSMDMetadata
from pykokoro.stages.g2p.kokorog2p import PhonemeSegment
from pykokoro.stages.synth.onnx import OnnxSynthesizerAdapter
from pykokoro.tokenizer import Tokenizer
from pykokoro.types import Trace

# Check if prosody libraries are available
try:
    from pykokoro.prosody import AUDIOMENTATIONS_AVAILABLE, LIBROSA_AVAILABLE

    if not AUDIOMENTATIONS_AVAILABLE and not LIBROSA_AVAILABLE:
        print("WARNING: prosody libraries not installed. Pitch and rate changes will be disabled.")
        print("Install with: pip install pykokoro[prosody]")
    elif not AUDIOMENTATIONS_AVAILABLE:
        print("INFO: Using librosa (install audiomentations for better quality)")
        print("      pip install audiomentations>=0.36.0")
except ImportError:
    print("WARNING: prosody module not available.")
    AUDIOMENTATIONS_AVAILABLE = False
    LIBROSA_AVAILABLE = False


def create_segment_with_prosody(
    text: str, tokenizer, lang: str, volume=None, pitch=None, rate=None
):
    """Helper to create a PhonemeSegment with prosody metadata.

    Args:
        text: Text to phonemize
        tokenizer: Tokenizer instance
        lang: Language code
        volume: Volume specification (e.g., 'loud', '+6dB', '+20%')
        pitch: Pitch specification (e.g., 'high', '+2st', '+10%')
        rate: Rate specification (e.g., 'fast', '120%', '+20%')

    Returns:
        PhonemeSegment with prosody metadata
    """
    # Phonemize the text
    phonemes = tokenizer.phonemize(text, lang=lang)

    # Tokenize the phonemes
    tokens = tokenizer.tokenize(phonemes)

    # Create SSMD metadata with prosody
    metadata = SSMDMetadata(prosody_volume=volume, prosody_pitch=pitch, prosody_rate=rate)

    # Create segment
    segment = PhonemeSegment(
        text=text,
        phonemes=phonemes,
        tokens=tokens,
        lang=lang,
        pause_after=0.0,
        ssmd_metadata=metadata.to_dict(),
    )

    return segment


def demo_volume(synth, cfg, tokenizer, lang):
    """Demonstrate volume control."""
    print("\n--- Volume Control Demo ---")
    print("Testing different volume levels...")

    segments = []

    # Absolute volume levels
    absolute_volumes = [
        ("This is x-soft volume.", "x-soft"),
        ("This is soft volume.", "soft"),
        ("This is medium volume.", "medium"),
        ("This is loud volume.", "loud"),
        ("This is x-loud volume!", "x-loud"),
    ]

    for text, volume in absolute_volumes:
        seg = create_segment_with_prosody(text, tokenizer, lang, volume=volume)
        segments.append(seg)
        # Add pause between samples
        segments.append(PhonemeSegment(text="", phonemes="", tokens=[], pause_after=0.5))

    # Relative volume (dB)
    print("\nTesting relative dB values...")
    segments.append(
        create_segment_with_prosody("This is plus six decibels.", tokenizer, lang, volume="+6dB")
    )
    segments.append(PhonemeSegment(text="", phonemes="", tokens=[], pause_after=0.5))

    segments.append(
        create_segment_with_prosody("This is minus three decibels.", tokenizer, lang, volume="-3dB")
    )
    segments.append(PhonemeSegment(text="", phonemes="", tokens=[], pause_after=0.5))

    # Relative volume (percentage)
    print("Testing relative percentage values...")
    segments.append(
        create_segment_with_prosody(
            "This is twenty percent louder.", tokenizer, lang, volume="+20%"
        )
    )
    segments.append(PhonemeSegment(text="", phonemes="", tokens=[], pause_after=0.5))

    segments.append(
        create_segment_with_prosody(
            "This is twenty percent quieter.", tokenizer, lang, volume="-20%"
        )
    )

    # Generate audio
    audio = synth.synthesize(segments, cfg, Trace())

    return audio


def demo_pitch(synth, cfg, tokenizer, lang):
    """Demonstrate pitch control."""
    print("\n--- Pitch Control Demo ---")

    if not AUDIOMENTATIONS_AVAILABLE and not LIBROSA_AVAILABLE:
        print("Skipping pitch demo (audiomentations/librosa not available)")
        return None

    print("Testing different pitch levels...")

    segments = []

    # Absolute pitch levels
    absolute_pitches = [
        ("This is x-low pitch.", "x-low"),
        ("This is low pitch.", "low"),
        ("This is medium pitch.", "medium"),
        ("This is high pitch.", "high"),
        ("This is x-high pitch!", "x-high"),
    ]

    for text, pitch in absolute_pitches:
        seg = create_segment_with_prosody(text, tokenizer, lang, pitch=pitch)
        segments.append(seg)
        # Add pause between samples
        segments.append(PhonemeSegment(text="", phonemes="", tokens=[], pause_after=0.5))

    # Relative pitch (semitones)
    print("\nTesting relative semitone values...")
    segments.append(
        create_segment_with_prosody("This is two semitones higher.", tokenizer, lang, pitch="+2st")
    )
    segments.append(PhonemeSegment(text="", phonemes="", tokens=[], pause_after=0.5))

    segments.append(
        create_segment_with_prosody("This is one semitone lower.", tokenizer, lang, pitch="-1st")
    )

    # Generate audio
    audio = synth.synthesize(segments, cfg, Trace())

    return audio


def demo_rate(synth, cfg, tokenizer, lang):
    """Demonstrate rate/speed control."""
    print("\n--- Rate/Speed Control Demo ---")

    if not AUDIOMENTATIONS_AVAILABLE and not LIBROSA_AVAILABLE:
        print("Skipping rate demo (audiomentations/librosa not available)")
        return None

    print("Testing different speed levels...")

    segments = []

    # Absolute rate levels
    absolute_rates = [
        ("This is x-slow speed.", "x-slow"),
        ("This is slow speed.", "slow"),
        ("This is medium speed.", "medium"),
        ("This is fast speed.", "fast"),
        ("This is x-fast speed!", "x-fast"),
    ]

    for text, rate in absolute_rates:
        seg = create_segment_with_prosody(text, tokenizer, lang, rate=rate)
        segments.append(seg)
        # Add pause between samples
        segments.append(PhonemeSegment(text="", phonemes="", tokens=[], pause_after=0.5))

    # Relative rate (percentage)
    print("\nTesting relative percentage values...")
    segments.append(
        create_segment_with_prosody("This is twenty percent faster.", tokenizer, lang, rate="120%")
    )
    segments.append(PhonemeSegment(text="", phonemes="", tokens=[], pause_after=0.5))

    segments.append(
        create_segment_with_prosody("This is twenty percent slower.", tokenizer, lang, rate="80%")
    )

    # Generate audio
    audio = synth.synthesize(segments, cfg, Trace())

    return audio


def demo_combined(synth, cfg, tokenizer, lang):
    """Demonstrate combined prosody effects."""
    print("\n--- Combined Prosody Demo ---")

    if not AUDIOMENTATIONS_AVAILABLE and not LIBROSA_AVAILABLE:
        print("Skipping combined demo (audiomentations/librosa not available)")
        return None

    print("Testing combinations of volume, pitch, and rate...")

    segments = []

    # Introduction
    segments.append(create_segment_with_prosody("Listen to these combinations.", tokenizer, lang))
    segments.append(PhonemeSegment(text="", phonemes="", tokens=[], pause_after=0.8))

    # Combination 1: Loud + High + Fast
    segments.append(
        create_segment_with_prosody(
            "This is loud, high pitched, and fast!",
            tokenizer,
            lang,
            volume="loud",
            pitch="high",
            rate="fast",
        )
    )
    segments.append(PhonemeSegment(text="", phonemes="", tokens=[], pause_after=0.8))

    # Combination 2: Soft + Low + Slow
    segments.append(
        create_segment_with_prosody(
            "This is soft, low pitched, and slow.",
            tokenizer,
            lang,
            volume="soft",
            pitch="low",
            rate="slow",
        )
    )
    segments.append(PhonemeSegment(text="", phonemes="", tokens=[], pause_after=0.8))

    # Combination 3: Custom values
    segments.append(
        create_segment_with_prosody(
            "Custom prosody with plus six decibels, plus two semitones, and one "
            "hundred twenty percent speed.",
            tokenizer,
            lang,
            volume="+6dB",
            pitch="+2st",
            rate="120%",
        )
    )
    segments.append(PhonemeSegment(text="", phonemes="", tokens=[], pause_after=0.8))

    # Conclusion
    segments.append(create_segment_with_prosody("That completes the demo.", tokenizer, lang))

    # Generate audio
    audio = synth.synthesize(segments, cfg, Trace())

    return audio


def main():
    """Run all prosody demos."""
    print("=" * 70)
    print("Prosody Control Demo")
    print("=" * 70)

    VOICE = "af_sarah"
    LANG = "en-us"
    SAMPLE_RATE = 24000

    print(f"\nVoice: {VOICE}")
    print(f"Language: {LANG}")

    config = PipelineConfig(
        voice=VOICE,
        generation=GenerationConfig(lang=LANG, pause_mode="manual"),
    )
    synth = OnnxSynthesizerAdapter()
    tokenizer = Tokenizer()

    # Demo 1: Volume
    audio_volume = demo_volume(synth, config, tokenizer, LANG)
    if audio_volume is not None:
        output_file = "prosody_volume_demo.wav"
        sf.write(output_file, audio_volume, SAMPLE_RATE)
        duration = len(audio_volume) / SAMPLE_RATE
        print(f"Created: {output_file} ({duration:.2f}s)")

    # Demo 2: Pitch
    audio_pitch = demo_pitch(synth, config, tokenizer, LANG)
    if audio_pitch is not None:
        output_file = "prosody_pitch_demo.wav"
        sf.write(output_file, audio_pitch, SAMPLE_RATE)
        duration = len(audio_pitch) / SAMPLE_RATE
        print(f"Created: {output_file} ({duration:.2f}s)")

    # Demo 3: Rate
    audio_rate = demo_rate(synth, config, tokenizer, LANG)
    if audio_rate is not None:
        output_file = "prosody_rate_demo.wav"
        sf.write(output_file, audio_rate, SAMPLE_RATE)
        duration = len(audio_rate) / SAMPLE_RATE
        print(f"Created: {output_file} ({duration:.2f}s)")

    # Demo 4: Combined
    audio_combined = demo_combined(synth, config, tokenizer, LANG)
    if audio_combined is not None:
        output_file = "prosody_combined_demo.wav"
        sf.write(output_file, audio_combined, SAMPLE_RATE)
        duration = len(audio_combined) / SAMPLE_RATE
        print(f"Created: {output_file} ({duration:.2f}s)")

    print("\n" + "=" * 70)
    print("Prosody Demo Complete!")
    print("=" * 70)
    print("\nProsody Features Demonstrated:")
    print("  ✓ Volume: silent, x-soft, soft, medium, loud, x-loud")
    print("  ✓ Volume (dB): +6dB, -3dB")
    print("  ✓ Volume (%): +20%, -20%")
    if AUDIOMENTATIONS_AVAILABLE or LIBROSA_AVAILABLE:
        print("  ✓ Pitch: x-low, low, medium, high, x-high")
        print("  ✓ Pitch (semitones): +2st, -1st")
        print("  ✓ Rate: x-slow, slow, medium, fast, x-fast")
        print("  ✓ Rate (%): 120%, 80%")
        print("  ✓ Combined prosody effects")
        if AUDIOMENTATIONS_AVAILABLE:
            print("\n  ℹ Using audiomentations (highest quality)")
        else:
            print("\n  ℹ Using librosa (install audiomentations for better quality)")
    else:
        print("  ⚠ Pitch and rate features require audiomentations or librosa")
        print("    Install with: pip install pykokoro[prosody]")

    print("\nNOTE: SSMD Prosody Shorthand Support (Coming Soon)")
    print("=" * 70)
    print("The SSMD library is being updated to support prosody shorthand:")
    print("  +loud+    - Increase volume")
    print("  >fast>    - Increase rate/speed")
    print("  ^high^    - Increase pitch")
    print("\nOnce SSMD parsing is updated, you'll be able to use prosody")
    print("directly in text without manually creating segments!")
    print("\nExample future usage:")
    print('  "This is +louder+ and >faster> and ^higher pitched^!"')


if __name__ == "__main__":
    main()
