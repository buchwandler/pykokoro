#!/usr/bin/env python3
"""
Model Quality Comparison Example for PyKokoro.

This example demonstrates how to generate audio with all available
model quality options (fp32, fp16, q8, q4, etc.) for comparison.

Configure the MODEL_VARIANT and MODEL_SOURCE variables to easily
switch between different variants and sources:
- Variants: 'v1.0' or 'v1.1-zh'
- Sources: 'huggingface' or 'github'
"""

from pathlib import Path

import soundfile as sf

from pykokoro import Kokoro

# ========================================================================
# CONFIGURATION - Change these to test different variants/sources
# ========================================================================
MODEL_VARIANT = "v1.0"  # Options: 'v1.0', 'v1.1-zh'
MODEL_SOURCE = "huggingface"  # Options: 'huggingface', 'github'

# Test text
TEST_TEXT = (
    "Hello! This is a quality comparison test. "
    "Listen carefully to compare the audio quality "
    "between different model quantization levels."
)

# Voice to use
VOICE = "af_sarah"  # Use 'af_maple' for v1.1-zh if needed


def get_available_qualities(variant: str, source: str) -> list[str]:
    """
    Get available quality options for a given variant and source.

    Args:
        variant: Model variant ('v1.0' or 'v1.1-zh')
        source: Model source ('huggingface' or 'github')

    Returns:
        List of available quality options
    """
    if source == "huggingface":
        # HuggingFace has 8 quality options for both variants
        # return ["fp32", "fp16", "q8", "q8f16", "q4", "q4f16", "uint8", "uint8f16"]
        return ["fp32", "fp16", "q8", "q4", "q4f16", "uint8", "uint8f16"]
    elif source == "github":
        if variant == "v1.0":
            # GitHub v1.0 has 4 quality options
            return ["fp32", "fp16", "fp16-gpu", "q8"]
        else:  # v1.1-zh
            # GitHub v1.1-zh only has fp32
            return ["fp32"]
    else:
        raise ValueError(f"Unknown source: {source}")


def main():
    """Generate audio with all available quality options."""
    print("=" * 70)
    print("PyKokoro Model Quality Comparison")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Variant: {MODEL_VARIANT}")
    print(f"  Source:  {MODEL_SOURCE}")
    print(f"  Voice:   {VOICE}")
    print(f"  Text:    {TEST_TEXT[:50]}...")
    print()

    # Get available qualities for this configuration
    available_qualities = get_available_qualities(MODEL_VARIANT, MODEL_SOURCE)
    print(f"Available quality options: {', '.join(available_qualities)}")
    print(f"Total options to test: {len(available_qualities)}\n")

    results = []

    # Generate audio for each quality option
    for idx, quality in enumerate(available_qualities, 1):
        print("-" * 70)
        print(f"[{idx}/{len(available_qualities)}] Generating with quality: {quality}")
        print("-" * 70)

        try:
            # Initialize TTS engine with specific quality
            kokoro = Kokoro(
                model_source=MODEL_SOURCE,
                model_variant=MODEL_VARIANT,
                model_quality=quality,
            )

            # Generate audio
            samples, sample_rate = kokoro.create(
                TEST_TEXT,
                voice=VOICE,
                speed=1.0,
                lang="en-us",
            )

            # Calculate duration
            duration = len(samples) / sample_rate

            # Save to file
            output_file = f"{quality}.wav"
            sf.write(output_file, samples, sample_rate)

            # Get file size
            output_file = Path(output_file)
            file_size_mb = output_file.stat().st_size / (1024 * 1024)

            print(f"✓ Created {output_file.name}")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Samples: {len(samples):,}")
            print(f"  File size: {file_size_mb:.2f} MB")

            results.append(
                {
                    "quality": quality,
                    "duration": duration,
                    "file": output_file.name,
                    "file_size_mb": file_size_mb,
                }
            )

        except Exception as e:
            print(f"✗ Failed to generate with quality '{quality}': {e}")
            continue

        print()

    # Print summary
    print("=" * 70)
    print("Summary - Quality Comparison Results")
    print("=" * 70)
    print(
        f"\nSuccessfully generated {len(results)} out of "
        f"{len(available_qualities)} quality options:\n"
    )

    # Print results table
    print(f"{'Quality':<12} {'Duration':<12} {'File Size':<12} {'Filename'}")
    print("-" * 70)
    for result in results:
        print(
            f"{result['quality']:<12} "
            f"{result['duration']:.2f}s{'':<7} "
            f"{result['file_size_mb']:.2f} MB{'':<5} "
            f"{result['file']}"
        )

    print("You can now listen to them to compare quality!\n")

    # Print quality information
    print("Quality Levels Explained:")
    print("  • fp32:      Full precision (32-bit float) - Highest quality")
    print("  • fp16:      Half precision (16-bit float) - High quality")
    print("  • fp16-gpu:  Half precision optimized for GPU")
    print("  • q8:        8-bit quantization - Good quality, smaller size")
    print("  • q8f16:     8-bit quantization with fp16 - Balanced")
    print("  • q4:        4-bit quantization - Smaller size")
    print("  • q4f16:     4-bit quantization with fp16 - Balanced")
    print("  • uint8:     Unsigned 8-bit integer quantization")
    print("  • uint8f16:  Unsigned 8-bit with fp16")
    print()

    # Print configuration tip
    print("Tip: To test a different configuration, edit these variables:")
    print(f"  MODEL_VARIANT = '{MODEL_VARIANT}'  # Try: 'v1.1-zh'")
    print(f"  MODEL_SOURCE = '{MODEL_SOURCE}'  # Try: 'github'")


if __name__ == "__main__":
    main()
