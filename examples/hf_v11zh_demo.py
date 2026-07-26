#!/usr/bin/env python3
"""
Demo script for HuggingFace v1.1-zh model support.

This example demonstrates how to use the Kokoro v1.1-zh model from HuggingFace
with various quantization levels and 103 available voices.

Usage:
    python examples/hf_v11zh_demo.py
"""

import pykokoro


def main():
    print("=" * 80)
    print("PyKokoro - HuggingFace v1.1-zh Model Demo")
    print("=" * 80)
    print()

    # Initialize Kokoro with HuggingFace v1.1-zh model
    print("Initializing Kokoro with HuggingFace v1.1-zh model...")
    print("Model source: onnx-community/Kokoro-82M-v1.1-zh-ONNX")
    print()

    try:
        kokoro = pykokoro.Kokoro(
            model_source="huggingface", model_variant="v1.1-zh", model_quality="fp32"
        )

        # Get available voices
        available_voices = kokoro.get_voices()
        print("✓ Model loaded successfully!")
        print(f"✓ Available voices: {len(available_voices)}")
        print(f"✓ Sample voices: {', '.join(available_voices[:10])}...")
        print()

        # Test with English text
        test_text = "Hello! This is a test of the HuggingFace v1.1-zh model."
        print(f"Generating audio for: '{test_text}'")

        # Use first available voice
        voice_to_use = available_voices[0]
        print(f"Using voice: {voice_to_use}")

        samples, sample_rate = kokoro.create(test_text, voice=voice_to_use, speed=1.0, lang="en-us")

        print("✓ Audio generated successfully!")
        print(f"  Duration: {len(samples) / sample_rate:.2f} seconds")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Samples: {len(samples):,}")
        print()

        # Save output
        import soundfile as sf

        output_file = "hf_v11zh_demo.wav"
        sf.write(output_file, samples, sample_rate)
        print(f"✓ Saved to: {output_file}")

        kokoro.close()
        print()
        print("=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
