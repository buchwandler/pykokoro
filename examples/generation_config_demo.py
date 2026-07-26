#!/usr/bin/env python3
"""
Demonstrate GenerationConfig for easier configuration management.

This example shows how to use GenerationConfig to simplify the PyKokoro API
by grouping related parameters into a reusable configuration object.

Benefits of using GenerationConfig:
- Cleaner, more maintainable code
- Easy to reuse configurations across multiple generations
- Self-documenting configuration objects
- IDE autocomplete for all available options
- Override individual parameters when needed

Usage:
    python examples/generation_config_demo.py

Output:
    example1_default_config.wav - Using default config values
    example2_custom_config.wav - Custom config for precise control
    example3_config_reuse.wav - Reusing config across generations
    example4_config_override.wav - Overriding config with kwargs
    example5_backward_compat.wav - Old API still works (no config)
"""

import soundfile as sf

import pykokoro
from pykokoro import GenerationConfig


def main():
    """Generate audio examples using GenerationConfig."""
    print("Initializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    # Example 1: Using default config (equivalent to no config)
    print("\n" + "=" * 70)
    print("Example 1: Default GenerationConfig")
    print("=" * 70)

    default_config = GenerationConfig()
    print(f"Config: speed={default_config.speed}, lang={default_config.lang}")
    print(f"        pause_mode={default_config.pause_mode}")

    samples, sr = kokoro.create(
        text="Hello! This uses the default configuration.",
        voice="af_sarah",
        config=default_config,
    )

    output1 = "example1_default_config.wav"
    sf.write(output1, samples, sr)
    print(f"✓ Generated: {output1}")

    # Example 2: Custom config for manual pause control
    print("\n" + "=" * 70)
    print("Example 2: Custom Config with Manual Pauses")
    print("=" * 70)

    manual_config = GenerationConfig(
        speed=1.2,
        pause_mode="manual",
        pause_clause=0.25,
        pause_sentence=0.5,
        pause_paragraph=1.0,
        pause_variance=0.05,
        random_seed=42,  # Reproducible pauses
    )

    print(f"Config: speed={manual_config.speed}")
    print(f"        pause_mode={manual_config.pause_mode}")
    print(f"        pause_clause={manual_config.pause_clause}s")
    print(f"        pause_sentence={manual_config.pause_sentence}s")
    print(f"        random_seed={manual_config.random_seed}")

    text = """
    Welcome to PyKokoro. This example demonstrates manual pause control
    with custom durations. Notice how pauses are consistent between runs
    thanks to the random seed.

    This is a second paragraph. The pause before it is longer than
    the pause between sentences.
    """

    samples, sr = kokoro.create(
        text=text,
        voice="af_sarah",
        config=manual_config,
    )

    output2 = "example2_custom_config.wav"
    sf.write(output2, samples, sr)
    print(f"✓ Generated: {output2}")

    # Example 3: Reusing config across multiple generations
    print("\n" + "=" * 70)
    print("Example 3: Reusing Config for Multiple Generations")
    print("=" * 70)

    narrator_config = GenerationConfig(
        speed=1.1,
        lang="en-us",
        pause_mode="tts",  # Let TTS handle pauses naturally
    )

    print("Generating 3 sentences with the same config...")
    sentences = [
        "First sentence using the narrator config.",
        "Second sentence with the same configuration.",
        "Third and final sentence, still consistent.",
    ]

    audio_segments = []
    for i, sentence in enumerate(sentences, 1):
        samples, sr = kokoro.create(
            text=sentence,
            voice="am_adam",
            config=narrator_config,
        )
        audio_segments.append(samples)
        print(f"  ✓ Generated sentence {i}")

    # Concatenate all segments
    import numpy as np

    combined = np.concatenate(audio_segments)
    output3 = "example3_config_reuse.wav"
    sf.write(output3, combined, sr)
    print(f"✓ Generated: {output3}")

    # Example 4: Overriding config with kwargs
    print("\n" + "=" * 70)
    print("Example 4: Overriding Config Parameters")
    print("=" * 70)

    base_config = GenerationConfig(
        speed=1.0,
        lang="en-us",
        pause_mode="manual",
        pause_sentence=0.6,
    )

    print(f"Base config: speed={base_config.speed}, pause_sentence={base_config.pause_sentence}s")
    print("Override: speed=1.5 (kwargs take priority)")

    samples, sr = kokoro.create(
        text="This sentence uses the base config but with faster speed.",
        voice="af_bella",
        config=base_config,
        speed=1.5,  # Override just the speed
    )

    output4 = "example4_config_override.wav"
    sf.write(output4, samples, sr)
    print(f"✓ Generated: {output4}")

    # Example 5: Backward compatibility - old API still works
    print("\n" + "=" * 70)
    print("Example 5: Backward Compatibility (No Config)")
    print("=" * 70)

    print("Old API (kwargs only, no config) still works:")
    samples, sr = kokoro.create(
        text="The original API is fully backward compatible.",
        voice="af_nicole",
        speed=1.2,
        lang="en-us",
        pause_mode="manual",
        pause_sentence=0.5,
    )

    output5 = "example5_backward_compat.wav"
    sf.write(output5, samples, sr)
    print(f"✓ Generated: {output5}")

    # Example 6: Config validation
    print("\n" + "=" * 70)
    print("Example 6: Config Validation")
    print("=" * 70)

    print("GenerationConfig validates parameters on creation:")
    try:
        GenerationConfig(speed=-1.0)
        print("  ✗ Should have raised ValueError!")
    except ValueError as e:
        print(f"  ✓ Caught error: {e}")

    try:
        GenerationConfig(pause_clause=-0.5)
        print("  ✗ Should have raised ValueError!")
    except ValueError as e:
        print(f"  ✓ Caught error: {e}")

    try:
        GenerationConfig(pause_mode="invalid")  # type: ignore[arg-type]
        print("  ✗ Should have raised ValueError!")
    except ValueError as e:
        print(f"  ✓ Caught error: {e}")

    kokoro.close()

    # Summary
    print("\n" + "=" * 70)
    print("Summary: GenerationConfig Benefits")
    print("=" * 70)
    print()
    print("✓ Cleaner API - Group related parameters")
    print("✓ Reusable - Define once, use many times")
    print("✓ Flexible - Override individual params with kwargs")
    print("✓ Self-documenting - Config objects show intent")
    print("✓ Type-safe - IDE autocomplete and type checking")
    print("✓ Validated - Catches invalid values early")
    print("✓ Backward compatible - Old API still works")
    print()
    print("Generated files:")
    print(f"  • {output1}")
    print(f"  • {output2}")
    print(f"  • {output3}")
    print(f"  • {output4}")
    print(f"  • {output5}")
    print()


if __name__ == "__main__":
    main()
