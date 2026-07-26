#!/usr/bin/env python3
"""
Backend Comparison Example for pykokoro.

This example demonstrates how different phonemization backends affect
the quality and pronunciation of synthesized speech. It compares:

1. gold+silver+espeak (default) - Full dictionaries with espeak fallback
2. gold_only_no_espeak - Gold dictionary only, no silver, no espeak
3. espeak_only - Pure espeak without dictionary lookup
4. goruut - Goruut backend (requires pygoruut)
5. misaki - Misaki G2P with espeak-ng fallback (requires misaki)

The example uses a phonetically rich text that includes various punctuation,
abbreviations, numbers, and both common and rare words to thoroughly test
each backend's capabilities.

Usage:
    python examples/backend_comparison.py

Output:
    backend_comparison.wav - All backend versions in one file
"""

import numpy as np
import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.tokenizer import Tokenizer, TokenizerConfig

# Phonetically rich English text with comprehensive coverage
# Includes: quotes, dashes, ellipses, abbreviations, numbers, punctuation
RICH_TEXT = """
"Why, hello there! Don't stay inside boy!" exclaimed Dr. Smith—a well-known expert—as he gazed at the azure sky. The 5-year-old boy asked, "What's that...?" Mr. Jones replied: "It's a fjord, my friend; quite extraordinary, don't you think?" She nodded... "Yes, absolutely!" The temperature reached 98°F, or approximately 37°C. Lt. Commander Harris served valiantly in the U.S. Navy for 15 years.
""".strip()

VOICE = "af_bella"  # American Female voice
LANG = "en-us"

# Backend configurations to test
BACKENDS = [
    (
        "gold plus silver plus espeak",
        {
            "load_gold": True,
            "load_silver": True,
            "use_espeak_fallback": True,
        },
    ),
    (
        "gold only, no espeak fallback",
        {
            "load_gold": True,
            "load_silver": False,
            "use_espeak_fallback": False,
        },
    ),
    (
        "espeak only, no dictionaries",
        {
            "load_gold": False,
            "load_silver": False,
            "use_espeak_fallback": True,
        },
    ),
    (
        "goruut only, no dictionaries",
        {
            "load_gold": False,
            "load_silver": False,
            "use_espeak_fallback": False,
            "use_goruut_fallback": True,
        },
    ),
    (
        "misaki with espeak fallback",
        {
            "backend": "misaki",  # Special marker for external Misaki G2P
        },
    ),
]


def print_separator(title: str) -> None:
    """Print a visual separator with title."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Generate backend comparison audio."""
    print("Backend Comparison Example")
    print_separator("Rich Test Text")
    print(RICH_TEXT.strip())
    print()

    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")

    pipe = KokoroPipeline(PipelineConfig(voice=VOICE, generation=GenerationConfig(lang=LANG)))

    # Test each backend
    all_samples = []
    sample_rate_value = 24000
    results = []

    for backend_name, backend_config in BACKENDS:
        print_separator(f"Backend: {backend_name}")

        try:
            # Special handling for Misaki (external library)
            if backend_config.get("backend") == "misaki":
                from misaki import en, espeak

                print("Configuration:")
                print("  backend: misaki (external G2P library)")
                print("  fallback: espeak-ng")

                # Misaki G2P with espeak-ng fallback
                fallback = espeak.EspeakFallback(british=False)
                g2p = en.G2P(trf=False, british=False, fallback=fallback)

                # Phonemize using Misaki
                print("\nPhonemizing with Misaki...")
                phonemes, _ = g2p(RICH_TEXT.strip())
                print(f"Phonemes ({len(phonemes)} chars):")
                print(f"  {phonemes[:150]}...")

                # For Misaki, we skip tokenization and go straight to audio
                tokens = []  # Not applicable for external phonemization

            else:
                # Standard pykokoro tokenizer flow
                # Create tokenizer config
                config = TokenizerConfig(**backend_config)
                print("Configuration:")
                print(f"  backend: {config.backend}")
                print(f"  load_gold: {config.load_gold}")
                print(f"  load_silver: {config.load_silver}")
                print(f"  use_espeak_fallback: {config.use_espeak_fallback}")
                print(f"  use_goruut_fallback: {config.use_goruut_fallback}")

                # Create tokenizer
                tokenizer = Tokenizer(config=config)

                # Phonemize
                print("\nPhonemizing...")
                phonemes = tokenizer.phonemize(RICH_TEXT.strip(), lang=LANG)
                print(f"Phonemes ({len(phonemes)} chars):")
                print(f"  {phonemes[:150]}...")

                # Tokenize
                tokens = tokenizer.tokenize(phonemes)
                print(f"Tokens: {len(tokens)}")

            # Generate announcement
            announcement = f"Backend: {backend_name}"
            print(f"\nGenerating announcement: {announcement}")
            filler_res = pipe.run(announcement)
            filler_samples, sample_rate_value = (
                filler_res.audio,
                filler_res.sample_rate,
            )

            # Generate audio from phonemes
            print("Generating audio from phonemes...")
            samples_res = pipe.run(
                phonemes,
                generation=GenerationConfig(lang=LANG, is_phonemes=True),
            )
            samples, sample_rate_value = samples_res.audio, samples_res.sample_rate

            duration = len(samples) / sample_rate_value
            print(f"Duration: {duration:.2f}s")

            # Add to results
            all_samples.append(filler_samples)
            all_samples.append(samples)

            # Add pause between backends
            pause = np.zeros(int(sample_rate_value * 0.5), dtype=np.float32)
            all_samples.append(pause)

            results.append(
                {
                    "name": backend_name,
                    "phonemes": phonemes,
                    "tokens": len(tokens) if tokens else 0,
                    "duration": duration,
                    "success": True,
                }
            )

        except ImportError as e:
            print(f"❌ ImportError: {e}")
            print("   This backend requires additional dependencies.")
            results.append(
                {
                    "name": backend_name,
                    "success": False,
                    "error": str(e),
                }
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(
                {
                    "name": backend_name,
                    "success": False,
                    "error": str(e),
                }
            )

    # Combine all audio
    if all_samples:
        print_separator("Combining All Backends")
        combined_samples = np.concatenate(all_samples)

        output_file = "backend_comparison.wav"
        sf.write(output_file, combined_samples, sample_rate_value)

        total_duration = len(combined_samples) / sample_rate_value
        print(f"\nCreated {output_file}")
        print(f"Total duration: {total_duration:.2f}s")

    # Summary table
    print_separator("COMPARISON SUMMARY")
    print(f"{'Backend':<40} {'Success':<10} {'Tokens':<10} {'Duration':<10}")
    print("-" * 70)

    for result in results:
        if result["success"]:
            print(
                f"{result['name']:<40} {'✓':<10} "
                f"{result['tokens']:<10} {result['duration']:<10.2f}s"
            )
        else:
            print(f"{result['name']:<40} {'✗':<10} {result.get('error', 'Unknown')[:30]}")

    print("\nBackend Comparison:")
    print("  • gold+silver+espeak: Default, best coverage and quality")
    print("  • gold_only: Good quality, reduced memory (~22-31 MB saved)")
    print("  • espeak_only: Fastest initialization, consistent but simpler")
    print("  • goruut: Alternative backend (requires pygoruut)")
    print("  • misaki: External G2P library with advanced features (requires misaki)")

    print("\nPhoneme Differences:")
    if len([r for r in results if r["success"]]) > 1:
        # Show if phonemes differ between backends
        phoneme_sets = [r["phonemes"] for r in results if r["success"]]
        if len(set(phoneme_sets)) > 1:
            print("  ⚠ Different backends produced different phonemes!")
            print("  This may affect pronunciation and audio quality.")
        else:
            print("  ✓ All backends produced identical phonemes.")

    print("\nListen to the audio file to compare quality and pronunciation!")


if __name__ == "__main__":
    main()
