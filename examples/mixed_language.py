#!/usr/bin/env python3
"""
Mixed-language TTS example using pykokoro.

This example demonstrates automatic language detection and phonemization
for text containing multiple languages. This is useful for:
- Technical documents with English terms in other languages
- Brand names and product names
- Code snippets in documentation
- Multilingual content

The mixed-language feature uses kokorog2p's preprocess_multilang and
lingua-language-detector to automatically detect language boundaries and
annotate text with language tags for proper phonemization.

Usage:
    python examples/mixed_language.py

Output:
    mixed_language_demo.wav - Generated mixed-language speech

Requirements:
    - kokorog2p with preprocess_multilang support
    - lingua-language-detector (optional, for detection)

Note:
    If lingua-language-detector is not installed, the system will fall back
    to single-language mode with a warning.
"""

import logging

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.tokenizer import Tokenizer, TokenizerConfig

# Configure logging to see mixed-language detection in action
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

# Example 1: German text with English technical terms
GERMAN_ENGLISH_TEXT = """
Willkommen zum Mixed-Language Demo!

In der modernen Softwareentwicklung verwenden wir viele englische Begriffe.
Zum Beispiel: Machine Learning, Cloud Computing, und Artificial Intelligence.

Wir arbeiten mit Python, JavaScript, und anderen Programming Languages.
Das Team hat ein neues Framework für Data Science entwickelt.

Unser Meeting findet im Conference Room statt.
Bitte bringen Sie Ihr Laptop mit.
"""

# Example 2: French text with English words
FRENCH_ENGLISH_TEXT = """
Bonjour! Bienvenue à notre présentation.

Le Streaming et le Cloud Computing sont très importants aujourd'hui.
Nous utilisons le Machine Learning pour améliorer nos Services.

Notre Startup développe des Applications mobiles avec React Native.
Le Team est composé de Developers expérimentés.
"""

# Example 3: Spanish text with English business terms
SPANISH_ENGLISH_TEXT = """
Hola! Este es un ejemplo de texto mixto.

En el Business moderno, usamos muchos términos en inglés.
Por ejemplo: Marketing Digital, Social Media, y E-commerce.

Nuestro Workflow incluye Daily Standup Meetings.
El Project Manager coordina todo el Team.
"""


def demo_mixed_language(text: str, primary_lang: str, voice: str, output_file: str):
    """Demonstrate mixed-language TTS.

    Args:
        text: Text containing multiple languages
        primary_lang: Primary language code
        voice: Voice to use for synthesis
        output_file: Output filename
    """
    print(f"\n{'=' * 70}")
    print(f"Demo: {primary_lang.upper()} with English terms")
    print(f"{'=' * 70}")
    print(f"Voice: {voice}")
    print(f"Primary language: {primary_lang}")
    print(f"Allowed languages: [{primary_lang}, en-us]")
    print()

    # Create tokenizer with mixed-language support
    config = TokenizerConfig(
        use_mixed_language=True,
        mixed_language_primary=primary_lang,
        mixed_language_allowed=[primary_lang, "en-us"],
        mixed_language_confidence=0.7,  # 70% confidence threshold
    )

    try:
        tokenizer = Tokenizer(config=config)

        # Show a preview of language detection
        print("Text preview (first 100 chars):")
        print(f"  {text.strip()[:100]}...")
        print()

        # Phonemize the text
        print("Phonemizing with automatic language detection...")
        phonemes = tokenizer.phonemize(text, lang=primary_lang)
        print(f"Generated {len(phonemes)} phoneme characters")
        print()

        # Generate audio
        print("Generating audio...")
        pipe = KokoroPipeline(
            PipelineConfig(
                voice=voice,
                generation=GenerationConfig(lang=primary_lang, speed=1.0),
                tokenizer_config=config,
            )
        )
        res = pipe.run(text)
        samples, sample_rate = res.audio, res.sample_rate

        # Save audio
        sf.write(output_file, samples, sample_rate)
        duration = len(samples) / sample_rate
        print(f"✓ Created {output_file}")
        print(f"  Duration: {duration:.2f} seconds")

    except ImportError as e:
        print("⚠ Warning: Mixed-language mode not available")
        print(f"  {e}")
        print("  Install lingua-language-detector: pip install lingua-language-detector")
        print("  Falling back to single-language mode...")
        print()


def demo_single_language_comparison():
    """Show comparison with single-language mode."""
    print(f"\n{'=' * 70}")
    print("Comparison: Single-Language Mode (Default)")
    print(f"{'=' * 70}")
    print("Using standard German phonemizer for all text...")
    print("(English words will be pronounced with German phonemization)")
    print()

    # Simple German text with one English word
    text = "Das ist ein Meeting."

    # Single-language mode (default)
    tokenizer = Tokenizer()  # Default config
    phonemes_single = tokenizer.phonemize(text, lang="de")
    print(f"Single-language phonemes: {phonemes_single}")

    # Mixed-language mode
    try:
        config = TokenizerConfig(
            use_mixed_language=True,
            mixed_language_primary="de",
            mixed_language_allowed=["de", "en-us"],
        )
        tokenizer_mixed = Tokenizer(config=config)
        phonemes_mixed = tokenizer_mixed.phonemize(text, lang="de")
        print(f"Mixed-language phonemes: {phonemes_mixed}")
        print()
        print("Notice the difference in pronunciation for 'Meeting'")
    except ImportError:
        print("(Mixed-language mode not available)")


def main():
    """Run all mixed-language demonstrations."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           Mixed-Language TTS Demonstration                       ║
║                                                                  ║
║  This demo shows automatic language detection for text with     ║
║  multiple languages (e.g., German text with English words).     ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # Demo 1: German + English
    demo_mixed_language(
        text=GERMAN_ENGLISH_TEXT,
        primary_lang="de",
        voice="ff_siwis",  # French voice works well for German
        output_file="mixed_language_german_demo.wav",
    )

    # Demo 2: French + English
    demo_mixed_language(
        text=FRENCH_ENGLISH_TEXT,
        primary_lang="fr-fr",
        voice="ff_siwis",  # Native French voice
        output_file="mixed_language_french_demo.wav",
    )

    # Demo 3: Spanish + English
    demo_mixed_language(
        text=SPANISH_ENGLISH_TEXT,
        primary_lang="es",
        voice="ef_dora",  # Spanish voice
        output_file="mixed_language_spanish_demo.wav",
    )

    # Demo 4: Comparison
    demo_single_language_comparison()

    print(f"\n{'=' * 70}")
    print("All demonstrations complete!")
    print(f"{'=' * 70}\n")

    print("Configuration notes:")
    print("  - use_mixed_language: Enable/disable automatic detection")
    print("  - mixed_language_primary: Fallback language (e.g., 'de', 'fr-fr')")
    print("  - mixed_language_allowed: Languages to detect (e.g., ['de', 'en-us'])")
    print("  - mixed_language_confidence: Detection threshold (0.0-1.0, default 0.7)")
    print()
    print("This feature requires:")
    print("  - kokorog2p with preprocess_multilang support")
    print("  - lingua-language-detector (pip install lingua-language-detector)")
    print()


if __name__ == "__main__":
    main()
