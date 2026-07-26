#!/usr/bin/env python3
"""
English TTS example using pykokoro.

This example demonstrates text-to-speech synthesis in English
using the Kokoro model with American and British English voices.

It also shows how to use phonemes directly with the is_phonemes parameter.

Usage:
    python examples/english.py

Output:
    english_demo.wav - Generated English speech audio from text
    english_phonemes_demo.wav - Generated English speech audio from phonemes

Available English voices:
    American Female: af_alloy, af_aoede, af_bella, af_heart, af_jessica,
                     af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky
    American Male: am_adam, am_echo, am_eric, am_fenrir, am_liam,
                   am_michael, am_onyx, am_puck, am_santa
    British Female: bf_alice, bf_emma, bf_isabella, bf_lily
    British Male: bm_daniel, bm_fable, bm_george, bm_lewis
"""

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.tokenizer import Tokenizer

# Quote about technology and the future
TEXT = (
    "The best way to predict the future is to create it. "
    "Technology is nothing without the imagination to use it wisely. "
    "[tomato](ph: təˈmeɪtoʊ)[tomato](ph: ………………)"
    "Every great innovation begins with a simple question: what if?"
)

VOICE = "af_heart"  # American Female voice
LANG = "en-us"  # American English


def main():
    """Generate English speech audio."""
    print("Initializing TTS engine...")
    generation = GenerationConfig(lang=LANG, speed=1.0)
    pipe = KokoroPipeline(PipelineConfig(voice=VOICE, generation=generation))

    # Example 1: Generate from text
    print("\n=== Example 1: Text-to-Speech ===")
    print(f"Text: {TEXT}")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")

    print("\nGenerating audio from text...")
    res = pipe.run(TEXT)
    samples, sample_rate = res.audio, res.sample_rate

    output_file = "english_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"Created {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    # Example 2: Generate from phonemes directly
    print("\n=== Example 2: Phonemes-to-Speech ===")
    # First, convert text to phonemes to show the phoneme representation
    tokenizer = Tokenizer()
    phonemes = tokenizer.phonemize(TEXT, lang=LANG)
    print(f"Phonemes: {phonemes[:100]}...")  # Show first 100 chars
    print(f"Phoneme length: {len(phonemes)} characters")

    print("\nGenerating audio from phonemes...")
    res_phonemes = pipe.run(
        phonemes,
        generation=GenerationConfig(lang=LANG, speed=1.0, is_phonemes=True),
    )
    samples_from_phonemes, sample_rate = (
        res_phonemes.audio,
        res_phonemes.sample_rate,
    )

    output_file_phonemes = "english_phonemes_demo.wav"
    sf.write(output_file_phonemes, samples_from_phonemes, sample_rate)

    duration_phonemes = len(samples_from_phonemes) / sample_rate
    print(f"Created {output_file_phonemes}")
    print(f"Duration: {duration_phonemes:.2f} seconds")

    # Example 3: Custom phonemes with markdown notation
    print("\n=== Example 3: Custom Phonemes with Markdown ===")
    # You can also use markdown notation like in the kokoro-onnx example:
    # [word](/phoneme/)
    custom_text = (
        '[PyKokoro]{ph="paɪkəkˈoʊɹoʊ"} is a Python library for text-to-speech.'
    )

    samples_custom = pipe.run(custom_text).audio
    sample_rate = res.sample_rate

    output_file_custom = "english_custom_demo.wav"
    sf.write(output_file_custom, samples_custom, sample_rate)
    print(f"Created {output_file_custom}")

    print("\nDone!")


if __name__ == "__main__":
    main()
