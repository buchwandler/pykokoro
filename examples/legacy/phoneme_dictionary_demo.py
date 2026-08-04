"""Demo script showing custom phoneme dictionary usage.

This script demonstrates how to use custom phoneme dictionaries to control
pronunciation of specific words in TTS conversion.
"""

from pathlib import Path

import soundfile as sf

import pykokoro
from pykokoro import Tokenizer, TokenizerConfig

# Example 1: Using phoneme dictionary with the Tokenizer directly
print("=" * 60)
print("Example 1: Direct Tokenizer Usage")
print("=" * 60)

# Path to phoneme dictionary
dict_path = Path(__file__).parent / "phoneme_dictionary.json"

# Create tokenizer config with phoneme dictionary
config = TokenizerConfig(phoneme_dictionary_path=str(dict_path))
tokenizer = Tokenizer(config=config)

# Test text with words from our dictionary
test_text = "Misaki uses Kubernetes and nginx for her PostgreSQL database."

# Phonemize without dictionary (for comparison)
config_no_dict = TokenizerConfig()
tokenizer_no_dict = Tokenizer(config=config_no_dict)
phonemes_without = tokenizer_no_dict.phonemize(test_text, "en-us")

# Phonemize with dictionary
phonemes_with = tokenizer.phonemize(test_text, "en-us")

print(f"\nText: {test_text}\n")
print("Without dictionary:")
print(f"  {phonemes_without}\n")
print("With dictionary:")
print(f"  {phonemes_with}\n")

# Example 2: Using phoneme dictionary for TTS conversion
print("=" * 60)
print("Example 2: Full TTS Conversion")
print("=" * 60)

# Create TTS engine with custom tokenizer
kokoro = pykokoro.Kokoro()
kokoro._tokenizer = tokenizer  # Use our custom tokenizer with the dictionary

# Convert text to audio
print("\nGenerating audio with custom pronunciations...")
samples, sample_rate = kokoro.create(
    test_text,
    voice="af_bella",
    speed=1.0,
    lang="en-us",
)

# Save to file
output_path = "phoneme_dictionary_demo.wav"
sf.write(output_path, samples, sample_rate)

duration = len(samples) / sample_rate
print(f"\nAudio saved to: {output_path}")
print(f"Duration: {duration:.2f} seconds")
print("Listen to hear the correct pronunciations!")

kokoro.close()

# Example 3: Case-insensitive matching (default)
print("\n" + "=" * 60)
print("Example 3: Case-Insensitive Matching")
print("=" * 60)

# Dictionary entries work regardless of case
test_variants = [
    "Misaki went to the store.",  # Original case
    "misaki went to the store.",  # Lowercase
    "MISAKI went to the store.",  # Uppercase
]

for text in test_variants:
    phonemes = tokenizer.phonemize(text, "en-us")
    has_custom = "misˈɑki" in phonemes
    print(f"{text:35} -> Custom phoneme: {has_custom}")

# Example 4: Creating your own dictionary
print("\n" + "=" * 60)
print("Example 4: Simple Dictionary Format")
print("=" * 60)

print("""
You can create a simple dictionary without metadata:

{
  "MyCharacter": "/mˌIkˈæɹəktɚ/",
  "MyPlace": "/mˌIplˈAs/",
  "TechTerm": "/tˈɛktˈɜɹm/"
}

Or use the metadata format (recommended for generated dictionaries):

{
  "_metadata": {
    "generated_from": "mybook.epub",
    "language": "en-us"
  },
  "entries": {
    "MyCharacter": {
      "phoneme": "/mˌIkˈæɹəktɚ/",
      "occurrences": 42,
      "verified": true
    }
  }
}

Both formats work equally well!
""")

print("\n" + "=" * 60)
print("Usage Tips")
print("=" * 60)
print("""
1. Phonemes MUST be in /phoneme/ format (with forward slashes)
2. Use IPA phoneme notation (same as used by espeak/Kokoro)
3. Test your phonemes with 'pykokoro sample' before converting a whole book
4. Matching is case-insensitive by default (use --phoneme-dict-case-sensitive to change)
5. Only exact word matches are replaced (word boundaries are enforced)

CLI Usage:
  # Use dictionary with convert command
  pykokoro convert book.epub --phoneme-dict phoneme_dictionary.json

  # Test with sample command
   pykokoro sample "Misaki uses Kubernetes" --phoneme-dict phoneme_dictionary.json -p

  # For specific CLI usage, consult the pykokoro documentation
""")
