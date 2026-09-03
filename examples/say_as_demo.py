#!/usr/bin/env python3
"""
Say-As Text Normalization Demo for PyKokoro.

This example demonstrates SSMD say-as intent and Spokenform-owned semantic
preparation for natural TTS output.

PyKokoro only carries the structural say-as metadata. Spokenform performs the
generic written-to-spoken realization.

Requirements:
    pip install pykokoro

Usage:
    python examples/say_as_demo.py

Output:
    say_as_numbers_demo.wav - Number normalizations
    say_as_text_demo.wav - Text normalizations
    say_as_datetime_demo.wav - Date/time normalizations
    say_as_mixed_demo.wav - All features combined

Supported interpret-as types:
    Numbers: cardinal, ordinal, digits, number, fraction
    Text: characters, expletive
    Date/Time: date, time
    Other: telephone, unit, address
"""

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig


def demo_numbers(pipe):
    """Demonstrate number normalization."""
    print("\n--- Number Normalization Demo ---")
    print("Testing cardinal, ordinal, digits, and fractions...")

    script = """
Welcome to the number normalization demo.

Cardinal numbers convert digits to words.
I have [123]{as="cardinal"} apples.
That's [1000]{as="cardinal"} more than yesterday.

Ordinal numbers show position or rank.
I came in [3]{as="ordinal"} place.
This is my [21]{as="ordinal"} attempt.

Digits speak each number separately.
My PIN is [1234]{as="digits"}.
The code is [5-5-5]{as="digits"}.

Fractions are also supported.
Use [1/2]{as="fraction"} cup of sugar.
Add [3/4]{as="fraction"} teaspoon of salt.

The alias [456]{as="number"} works like cardinal.
"""

    print(f"\nInput text:\n{script.strip()}\n")

    # Generate audio
    audio = pipe.run(script).audio

    return audio


def demo_text_normalization(pipe):
    """Demonstrate text normalization."""
    print("\n--- Text Normalization Demo ---")
    print("Testing characters and expletive censoring...")

    script = """
Text normalization examples.

The characters type spells out letters.
Please spell [ABC]{as="characters"}.
My initials are [J.K.R.]{as="characters"}.

Expletive censoring for content filtering.
This replaces [inappropriate]{as="expletive"} words.
No more [profanity]{as="expletive"} in output.
"""

    print(f"\nInput text:\n{script.strip()}\n")

    # Generate audio
    audio = pipe.run(script).audio

    return audio


def demo_datetime(pipe):
    """Demonstrate date and time normalization."""
    print("\n--- Date and Time Normalization Demo ---")
    print("Testing date and time formatting...")

    script = """
Date and time examples.

Dates are formatted naturally.
Today is [12/31/2024]{as="date"}.
The meeting is on [2024-01-15]{as="date"}.

Times work in multiple formats.
The call is at [14:30]{as="time"}.
Lunch is at [12:00]{as="time"}.
"""

    print(f"\nInput text:\n{script.strip()}\n")

    # Generate audio
    audio = pipe.run(script).audio

    return audio


def demo_telephone_and_units(pipe):
    """Demonstrate telephone and unit normalization."""
    print("\n--- Telephone and Unit Normalization Demo ---")
    print("Testing telephone numbers and units...")

    script = """
Telephone and unit examples.

Phone numbers are spoken digit by digit.
Call [+1-555-0123]{as="telephone"}.
My number is [555-7890]{as="telephone"}.

Units are expanded naturally.
Add [5kg]{as="unit"} of flour.
The distance is [10km]{as="unit"}.
It weighs [2.5lb]{as="unit"}.
"""

    print(f"\nInput text:\n{script.strip()}\n")

    # Generate audio
    audio = pipe.run(script).audio

    return audio


def demo_mixed_features(pipe):
    """Demonstrate all features together."""
    print("\n--- Mixed Features Demo ---")
    print("Testing all say-as types in one script...")

    script = """
Welcome to the complete say-as demonstration.

Start by calling [+1-555-0123]{as="telephone"}.
The date is [01/15/2024]{as="date"}.
Meeting time: [14:30]{as="time"}.

We have [1234]{as="cardinal"} participants.
You're our [100]{as="ordinal"} caller!
Enter code [9-8-7-6]{as="digits"}.

Measurements needed:
[2.5kg]{as="unit"} sugar.
[1/2]{as="fraction"} cup milk.
[250ml]{as="unit"} water.

Spell your name using [A-B-C]{as="characters"}.

Thank you, and watch your [language]{as="expletive"}!
"""

    print(f"\nInput text:\n{script.strip()}\n")

    # Generate audio
    audio = pipe.run(script).audio

    return audio


def demo_multilingual(pipe):
    """Demonstrate multi-language support."""
    print("\n--- Multi-Language Support Demo ---")
    print("Testing say-as with different languages...")

    # English
    script_en = """
English: I have [123]{as="cardinal"} items.
The [3]{as="ordinal"} person wins.
"""

    # Note: For other languages, you'd need to switch voices
    # This demo uses English voice for all examples
    print(f"\nEnglish example:\n{script_en.strip()}\n")

    audio = pipe.run(script_en).audio

    return audio


def demo_error_handling(pipe):
    """Demonstrate error handling and edge cases."""
    print("\n--- Error Handling Demo ---")
    print("Testing edge cases and error handling...")

    script = """
Error handling examples.

Invalid numbers fall back to original text.
This is [invalid]{as="cardinal"} input.

Empty strings are handled gracefully.
Here's an empty cardinal: []{as="cardinal"}.

Unsupported types return original text.
This uses [test]{as="unknown_type"} markup.

Mixed valid and invalid:
Valid: [42]{as="cardinal"}.
Invalid: [abc]{as="ordinal"}.
Valid: [XYZ]{as="characters"}.
"""

    print(f"\nInput text:\n{script.strip()}\n")

    # Generate audio
    audio = pipe.run(script).audio

    return audio


def main():
    """Run all say-as demos."""
    print("=" * 70)
    print("Say-As Text Normalization Demo")
    print("=" * 70)

    VOICE = "af_sarah"
    SAMPLE_RATE = 24000

    print(f"\nVoice: {VOICE}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print("\nFeatures demonstrated:")
    print("  - Cardinal numbers (123 → one hundred twenty-three)")
    print("  - Ordinal numbers (3 → third)")
    print("  - Digits (1234 → one two three four)")
    print("  - Fractions (1/2 → one half)")
    print("  - Characters (ABC → A B C)")
    print("  - Expletive censoring (word → beep)")
    print("  - Dates (12/31/2024 → December thirty-first, 2024)")
    print("  - Times (14:30 → two thirty PM)")
    print("  - Telephone (+1-555-0123 → plus one five five five...)")
    print("  - Units (5kg → five kilograms)")

    # Initialize Kokoro
    pipe = KokoroPipeline(
        PipelineConfig(voice=VOICE, generation=GenerationConfig(lang="en-us", speed=1.0))
    )

    # Demo 1: Numbers
    audio_numbers = demo_numbers(pipe)
    output_file = artifact_path("say_as_numbers_demo.wav")
    sf.write(output_file, audio_numbers, SAMPLE_RATE)
    duration = len(audio_numbers) / SAMPLE_RATE
    print(f"✓ Created: {output_file} ({duration:.2f}s)")

    # Demo 2: Text normalization
    audio_text = demo_text_normalization(pipe)
    output_file = artifact_path("say_as_text_demo.wav")
    sf.write(output_file, audio_text, SAMPLE_RATE)
    duration = len(audio_text) / SAMPLE_RATE
    print(f"✓ Created: {output_file} ({duration:.2f}s)")

    # Demo 3: Date/Time
    audio_datetime = demo_datetime(pipe)
    output_file = artifact_path("say_as_datetime_demo.wav")
    sf.write(output_file, audio_datetime, SAMPLE_RATE)
    duration = len(audio_datetime) / SAMPLE_RATE
    print(f"✓ Created: {output_file} ({duration:.2f}s)")

    # Demo 4: Telephone and Units
    audio_tel_units = demo_telephone_and_units(pipe)
    output_file = artifact_path("say_as_telephone_units_demo.wav")
    sf.write(output_file, audio_tel_units, SAMPLE_RATE)
    duration = len(audio_tel_units) / SAMPLE_RATE
    print(f"✓ Created: {output_file} ({duration:.2f}s)")

    # Demo 5: Mixed features
    audio_mixed = demo_mixed_features(pipe)
    output_file = artifact_path("say_as_mixed_demo.wav")
    sf.write(output_file, audio_mixed, SAMPLE_RATE)
    duration = len(audio_mixed) / SAMPLE_RATE
    print(f"✓ Created: {output_file} ({duration:.2f}s)")

    # Demo 6: Multi-language
    audio_multilang = demo_multilingual(pipe)
    output_file = artifact_path("say_as_multilingual_demo.wav")
    sf.write(output_file, audio_multilang, SAMPLE_RATE)
    duration = len(audio_multilang) / SAMPLE_RATE
    print(f"✓ Created: {output_file} ({duration:.2f}s)")

    # Demo 7: Error handling
    audio_errors = demo_error_handling(pipe)
    output_file = artifact_path("say_as_error_handling_demo.wav")
    sf.write(output_file, audio_errors, SAMPLE_RATE)
    duration = len(audio_errors) / SAMPLE_RATE
    print(f"✓ Created: {output_file} ({duration:.2f}s)")

    print("\n" + "=" * 70)
    print("Say-As Demo Complete!")
    print("=" * 70)
    print("\nSay-As Types Demonstrated:")
    print("  Numbers:")
    print("    ✓ cardinal - Regular numbers as words")
    print("    ✓ ordinal - Position numbers (first, second, third)")
    print("    ✓ digits - Spell out each digit separately")
    print("    ✓ fraction - Fractions (one half, three quarters)")
    print("    ✓ number - Alias for cardinal")
    print("  Text:")
    print("    ✓ characters - Spell out letter by letter")
    print("    ✓ expletive - Censor inappropriate words")
    print("  Date/Time:")
    print("    ✓ date - Natural date formatting")
    print("    ✓ time - Natural time formatting")
    print("  Other:")
    print("    ✓ telephone - Phone number formatting")
    print("    ✓ unit - Measurement units (kg, km, etc.)")
    print("    ✓ address - Address formatting (basic)")

    print("\nSyntax:")
    print('  [text]{as="interpret-as"}')
    print('  [text]{as="interpret-as" format="fmt"}')
    print('  [text]{as="interpret-as" detail="N"}')

    print("\nExamples:")
    print('  [123]{as="cardinal"} → "one hundred twenty-three"')
    print('  [3]{as="ordinal"} → "third"')
    print('  [1234]{as="digits"} → "one two three four"')
    print('  [ABC]{as="characters"} → "A B C"')
    print('  [+1-555-0123]{as="telephone"} → "plus one five five five..."')
    print('  [12/31/2024]{as="date"} → "December thirty-first, 2024"')
    print('  [14:30]{as="time"} → "two thirty PM"')
    print('  [5kg]{as="unit"} → "five kilograms"')
    print('  [1/2]{as="fraction"} → "one half"')

    print("\nLanguage Support:")
    print("  The say-as feature is realized by Spokenform.")
    print("  - English (en-us, en-gb)")
    print("  - French (fr-fr)")
    print("  - German (de-de)")
    print("  - Spanish (es-es)")
    print("  - And many more (Italian, Portuguese, Russian, Japanese, etc.)")
    print("  - Falls back to English for unsupported languages")

    print("\nIntegration with SSMD:")
    print("  Say-as is fully integrated with other SSMD features:")
    print('  - Combines with prosody: [+1-555-0123]{as="telephone" volume="loud"}')
    print('  - Combines with pauses: [123]{as="cardinal"} ...s items')
    print('  - Combines with emphasis: [42]{as="ordinal" emphasis="moderate"} place!')
    print('  - Works with voice switching: [100]{as="cardinal" voice="af_sarah"}')


if __name__ == "__main__":
    main()
