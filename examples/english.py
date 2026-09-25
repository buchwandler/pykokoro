#!/usr/bin/env python3
"""English text and phoneme synthesis examples using the request engine.

Run with ``python examples/english.py``. WAV files are written to the
examples output directory.
"""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig, SynthesisSegment

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

TEXT = (
    "The best way to predict the future is to create it. "
    "Technology is nothing without the imagination to use it wisely. "
    "Every great innovation begins with a simple question: what if?"
)
VOICE = "af_heart"
LANG = "en-us"


def main() -> None:
    """Generate English speech from plain text and phoneme requests."""
    config = SynthesisConfig(
        voice=VOICE,
        generation=GenerationConfig(lang=LANG, speed=1.0),
    )
    with KokoroSynthesizer(config) as synthesizer:
        print("=== Example 1: Text-to-Speech ===")
        print(f"Text: {TEXT}\nVoice: {VOICE}\nLanguage: {LANG}")
        rendered = synthesizer.synthesize_text(TEXT, language=LANG, voice=VOICE)
        output = artifact_path("english_demo.wav")
        rendered.save_wav(output)
        print(f"Created {output} ({len(rendered.audio) / rendered.sample_rate:.2f} seconds)")

        print("\n=== Example 2: Phonemes-to-Speech ===")
        text_request = SynthesisSegment(id="english-text", text=TEXT, language=LANG, voice=VOICE)
        phonemes = synthesizer.prepare(text_request).phonemes
        print(f"Phonemes: {phonemes[:100]}...")
        print(f"Phoneme length: {len(phonemes)} characters")
        phoneme_result = synthesizer.synthesize(
            SynthesisSegment(
                id="english-phonemes",
                text=phonemes,
                language=LANG,
                voice=VOICE,
                phonemes=phonemes,
            )
        )
        output = artifact_path("english_phonemes_demo.wav")
        phoneme_result.save_wav(output)
        print(
            f"Created {output} ({len(phoneme_result.audio) / phoneme_result.sample_rate:.2f} seconds)"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
