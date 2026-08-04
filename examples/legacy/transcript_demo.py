#!/usr/bin/env python3
"""
Transcript compile demo for pykokoro.

This example compiles text into a transcript dict, prints it,
then uses the transcript to generate audio.

Usage:
    python examples/transcript_demo.py

Output:
    transcript_demo.wav - Generated audio from transcript
"""

import json

import soundfile as sf

import pykokoro

TEXT = "Hello from the audio-ready transcript demo."
VOICE = "af_sarah"
LANG = "en-us"


def main() -> None:
    print("Initializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    print("\nCompiling transcript...")
    transcript = kokoro.compile(TEXT, voice=VOICE, lang=LANG)

    print("\nTranscript dict:")
    print(json.dumps(transcript, indent=2, ensure_ascii=False))

    print("\nGenerating audio from transcript...")
    samples, sample_rate = kokoro.create_from_transcript(transcript)

    output_file = "transcript_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"Created {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    kokoro.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
