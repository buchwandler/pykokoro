#!/usr/bin/env python3
"""
Demonstrate inter-word pause control using pykokoro with SSMD break syntax.

This example shows how to use SSMD break markers (...c, ...s, ...p, ...500ms)
to control timing in speech synthesis. Break markers are automatically
detected and processed.

SSMD Break Markers:
- ...n - No pause (0ms)
- ...w - Weak pause (150ms)
- ...c - Clause/comma pause (300ms)
- ...s - Sentence pause (600ms)
- ...p - Paragraph pause (1000ms)
- ...500ms - Custom pause (500 milliseconds)
- ...2s - Custom pause (2 seconds)

Note: Bare ... (ellipsis) is NOT treated as a pause.

Usage:
    python examples/pauses_demo.py

Output:
    example1_basic_pauses.wav - Basic pause demonstration
    example2_custom_durations.wav - Custom pause durations
    example3_leading_pause.wav - Leading pause example
"""

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig


def main():
    """Generate example audio files with pauses."""
    print("Initializing TTS engine...")
    pipe = KokoroPipeline(
        PipelineConfig(voice="af_sarah", generation=GenerationConfig(lang="en-us", speed=1.0))
    )

    # Example 1: Basic SSMD pause markers
    print("\n" + "=" * 60)
    print("Example 1: Basic SSMD Pauses")
    print("=" * 60)

    text1 = "Chapter 5 ...p I'm Klaus. ...c Welcome to the show!"

    print(f"Text: {text1}")
    print("Pause markers: ...c = 0.3s, ...s = 0.6s, ...p = 1.0s")

    res1 = pipe.run(text1, voice="am_michael")
    samples, sample_rate = res1.audio, res1.sample_rate

    output1 = artifact_path("example1_basic_pauses.wav")
    sf.write(output1, samples, sample_rate)
    duration1 = len(samples) / sample_rate
    print(f"✓ Generated: {output1}")
    print(f"  Duration: {duration1:.2f}s")

    # Example 2: Custom pause durations
    print("\n" + "=" * 60)
    print("Example 2: Custom Pause Durations")
    print("=" * 60)

    text2 = "Quick pause ...c Medium pause ...s Long pause ...p Done!"

    print(f"Text: {text2}")
    print("Custom durations: ...c = 0.2s, ...s = 0.5s, ...p = 1.5s")

    res2 = pipe.run(
        text2,
        voice="af_sarah",
        generation=GenerationConfig(
            lang="en-us",
            speed=1.0,
            pause_clause=0.2,
            pause_sentence=0.5,
            pause_paragraph=1.5,
        ),
    )
    samples, sample_rate = res2.audio, res2.sample_rate

    output2 = artifact_path("example2_custom_durations.wav")
    sf.write(output2, samples, sample_rate)
    duration2 = len(samples) / sample_rate
    print(f"✓ Generated: {output2}")
    print(f"  Duration: {duration2:.2f}s")

    # Example 3: Leading pause
    print("\n" + "=" * 60)
    print("Example 3: Leading Pause")
    print("=" * 60)

    text3 = "...p After a long pause, we begin speaking."

    print(f"Text: {text3}")
    print("Note: Pause marker at start creates silence before speech")

    res3 = pipe.run(text3, voice="am_adam")
    samples, sample_rate = res3.audio, res3.sample_rate

    output3 = artifact_path("example3_leading_pause.wav")
    sf.write(output3, samples, sample_rate)
    duration3 = len(samples) / sample_rate
    print(f"✓ Generated: {output3}")
    print(f"  Duration: {duration3:.2f}s (includes 1.0s initial silence)")

    # Example 4: Consecutive pauses
    print("\n" + "=" * 60)
    print("Example 4: Consecutive Pauses (Additive)")
    print("=" * 60)

    text4 = "First sentence. ...p ...s Second sentence after a very long pause."

    print(f"Text: {text4}")
    print("Note: Consecutive pauses add together (1.0s + 0.6s = 1.6s)")

    res4 = pipe.run(text4, voice="af_bella")
    samples, sample_rate = res4.audio, res4.sample_rate

    output4 = artifact_path("example4_consecutive_pauses.wav")
    sf.write(output4, samples, sample_rate)
    duration4 = len(samples) / sample_rate
    print(f"✓ Generated: {output4}")
    print(f"  Duration: {duration4:.2f}s")

    # Example 5: Custom time-based pauses
    print("\n" + "=" * 60)
    print("Example 5: Custom Time-Based Pauses")
    print("=" * 60)

    text5 = "Wait ...500ms please ...2s Thank you!"

    print(f"Text: {text5}")
    print("Note: Use ...500ms or ...2s for exact time pauses")

    res5 = pipe.run(text5, voice="af_nicole")
    samples, sample_rate = res5.audio, res5.sample_rate

    output5 = artifact_path("example5_custom_time_pauses.wav")
    sf.write(output5, samples, sample_rate)
    duration5 = len(samples) / sample_rate
    print(f"✓ Generated: {output5}")
    print(f"  Duration: {duration5:.2f}s")

    print("\n" + "=" * 60)
    print("All examples generated successfully!")
    print("=" * 60)
    print("\nTotal files created: 5")
    total_duration = duration1 + duration2 + duration3 + duration4 + duration5
    print(f"Total duration: {total_duration:.2f}s")


if __name__ == "__main__":
    main()
