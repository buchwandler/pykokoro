#!/usr/bin/env python3
"""
English punctuation stress test example using pykokoro.

This example demonstrates how pykokoro handles various punctuation marks
including semicolons, colons, commas, periods, exclamation marks, question
marks, em dashes, ellipses, quotes, and parentheses.

Usage:
    python examples/punctuation.py

Output:
    punctuation_demo.wav - Generated English speech with heavy punctuation
"""

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Text with heavy punctuation usage
TEXT = """
"Well," said the professor; "this is quite extraordinary!"

The experiment — which took years to complete — yielded surprising results:
success rates of 95%, 87%, and 72% (in that order).

"But wait..." she paused, "are you absolutely sure?"

Yes! No? Maybe... Who knows: life is full of mysteries;
that's what makes it interesting.

Consider this: the data shows (a) increased efficiency;
(b) reduced costs; and (c) improved outcomes — all remarkable achievements!

"To be, or not to be?" — that is the question.

He shouted: "Eureka!" Then whispered... "Finally."

The results were: excellent (A+), good (B), average (C);
however — and this is important — none failed!

"Why?" she asked. "Because," he replied, "science never sleeps..."
"""

VOICE = "af_bella"  # American Female voice (good for expressive reading)
LANG = "en-us"  # American English


def main():
    """Generate English speech with heavy punctuation."""
    print("Initializing TTS engine...")
    pipe = KokoroPipeline(
        PipelineConfig(voice=VOICE, generation=GenerationConfig(lang=LANG, speed=1.0))
    )

    print("Text with punctuation marks: ; : , . ! ? — … \" ( ) ' '")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")

    print("\nGenerating audio...")
    res = pipe.run(TEXT)
    samples, sample_rate = res.audio, res.sample_rate

    output_file = artifact_path("punctuation_demo.wav")
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
