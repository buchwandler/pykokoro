#!/usr/bin/env python3
"""
English contractions stress test example using pykokoro.

This example demonstrates how pykokoro handles various English contractions
including 've (have), 's (has/is), 'd (had/would), and past tense -ed forms.

Usage:
    python examples/contractions.py

Output:
    contractions_demo.wav - Generated English speech testing contractions
"""

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Text with extensive contraction usage
TEXT = """
I've been waiting for this moment. You've already heard the news, haven't you?
We've worked together for years, and they've never let us down.

She's completed the project. He's finished his work too.
It's been a long day, but that's just how it goes sometimes.

I'd hoped to see you earlier. You'd mentioned you'd be here by noon.
We'd planned everything carefully, and they'd promised to help.

Yesterday, I walked to the store and talked with the owner.
She baked fresh bread and served it warm.
The children played outside and laughed all afternoon.

I've learned so much. You've taught me well.
He's mastered the technique. She's perfected her craft.

I'd traveled there before. You'd asked about it, hadn't you?
They'd finished early, so we'd decided to celebrate.

The team worked tirelessly. Everyone contributed their best efforts.
She mentioned the deadline. He questioned the results.
They organized the event beautifully.

I've never seen anything like it. You've got to be kidding!
We've accomplished what seemed impossible. They've exceeded all expectations.

I'd thought it was finished. She'd believed we'd succeed.
He'd promised he'd complete it. We'd assumed they'd arrive on time.

The students studied hard and passed their exams.
The flowers bloomed and attracted countless butterflies.
The musicians practiced daily and improved dramatically.

Haven't I told you before? Hasn't she mentioned this?
Hadn't we discussed this already? Wouldn't you agree?

I've got this. You've done well. He's right about that.
She's wrong, isn't she? It's complicated, hasn't it been?

I'd love to help. You'd better hurry. They'd rather wait.
We'd sooner leave than stay in these conditions.
"""

VOICE = "af_bella"  # American Female voice (good for clear articulation)
LANG = "en-us"  # American English


def main():
    """Generate English speech testing contractions and past tense forms."""
    print("Initializing TTS engine...")
    pipe = KokoroPipeline(
        PipelineConfig(voice=VOICE, generation=GenerationConfig(lang=LANG, speed=1.0))
    )

    print("Testing contractions: 've (have), 's (has/is), 'd (had/would)")
    print("Testing past tense: -ed endings and irregular forms")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")

    print("\nGenerating audio...")
    res = pipe.run(TEXT)
    samples, sample_rate = res.audio, res.sample_rate

    output_file = artifact_path("contractions_demo.wav")
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
