#!/usr/bin/env python3
"""
English homographs stress test example using pykokoro.

This example demonstrates how pykokoro handles homographs - words that are spelled
the same but have different pronunciations and meanings depending on context.
Examples include: lead (to guide) vs lead (metal), read (present) vs read (past),
wind (breeze) vs wind (to turn), tear (rip) vs tear (crying), etc.

The TTS engine must use context to determine correct pronunciation.

Usage:
    python examples/homographs.py

Output:
    homographs_demo.wav - Generated English speech testing homograph pronunciation
"""

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Text with extensive homograph usage
TEXT = """
A Homograph Challenge Text: The Executive's Dilemma

The executive decided to present her team with a gift to acknowledge their lead role
in completing the project. She asked them to read the report that was read aloud at
yesterday's meeting. The document detailed how to properly wind the rope and wind down
after long hours.

One employee began to object to the proposal, noting that they'd need to object to the
plan's mechanics. "We need to live up to our standards," she said, "especially since we
live near the production facility."

The discussion revealed a tear in the fabric of their strategy—they would need to tear
down old systems. Someone mentioned they saw a tear fall from the investor's eye during
the pitch.

"Don't desert this project," warned the manager, "or you'll regret abandoning us in the
desert market." The team's bass player suggested they lower the bass frequencies in
their audio materials.

The bow of the company ship pointed toward innovation. "We must all bow to market
demands," the CEO proclaimed. The musician would lead the orchestra, though some were
concerned she might lead them astray with experimental choices.

One accountant asked about the permit they'd need to permit construction. Another
employee noted that the rare sow would sow seeds in the experimental garden. There was
discussion about whether to close the warehouse doors to close the gap in their supply
chain.

They had to record this meeting—such an important record of their strategic direction.
The factory worker explained how to produce goods and the financial impact of their
produce line.

The consultant asked them to subject each proposal to scrutiny; every subject deserved
attention. Finally, someone mentioned they'd need to subject their entire operation to
this transformation.
"""

VOICE = "af_bella"  # American Female voice (good for clear articulation)
LANG = "en-us"  # American English


def main():
    """Generate English speech testing homograph pronunciation."""
    print("Initializing TTS engine...")
    pipe = KokoroPipeline(
        PipelineConfig(voice=VOICE, generation=GenerationConfig(lang=LANG, speed=1.0))
    )

    print("Testing homographs (same spelling, different pronunciation):")
    print("  - lead (guide) vs lead (metal)")
    print("  - read (present) vs read (past)")
    print("  - wind (breeze) vs wind (turn)")
    print("  - tear (rip) vs tear (crying)")
    print("  - object (thing) vs object (protest)")
    print("  - desert (abandon) vs desert (arid land)")
    print("  - bow (front of ship) vs bow (bend)")
    print("  - bass (fish) vs bass (low frequency)")
    print("  - live (alive) vs live (reside)")
    print("  - close (shut) vs close (near)")
    print("  - record (noun) vs record (verb)")
    print("  - produce (verb) vs produce (noun)")
    print("  - present (gift) vs present (show)")
    print("  - subject (topic) vs subject (to expose)")
    print(f"\nVoice: {VOICE}")
    print(f"Language: {LANG}")

    print("\nGenerating audio...")
    res = pipe.run(TEXT)
    samples, sample_rate = res.audio, res.sample_rate

    output_file = artifact_path("homographs_demo.wav")
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")
    print("\nNote: Listen carefully to check if pronunciations match context!")


if __name__ == "__main__":
    main()
