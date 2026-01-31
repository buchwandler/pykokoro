#!/usr/bin/env python3
"""
Demonstrate automatic pause insertion with pause_mode="auto".

This example shows how pause_mode="auto" automatically adds natural pauses
between clauses, sentences, and paragraphs for more natural-sounding speech
without manual pause markers.

Usage:
    python examples/automatic_pauses_demo.py

Output:
    automatic_pauses_demo.wav - Text with automatic natural pauses
"""

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig


def main():
    """Generate example with automatic pauses."""
    print("Initializing TTS engine...")
    generation = GenerationConfig(
        lang="en-us",
        pause_mode="auto",
        pause_clause=0.25,
        pause_sentence=0.5,
        pause_paragraph=2.0,
        pause_variance=0.05,
        random_seed=None,
    )
    pipe = KokoroPipeline(PipelineConfig(voice="af_sarah", generation=generation))

    # Text with multiple paragraphs, sentences, and clauses
    # No manual pause markers needed - pauses are added automatically!
    text = """
    The future of artificial intelligence is rapidly evolving. Machine learning
    models are becoming more sophisticated, efficient, and accessible to developers
    worldwide. This democratization of AI technology promises to revolutionize
    industries from healthcare to transportation.

    Neural networks, the foundation of modern AI, consist of interconnected layers
    that process information hierarchically. Each layer extracts increasingly
    complex features from the input data, enabling the network to learn patterns
    and make predictions. Deep learning, a subset of machine learning, uses many
    layers to achieve remarkable results in computer vision, natural language
    processing, and speech recognition.

    As we look to the future, the integration of AI into everyday life will
    continue to accelerate. From smart homes to autonomous vehicles, AI-powered
    systems are transforming how we live, work, and interact with technology.
    """

    print("=" * 70)
    print("Generating with AUTOMATIC pauses (pause_mode='auto')")
    print("=" * 70)
    print("\nKey features:")
    print("  • pause_mode='auto' - PyKokoro inserts pauses at boundaries")
    print("  • Automatic pause insertion:")
    print("    - Short pauses after clauses (within sentence)")
    print("    - Medium pauses after sentences (within paragraph)")
    print("    - Long pauses after paragraphs")
    print("  • Gaussian variance for natural rhythm")
    print("  • NO manual pause markers needed!")
    print()

    print("Processing text...")
    print(f"Text length: {len(text)} characters")
    print()

    # Generate with automatic pauses
    res = pipe.run(text)
    samples, sample_rate = res.audio, res.sample_rate

    output_file = "automatic_pauses_demo.wav"
    sf.write(output_file, samples, sample_rate)
    duration = len(samples) / sample_rate

    print("✓ Generation complete!")
    print()
    print("=" * 70)
    print(f"Generated: {output_file}")
    print(f"Duration: {duration:.2f} seconds ({duration / 60:.1f} minutes)")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total samples: {len(samples):,}")
    print("=" * 70)
    print()

    print("Comparison with other approaches:")
    print()
    print("1. TTS-controlled pauses (default):")
    print("   pipe.run(text, voice='af_sarah')")
    print("   → TTS generates natural pauses automatically")
    print()
    print("2. SSMD break markers:")
    print("   text = 'Hello ...c world ...s How are you?'")
    print("   pipe.run(text, voice='af_sarah')")
    print("   → SSMD breaks automatically detected and processed")
    print()
    print("3. Automatic pause control (this example):")
    print(
        "   pipe.run(text, voice='af_sarah', "
        "generation=GenerationConfig(pause_mode='auto'))"
    )
    print("   → PyKokoro controls pauses precisely at linguistic boundaries")
    print()

    print("Tips for best results:")
    print("  • Use pause_mode='auto' for automatic boundary pauses")
    print("  • Use pause_mode='tts' (default) to let TTS handle pauses naturally")
    print("  • Adjust pause_clause/sentence/paragraph to match your content style")
    print("  • Set pause_variance=0.0 for consistent timing (e.g., training data)")
    print("  • Set random_seed for reproducible output")
    print()


if __name__ == "__main__":
    main()
