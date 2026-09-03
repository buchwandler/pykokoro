#!/usr/bin/env python3
"""
Hindi TTS example using pykokoro.

This example demonstrates text-to-speech synthesis in Hindi
using the Kokoro model with Hindi voices.

Usage:
    python examples/hindi.py

Output:
    hindi_demo.wav - Generated Hindi speech audio

Available Hindi voices:
    - hf_alpha, hf_beta (female)
    - hm_omega, hm_psi (male)
"""

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Hindi wisdom about life and perseverance
TEXT = "जीवन में सफलता उन्हें मिलती है जो कभी हार नहीं मानते। हर सुबह एक नई शुरुआत है, हर दिन एक नया अवसर है।"

VOICE = "hf_alpha"  # Hindi Female voice
LANG = "hi"  # Hindi


def main():
    """Generate Hindi speech audio."""
    print("Initializing TTS engine...")
    pipe = KokoroPipeline(
        PipelineConfig(voice=VOICE, generation=GenerationConfig(lang=LANG, speed=1.0))
    )

    print(f"Text: {TEXT}")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")

    print("\nGenerating audio...")
    res = pipe.run(TEXT)
    samples, sample_rate = res.audio, res.sample_rate

    output_file = artifact_path("hindi_demo.wav")
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
