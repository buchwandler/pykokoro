#!/usr/bin/env python3
"""
Voice mixing/blending demonstration using pykokoro.

This example shows how to blend multiple voices together to create unique
hybrid voices. Voice blending combines the characteristics of two or more
voices using weighted averaging.

Features demonstrated:
- Simple 50/50 voice blends
- Custom weight distributions (e.g., 70/30, 33/33/34)
- Blending voices with different genders
- Blending voices with different accents
- Using CLI-style blend strings

Usage:
    python examples/voices.py

Output:
    voices_demo.wav - Audio showcasing various voice blends
"""

import numpy as np
import soundfile as sf

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.onnx_backend import Kokoro
from pykokoro.stages.audio_generation.onnx import OnnxAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
from pykokoro.stages.phoneme_processing.onnx import OnnxPhonemeProcessorAdapter
from pykokoro.voice_manager import VoiceBlend

# Sample rate constant
SAMPLE_RATE = 24000


def create_silence(duration_seconds: float = 0.5) -> np.ndarray:
    """Create a silent pause."""
    return np.zeros(int(duration_seconds * SAMPLE_RATE), dtype=np.float32)


def main():
    """Generate voice blending demonstrations."""
    print("Initializing TTS engine...")
    kokoro_backend = Kokoro()
    pipeline = KokoroPipeline(
        PipelineConfig(
            voice="af_sarah",
            generation=GenerationConfig(lang="en-us", speed=1.0),
        ),
        phoneme_processing=OnnxPhonemeProcessorAdapter(kokoro_backend),
        audio_generation=OnnxAudioGenerationAdapter(kokoro_backend),
        audio_postprocessing=OnnxAudioPostprocessingAdapter(kokoro_backend),
    )

    available_voices = kokoro_backend.get_voices()
    if not available_voices:
        raise RuntimeError("No voices available. Download voices before running.")

    voice_lengths: dict[str, int] = {}

    def get_voice_length(voice_name: str) -> int | None:
        if voice_name in voice_lengths:
            return voice_lengths[voice_name]
        try:
            style = kokoro_backend.get_voice_style(voice_name)
        except KeyError:
            return None
        voice_lengths[voice_name] = style.shape[0]
        return voice_lengths[voice_name]

    def find_voice(
        preferred: str,
        target_length: int | None = None,
    ) -> tuple[str, int]:
        preferred_length = get_voice_length(preferred)
        if preferred_length is not None and (
            target_length is None or preferred_length == target_length
        ):
            return preferred, preferred_length

        prefix = preferred.split("_", maxsplit=1)[0]
        for candidate in available_voices:
            if not candidate.startswith(prefix + "_"):
                continue
            length = get_voice_length(candidate)
            if length is None:
                continue
            if target_length is None or length == target_length:
                return candidate, length

        for candidate in available_voices:
            length = get_voice_length(candidate)
            if length is None:
                continue
            if target_length is None or length == target_length:
                return candidate, length

        fallback = available_voices[0]
        length = get_voice_length(fallback)
        if length is None:
            raise RuntimeError("Unable to determine voice length for fallback voice.")
        return fallback, length

    # Test sentence
    test_text = (
        "This is a demonstration of voice blending technology. "
        "You can combine multiple voices to create unique hybrid voices."
    )

    audio_parts = []

    # Demo configurations: (description, voice_blend_string)
    demos = [
        # 1. Introduction with a single voice
        (
            "Introduction - Pure voice (af_sarah)",
            None,  # Will use direct voice parameter
            "af_sarah",
        ),
        # 2. 50/50 blends - Same gender
        (
            "Fifty-fifty blend of Sarah and Nicole",
            "af_sarah:50,af_nicole:50",
            None,
        ),
        (
            "Fifty-fifty blend of Michael and Adam",
            "am_michael:50,am_adam:50",
            None,
        ),
        # 3. 50/50 blends - Cross-gender
        (
            "Fifty-fifty blend of Sarah and Michael",
            "af_sarah:50,am_michael:50",
            None,
        ),
        # 4. 70/30 blend - Dominant voice
        (
            "Seventy-thirty blend - more Sarah, less Nicole",
            "af_sarah:70,af_nicole:30",
            None,
        ),
        # 5. Three-way blend
        (
            "Three-way blend - Sarah, Nicole, and Bella",
            "af_sarah:33,af_nicole:33,af_bella:34",
            None,
        ),
        # 6. Accent mixing - American & British
        (
            "Accent blend - American Sarah and British Emma",
            "af_sarah:50,bf_emma:50",
            None,
        ),
        # 7. Subtle blend (90/10)
        (
            "Subtle blend - ninety percent Sarah, ten percent Nicole",
            "af_sarah:90,af_nicole:10",
            None,
        ),
    ]

    print(f"\nGenerating {len(demos)} voice blend demonstrations...\n")

    for i, (description, blend_str, single_voice) in enumerate(demos, 1):
        print(f"[{i}/{len(demos)}] {description}")

        # Add announcement of what's being demonstrated
        announcement_text = f"Demonstration {i}. {description}."

        # Generate announcement with a neutral voice
        announcement_voice, _ = find_voice("af_sarah")
        announcement_result = pipeline.run(
            announcement_text,
            voice=announcement_voice,
        )
        audio_parts.append(announcement_result.audio)
        sample_rate = announcement_result.sample_rate
        audio_parts.append(create_silence(0.8))

        # Generate the test text with the blended/single voice
        if single_voice:
            # Use a single voice directly
            resolved_voice, _ = find_voice(single_voice)
            if resolved_voice != single_voice:
                print(f"  Using '{resolved_voice}' instead of '{single_voice}'")
            result = pipeline.run(
                test_text,
                voice=resolved_voice,
            )
            samples, sample_rate = result.audio, result.sample_rate
        else:
            # Parse and use voice blend
            blend = VoiceBlend.parse(blend_str)
            resolved_voices: list[tuple[str, float]] = []
            target_length: int | None = None
            for voice_name, weight in blend.voices:
                resolved_name, resolved_length = find_voice(voice_name, target_length)
                if target_length is None:
                    target_length = resolved_length
                if resolved_name != voice_name:
                    print(f"  Using '{resolved_name}' instead of '{voice_name}'")
                resolved_voices.append((resolved_name, weight))

            blend = VoiceBlend(
                voices=resolved_voices,
                interpolation=blend.interpolation,
            )
            result = pipeline.run(
                test_text,
                voice=blend,
            )
            samples, sample_rate = result.audio, result.sample_rate

        audio_parts.append(samples)
        audio_parts.append(create_silence(1.5))  # Longer pause between demos

    # Add conclusion
    print(f"[{len(demos) + 1}/{len(demos) + 1}] Conclusion")
    conclusion_text = (
        "This concludes the voice blending demonstration. "
        "You can use the --voice parameter with blend strings "
        "in the command line interface, "
        "or create VoiceBlend objects programmatically."
    )
    conclusion_voice, _ = find_voice("af_sarah")
    conclusion_result = pipeline.run(
        conclusion_text,
        voice=conclusion_voice,
    )
    audio_parts.append(conclusion_result.audio)
    sample_rate = conclusion_result.sample_rate

    # Concatenate all audio
    print("\nConcatenating audio segments...")
    final_audio = np.concatenate(audio_parts)

    # Save to file
    output_file = "voices_demo.wav"
    sf.write(output_file, final_audio, sample_rate)

    duration = len(final_audio) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")
    print("\nVoice blend format: 'voice1:weight1,voice2:weight2'")
    print("Example CLI usage:")
    print("  pykokoro sample 'Hello world' --voice 'af_sarah:50,am_michael:50'")

    # Cleanup
    kokoro_backend.close()


if __name__ == "__main__":
    main()
