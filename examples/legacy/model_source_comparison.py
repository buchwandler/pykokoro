#!/usr/bin/env python3
"""
Multi-Source Model Comparison Example for pykokoro.

This example demonstrates how to generate audio using all 4 combinations:
1. HuggingFace v1.0 - Multi-language, 54 voices, 8 quality options
   from onnx-community/Kokoro-82M-v1.0-ONNX
2. HuggingFace v1.1-zh - English + Chinese, 103 voices, 8 quality options
   from onnx-community/Kokoro-82M-v1.1-zh-ONNX
3. GitHub v1.0 - Multi-language, 54 voices, 4 quality options
   from the central catalog GitHub release
4. Upstream-only profiles use the pinned catalog HTTPS distribution

Usage:
    python examples/model_source_comparison.py

Output:
    hf_v1.0_demo.wav - Audio from HuggingFace v1.0 model
    hf_v1.1_zh_demo.wav - Audio from HuggingFace v1.1-zh model
    github_v1.0_demo.wav - Audio from GitHub v1.0 model
    github_v1.1_zh_demo.wav - Audio from GitHub v1.1-zh model

The example shows how the same text is synthesized using all available model sources,
allowing you to compare quality, voices, and performance.

Note:
    - HuggingFace sources download individual voice files and combine them
    - GitHub sources download pre-combined voices.bin files
    - v1.1-zh models have 103 voices (including English and Chinese voices)
    - Voice names are loaded dynamically from the model's voices.bin file
"""

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.onnx_backend import Kokoro
from pykokoro.stages.audio_generation.onnx import OnnxAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
from pykokoro.stages.phoneme_processing.onnx import OnnxPhonemeProcessorAdapter

# Text for English models (HuggingFace and GitHub v1.0)
ENGLISH_TEXT = (
    "Hello! This is a demonstration of the PyKokoro text-to-speech library. "
    "We are comparing different model sources to show their capabilities. "
    "Technology enables us to communicate across boundaries."
)

# Text for Chinese model (GitHub v1.1-zh)
CHINESE_TEXT = (
    "你好！这是PyKokoro文本转语音库的演示。"
    "我们正在比较不同的模型来源，以展示它们的能力。"
    "技术使我们能够跨越界限进行交流。"
)


def main():
    """Generate audio using all four model source/variant combinations."""

    def build_pipeline(kokoro_backend: Kokoro, voice: str, lang: str) -> KokoroPipeline:
        return KokoroPipeline(
            PipelineConfig(
                voice=voice,
                generation=GenerationConfig(lang=lang, speed=1.0),
            ),
            phoneme_processing=OnnxPhonemeProcessorAdapter(kokoro_backend),
            audio_generation=OnnxAudioGenerationAdapter(kokoro_backend),
            audio_postprocessing=OnnxAudioPostprocessingAdapter(kokoro_backend),
        )

    # =========================================================================
    # Example 1: HuggingFace v1.0 Model
    # =========================================================================
    print("=" * 70)
    print("Example 1: HuggingFace v1.0 Model Source")
    print("=" * 70)
    print("Source: onnx-community/Kokoro-82M-v1.0-ONNX")
    print(f"Text: {ENGLISH_TEXT[:50]}...")

    print("\nInitializing TTS engine with HuggingFace v1.0 model...")
    kokoro_hf_v10 = Kokoro(
        model_source="huggingface",
        model_variant="v1.0",
        model_quality="fp32",
    )

    # Get and display available voices
    available_voices = kokoro_hf_v10.get_voices()
    print(f"Available voices: {len(available_voices)}")
    print(f"Sample voices: {', '.join(available_voices[:5])}...")

    # Use af_sarah as default voice
    voice_to_use = "af_sarah" if "af_sarah" in available_voices else available_voices[0]
    print(f"\nGenerating audio with HuggingFace v1.0 using voice '{voice_to_use}'...")
    pipeline_hf_v10 = build_pipeline(kokoro_hf_v10, voice_to_use, "en-us")
    result_hf_v10 = pipeline_hf_v10.run(ENGLISH_TEXT, voice=voice_to_use)
    samples_hf_v10, sample_rate = result_hf_v10.audio, result_hf_v10.sample_rate

    output_file_hf_v10 = artifact_path("hf_v1.0_demo.wav")
    sf.write(output_file_hf_v10, samples_hf_v10, sample_rate)

    duration_hf_v10 = len(samples_hf_v10) / sample_rate
    print(f"✓ Created {output_file_hf_v10}")
    print(f"  Duration: {duration_hf_v10:.2f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Samples: {len(samples_hf_v10):,}")

    kokoro_hf_v10.close()

    # =========================================================================
    # Example 2: HuggingFace v1.1-zh Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example 2: HuggingFace v1.1-zh Model Source")
    print("=" * 70)
    print("Source: onnx-community/Kokoro-82M-v1.1-zh-ONNX")
    print(f"Text: {ENGLISH_TEXT[:50]}...")

    print("\nInitializing TTS engine with HuggingFace v1.1-zh model...")
    kokoro_hf_v11zh = Kokoro(
        model_source="huggingface",
        model_variant="v1.1-zh",
        model_quality="q8",  # v1.1-zh supports all quantization levels
    )

    # Get and display available voices
    available_voices_v11zh = kokoro_hf_v11zh.get_voices()
    print(f"Available voices in v1.1-zh: {len(available_voices_v11zh)}")

    # Use af_maple (English female voice from v1.1-zh)
    voice_v11zh = "af_maple" if "af_maple" in available_voices_v11zh else available_voices_v11zh[0]
    print(f"Using English voice: {voice_v11zh}")
    print(f"\nGenerating audio with HuggingFace v1.1-zh using voice '{voice_v11zh}'...")
    pipeline_hf_v11zh = build_pipeline(kokoro_hf_v11zh, voice_v11zh, "en-us")
    result_hf_v11zh = pipeline_hf_v11zh.run(ENGLISH_TEXT, voice=voice_v11zh)
    samples_hf_v11zh, sample_rate = (
        result_hf_v11zh.audio,
        result_hf_v11zh.sample_rate,
    )

    output_file_hf_v11zh = artifact_path("hf_v1.1_zh_demo.wav")
    sf.write(output_file_hf_v11zh, samples_hf_v11zh, sample_rate)

    duration_hf_v11zh = len(samples_hf_v11zh) / sample_rate
    print(f"✓ Created {output_file_hf_v11zh}")
    print(f"  Duration: {duration_hf_v11zh:.2f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Samples: {len(samples_hf_v11zh):,}")

    kokoro_hf_v11zh.close()

    # =========================================================================
    # Example 3: GitHub v1.0 Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example 3: GitHub v1.0 Model Source")
    print("=" * 70)
    print("Source: buchwandler/kokoro-onnx-models catalog (v1.0)")
    print(f"Text: {ENGLISH_TEXT[:50]}...")

    print("\nInitializing TTS engine with GitHub v1.0 model...")
    kokoro_github_v10 = Kokoro(
        model_source="github",
        model_variant="v1.0",
        model_quality="fp16",  # GitHub v1.0 supports: fp32, fp16, fp16-gpu, q8
    )

    # Get and display available voices
    available_voices_github = kokoro_github_v10.get_voices()
    print(f"Available voices: {len(available_voices_github)}")
    print(f"Sample voices: {', '.join(available_voices_github[:5])}...")

    # Use the same voice as Example 1 if available
    voice_github = (
        voice_to_use if voice_to_use in available_voices_github else available_voices_github[0]
    )
    print(f"\nGenerating audio with GitHub v1.0 using voice '{voice_github}'...")
    pipeline_github_v10 = build_pipeline(kokoro_github_v10, voice_github, "en-us")
    result_github_v10 = pipeline_github_v10.run(ENGLISH_TEXT, voice=voice_github)
    samples_github_v10, sample_rate = (
        result_github_v10.audio,
        result_github_v10.sample_rate,
    )

    output_file_github_v10 = artifact_path("github_v1.0_demo.wav")
    sf.write(output_file_github_v10, samples_github_v10, sample_rate)

    duration_github_v10 = len(samples_github_v10) / sample_rate
    print(f"✓ Created {output_file_github_v10}")
    print(f"  Duration: {duration_github_v10:.2f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Samples: {len(samples_github_v10):,}")

    kokoro_github_v10.close()

    # =========================================================================
    # Example 4: GitHub v1.1-zh Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example 4: GitHub v1.1-zh Model Source")
    print("=" * 70)
    print("Source: buchwandler/kokoro-onnx-models catalog (v1.1-zh)")
    print("Note: Testing with English text and English voice from v1.1-zh model")
    print(f"Text: {ENGLISH_TEXT[:50]}...")

    print("\nInitializing TTS engine with GitHub v1.1-zh model...")
    kokoro_github_v11zh = Kokoro(
        model_source="github",
        model_variant="v1.1-zh",
        model_quality="fp32",  # GitHub v1.1-zh only supports fp32
    )

    # Get and display available voices
    available_voices_github_v11zh = kokoro_github_v11zh.get_voices()
    print(f"Available voices in v1.1-zh: {len(available_voices_github_v11zh)}")

    # Find English voices (af_maple, af_sol, bf_vale)
    english_voices = [
        v for v in available_voices_github_v11zh if v in ["af_maple", "af_sol", "bf_vale"]
    ]
    english_list = ", ".join(english_voices) if english_voices else "None found"
    print(f"English voices: {english_list}")

    # Use af_maple (English female voice) or first available
    voice_github_v11zh = (
        "af_maple"
        if "af_maple" in available_voices_github_v11zh
        else available_voices_github_v11zh[0]
    )
    print(f"\nGenerating audio with GitHub v1.1-zh using voice '{voice_github_v11zh}'...")
    pipeline_github_v11zh = build_pipeline(kokoro_github_v11zh, voice_github_v11zh, "en-us")
    result_github_v11zh = pipeline_github_v11zh.run(ENGLISH_TEXT, voice=voice_github_v11zh)
    samples_github_v11zh, sample_rate = (
        result_github_v11zh.audio,
        result_github_v11zh.sample_rate,
    )

    output_file_github_v11zh = artifact_path("github_v1.1_zh_demo.wav")
    sf.write(output_file_github_v11zh, samples_github_v11zh, sample_rate)

    duration_github_v11zh = len(samples_github_v11zh) / sample_rate
    print(f"✓ Created {output_file_github_v11zh}")
    print(f"  Duration: {duration_github_v11zh:.2f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Samples: {len(samples_github_v11zh):,}")

    kokoro_github_v11zh.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary - All 4 Model Source/Variant Combinations")
    print("=" * 70)
    print(f"HuggingFace v1.0:    {duration_hf_v10:.2f}s - {output_file_hf_v10}")
    print(f"HuggingFace v1.1-zh: {duration_hf_v11zh:.2f}s - {output_file_hf_v11zh}")
    print(f"GitHub v1.0:         {duration_github_v10:.2f}s - {output_file_github_v10}")
    print(f"GitHub v1.1-zh:      {duration_github_v11zh:.2f}s - {output_file_github_v11zh}")
    print("\nComparison complete! You can now listen to the generated audio files.")
    print("\nModel Source Characteristics:")
    print("  • HuggingFace v1.0:    54 voices, 8 quality options (fp32, fp16, q8, q4, etc.)")
    print("  • HuggingFace v1.1-zh: 103 voices, 8 quality options (fp32, fp16, q8, q4, etc.)")
    print("  • GitHub v1.0:         54 voices, 4 quality options (fp32, fp16, fp16-gpu, q8)")
    print("  • GitHub v1.1-zh:      103 voices, 1 quality option (fp32 only)")
    print("\nKey Differences:")
    print("  • v1.0 models: 54 voices (English, Spanish, French, German, etc.)")
    print("  • v1.1-zh models: 103 voices (includes all v1.0 + Chinese voices)")
    print("  • HuggingFace: More quantization options, downloads individual voice files")
    print("  • GitHub: Faster setup with pre-combined voices.bin, fewer quality options")


if __name__ == "__main__":
    main()
