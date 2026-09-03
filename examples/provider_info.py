#!/usr/bin/env python3
"""
Display available ONNX Runtime providers and test TTS with auto provider selection.

This example:
1. Lists all ONNX Runtime execution providers available on your system
2. Shows which providers are currently installed
3. Generates speech using automatic provider selection
4. Provides installation instructions for unavailable providers

Usage:
    python examples/provider_info.py

Output:
    provider_auto_demo.wav - Generated audio using auto-selected provider
"""

from __future__ import annotations

import onnxruntime as ort

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.onnx_session import (
    get_available_execution_providers,
    resolve_execution_provider,
)

# Test sentence
TEST_TEXT = "PyKokoro supports multiple hardware acceleration providers for optimal performance."


PROVIDER_INFO = {
    "CPUExecutionProvider": {
        "name": "CPU",
        "description": "Standard CPU execution (always available)",
        "install": "Built-in",
        "performance": "Baseline",
    },
    "CUDAExecutionProvider": {
        "name": "NVIDIA CUDA",
        "description": "NVIDIA GPU acceleration",
        "install": "pip install pykokoro[gpu]",
        "performance": "Excellent for NVIDIA GPUs",
    },
    "OpenVINOExecutionProvider": {
        "name": "Intel OpenVINO",
        "description": "Intel CPU/GPU/VPU optimization",
        "install": "pip install pykokoro[openvino]",
        "performance": "Excellent for Intel hardware",
    },
    "DmlExecutionProvider": {
        "name": "DirectML",
        "description": "Windows DirectML (AMD/Intel/NVIDIA)",
        "install": "pip install pykokoro[directml]",
        "performance": "Good for Windows GPUs",
    },
    "CoreMLExecutionProvider": {
        "name": "Apple CoreML",
        "description": "Apple Silicon/GPU acceleration",
        "install": "pip install pykokoro[coreml]",
        "performance": "Excellent for Apple Silicon",
    },
    "TensorrtExecutionProvider": {
        "name": "NVIDIA TensorRT",
        "description": "NVIDIA TensorRT optimization",
        "install": "pip install onnxruntime-gpu",
        "performance": "Excellent for NVIDIA GPUs",
    },
    "ROCMExecutionProvider": {
        "name": "AMD ROCm",
        "description": "AMD GPU acceleration",
        "install": "pip install onnxruntime-rocm",
        "performance": "Excellent for AMD GPUs",
    },
    "NnapiExecutionProvider": {
        "name": "Android NNAPI",
        "description": "Android Neural Networks API acceleration",
        "install": "Use an ONNX Runtime build with NNAPI support",
        "performance": "Depends on Android hardware",
    },
    "XnnpackExecutionProvider": {
        "name": "XNNPACK",
        "description": "Optimized CPU kernels for supported platforms",
        "install": "Use an ONNX Runtime build with XNNPACK support",
        "performance": "Good on supported CPUs",
    },
}


def print_provider_info():
    """Print information about available ONNX Runtime providers."""
    print("=" * 70)
    print("ONNX RUNTIME EXECUTION PROVIDERS")
    print("=" * 70)

    available_providers = get_available_execution_providers()

    print(f"\nONNX Runtime {ort.__version__}")
    print(f"Found {len(available_providers)} provider(s) on your system:\n")

    # Print available providers
    print("✓ AVAILABLE PROVIDERS:")
    print("-" * 70)
    for provider in available_providers:
        info = PROVIDER_INFO.get(
            provider,
            {"name": provider, "description": "Unknown provider"},
        )
        print(f"  • {info['name']:<20} - {info['description']}")
        if "performance" in info:
            print(f"    Performance: {info['performance']}")
    print()

    # Print unavailable providers with installation instructions
    all_known_providers = set(PROVIDER_INFO)
    unavailable = all_known_providers - set(available_providers)

    if unavailable:
        print("✗ UNAVAILABLE PROVIDERS (install for better performance):")
        print("-" * 70)
        for provider in sorted(unavailable):
            info = PROVIDER_INFO[provider]
            print(f"  • {info['name']:<20} - {info['description']}")
            print(f"    Install: {info['install']}")
            print(f"    Performance: {info['performance']}")
            print()


def test_auto_provider():
    """Test TTS generation with auto provider selection."""
    print("=" * 70)
    print("TESTING AUTO PROVIDER SELECTION")
    print("=" * 70)

    available_providers = get_available_execution_providers()
    selected_provider = resolve_execution_provider(
        "auto",
        available=available_providers,
    )

    print("\nInitializing Kokoro with provider='auto'...")
    print(f"Resolved provider: {selected_provider}")

    print("\nGenerating speech...")
    print(f'Text: "{TEST_TEXT}"')
    print("Voice: af_bella")

    config = PipelineConfig(
        voice="af_bella",
        generation=GenerationConfig(lang="en-us", speed=1.0),
        provider="auto",
    )
    output_file = artifact_path("provider_auto_demo.wav")
    with KokoroPipeline(config) as pipeline:
        result = pipeline.run(TEST_TEXT)
        result.save_wav(output_file)

    duration = len(result.audio) / result.sample_rate
    print("\n✓ Success!")
    print(f"  Generated {duration:.2f}s of audio")
    print(f"  Saved to: {output_file}")


def print_recommendations():
    """Print recommendations based on available providers."""
    print()
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    available = get_available_execution_providers()

    if len(available) == 1 and available[0] == "CPUExecutionProvider":
        print("\n⚠ Only CPU provider is available.")
        print("\nFor faster performance, consider installing GPU acceleration:")
        print("  • NVIDIA GPU:  pip install pykokoro[gpu]")
        print("  • Windows GPU: pip install pykokoro[directml]")
        print("  • Apple:       pip install pykokoro[coreml]")
    else:
        print(f"\n✓ You have {len(available)} provider(s) available!")
        if "CUDAExecutionProvider" in available:
            print("  Recommended: CUDA for optimal NVIDIA GPU performance")
        elif "CoreMLExecutionProvider" in available:
            print("  Recommended: CoreML for optimal Apple Silicon performance")
        elif "DmlExecutionProvider" in available:
            print("  Recommended: DirectML for Windows GPU acceleration")
        elif "OpenVINOExecutionProvider" in available:
            print("  Note: OpenVINO is installed but incompatible with Kokoro models")
            print("  Using CPU fallback automatically")

    print("\nUsage in your code:")
    print("  # Automatic selection (recommended)")
    print("  pipe = KokoroPipeline(PipelineConfig(provider='auto'))")
    print()
    print("  # Or explicit selection")
    print("  pipe = KokoroPipeline(PipelineConfig(provider='cuda'))  # Force CUDA")
    print("  pipe = KokoroPipeline(PipelineConfig(provider='cpu'))   # Force CPU")


def main():
    """Main function."""
    print("\n")
    print_provider_info()
    test_auto_provider()
    print_recommendations()
    print("\n")


if __name__ == "__main__":
    main()
