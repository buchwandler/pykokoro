#!/usr/bin/env python3
"""
Benchmark different ONNX Runtime providers.

This script compares CPU, CUDA, OpenVINO, DirectML, and CoreML performance
for TTS generation. It helps you determine which provider gives the best
performance on your system.

Usage:
    python examples/gpu_benchmark.py

Output:
    Performance comparison of available providers
"""

import time

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Test text - medium length to get meaningful timings
TEXT = (
    """
The quick brown fox jumps over the lazy dog.
She sells seashells by the seashore.
Peter Piper picked a peck of pickled peppers.
How much wood would a woodchuck chuck if a woodchuck could chuck wood?
"""
    * 5
)  # Repeat for longer audio


def benchmark_provider(
    provider_name: str, provider_options: dict | None = None
) -> tuple[float, float] | None:
    """
    Benchmark a specific provider.

    Args:
        provider_name: Name of the provider to test
        provider_options: Optional provider configuration

    Returns:
        Tuple of (elapsed_time, real_time_factor) or None if failed
    """
    try:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {provider_name.upper()}")
        if provider_options:
            print(f"Options: {provider_options}")
        print(f"{'=' * 60}")

        # Initialize with specific provider
        pipe = KokoroPipeline(
            PipelineConfig(
                voice="af_bella",
                provider=provider_name,  # type: ignore[arg-type]
                provider_options=provider_options,
                generation=GenerationConfig(lang="en-us", speed=1.0),
                # Optional: Test with different model sources
                # model_source="github",  # GitHub source
                # model_variant="v1.0",  # English model
                # model_quality="fp16",  # Use fp16 quality
            )
        )

        # Warmup run (important for GPU providers)
        print("Running warmup...")
        pipe.run(TEXT[:100])

        # Actual benchmark
        print("Running benchmark...")
        start = time.time()
        res = pipe.run(TEXT)
        samples, sr = res.audio, res.sample_rate
        elapsed = time.time() - start

        # Calculate metrics
        audio_duration = len(samples) / sr
        rtf = elapsed / audio_duration  # Real-time factor

        print(f"✓ Generated {audio_duration:.2f}s of audio in {elapsed:.2f}s")
        print(f"  Real-time factor: {rtf:.2f}x")
        kokoro_backend = getattr(pipe.synth, "_kokoro", None)
        if kokoro_backend and kokoro_backend._session:
            print(f"  Actual providers: {kokoro_backend._session.get_providers()}")

        # Save sample output
        suffix = f"_{list(provider_options.keys())[0]}" if provider_options else ""
        output_file = artifact_path(f"benchmark_{provider_name}{suffix}.wav")
        sf.write(output_file, samples, sr)
        print(f"  Saved sample to: {output_file}")

        return elapsed, rtf

    except Exception as e:
        print(f"✗ {provider_name.upper()} failed: {e}")
        return None


def main():
    """Run benchmarks for all available providers."""
    print("=" * 60)
    print("PYKOKORO PROVIDER BENCHMARK")
    print("=" * 60)
    print(f"\nTest text length: {len(TEXT)} characters")
    print("Voice: af_bella")
    print("Speed: 1.0")

    # Test all providers with default settings
    providers_to_test = ["cpu", "cuda", "openvino", "directml", "coreml"]
    results = {}

    for provider in providers_to_test:
        result = benchmark_provider(provider)
        if result:
            elapsed, rtf = result
            results[provider] = (elapsed, rtf)

    # Additional tests with custom options
    print(f"\n{'=' * 60}")
    print("TESTING CUSTOM PROVIDER OPTIONS")
    print(f"{'=' * 60}")

    # Test CPU with different thread counts
    for threads in [2, 4, 8]:
        provider_opts = {"intra_op_num_threads": threads}
        result = benchmark_provider("cpu", provider_opts)
        if result:
            elapsed, rtf = result
            results[f"cpu_{threads}t"] = (elapsed, rtf)

    # Test OpenVINO with different precisions (if available)
    try:
        import onnxruntime as rt

        if "OpenVINOExecutionProvider" in rt.get_available_providers():
            for precision in ["FP32", "FP16"]:
                provider_opts = {"precision": precision, "num_of_threads": 8}
                result = benchmark_provider("openvino", provider_opts)
                if result:
                    elapsed, rtf = result
                    results[f"openvino_{precision.lower()}"] = (elapsed, rtf)
    except ImportError:
        pass

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if not results:
        print("No providers were successfully tested!")
        print("\nTip: Install GPU/accelerator packages:")
        print("  pip install pykokoro[gpu]       # NVIDIA CUDA")
        print("  pip install pykokoro[openvino]  # Intel OpenVINO")
        print("  pip install pykokoro[directml]  # Windows DirectML")
        print("  pip install pykokoro[coreml]    # macOS CoreML")
        return

    # Sort by elapsed time (fastest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1][0])

    print(f"\n{'Provider':<20} {'Time (s)':>10} {'RTF':>8} {'Speedup':>10}")
    print("-" * 60)

    baseline_time = sorted_results[-1][1][0]  # Slowest (usually CPU)

    for provider, (elapsed, rtf) in sorted_results:
        speedup = baseline_time / elapsed
        print(f"{provider.upper():<20} {elapsed:>10.2f} {rtf:>8.2f}x {speedup:>9.1f}x")

    # Recommendations
    print(f"\n{'=' * 60}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 60}")

    fastest = sorted_results[0]
    print(f"\nFastest configuration: {fastest[0].upper()}")
    print(f"  Time: {fastest[1][0]:.2f}s (RTF: {fastest[1][1]:.2f}x)")

    if fastest[1][1] < 0.5:
        print("\n✓ Excellent performance! Real-time factor < 0.5x")
    elif fastest[1][1] < 1.0:
        print("\n✓ Good performance! Real-time factor < 1.0x")
    else:
        print("\n⚠ Performance could be improved. Consider using a GPU accelerator.")

    print("\nTo use the fastest configuration in your code:")
    base_provider = fastest[0].split("_")[0]
    if "_" in fastest[0]:
        # Has custom options
        if "t" in fastest[0]:  # Thread count
            threads = fastest[0].split("_")[1].replace("t", "")
            print("  kokoro = pykokoro.Kokoro(")
            print(f'      provider="{base_provider}",')
            print(f'      provider_options={{"intra_op_num_threads": {threads}}}')
            print("  )")
        elif fastest[0].startswith("openvino"):  # Precision
            precision = fastest[0].split("_")[1].upper()
            print("  kokoro = pykokoro.Kokoro(")
            print(f'      provider="{base_provider}",')
            print(f'      provider_options={{"precision": "{precision}", "num_of_threads": 8}}')
            print("  )")
    else:
        print(f'  kokoro = pykokoro.Kokoro(provider="{fastest[0]}")')

    print("\n" + "=" * 60)
    print("Provider configuration options:")
    print("=" * 60)
    print("\nCommon options:")
    print("  - intra_op_num_threads: Thread count for parallel operations")
    print("  - inter_op_num_threads: Thread count across operations")
    print("  - graph_optimization_level: 0-3 or GraphOptimizationLevel enum")
    print("\nOpenVINO-specific:")
    print("  - precision: 'FP32', 'FP16', 'BF16'")
    print("  - device_type: 'CPU_FP32', 'GPU', etc.")
    print("  - num_of_threads: Number of inference threads")
    print("\nCUDA-specific:")
    print("  - device_id: GPU device ID")
    print("  - gpu_mem_limit: Memory limit in bytes")
    print("  - arena_extend_strategy: 'kNextPowerOfTwo', 'kSameAsRequested'")
    print("\nSee examples/provider_config_demo.py for more details.")


if __name__ == "__main__":
    main()
