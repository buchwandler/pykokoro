#!/usr/bin/env python3
"""
CPU Provider Benchmark - Parameter Optimization

This script benchmarks the CPU provider with different configuration parameters
to help you find the optimal settings for your system.

It tests:
- Different thread counts (intra_op_num_threads)
- Graph optimization levels
- Execution modes
- Memory arena settings

Usage:
    python examples/cpu_benchmark.py

Output:
    Performance comparison of different CPU configurations
"""

import os
import time
from typing import Any

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig

# Test text - medium length to get meaningful timings
TEXT = """
The quick brown fox jumps over the lazy dog.
She sells seashells by the seashore.
Peter Piper picked a peck of pickled peppers.
How much wood would a woodchuck chuck if a woodchuck could chuck wood?
"""

# Shorter text for quick tests
SHORT_TEXT = "The quick brown fox jumps over the lazy dog."


def benchmark_config(
    config_name: str,
    provider_options: dict[str, Any],
    text: str = TEXT,
    warmup: bool = True,
) -> tuple[float, float, int] | None:
    """
    Benchmark a specific CPU configuration.

    Args:
        config_name: Name/description of this configuration
        provider_options: Provider options to test
        text: Text to synthesize
        warmup: Whether to run warmup iteration

    Returns:
        Tuple of (elapsed_time, real_time_factor, audio_samples) or None if failed
    """
    try:
        # Initialize with specific configuration
        pipe = KokoroPipeline(
            PipelineConfig(
                voice="af_bella",
                provider="cpu",
                provider_options=provider_options,
                generation=GenerationConfig(lang="en-us", speed=1.0),
            )
        )

        # Warmup run
        if warmup:
            pipe.run(SHORT_TEXT)

        # Actual benchmark
        start = time.time()
        res = pipe.run(text)
        samples, sr = res.audio, res.sample_rate
        elapsed = time.time() - start

        # Calculate metrics
        audio_duration = len(samples) / sr
        rtf = elapsed / audio_duration  # Real-time factor

        return elapsed, rtf, len(samples)

    except Exception as e:
        print(f"✗ {config_name} failed: {e}")
        return None


def test_thread_counts():
    """Test different thread counts."""
    print("\n" + "=" * 70)
    print("TEST 1: Thread Count Optimization (intra_op_num_threads)")
    print("=" * 70)
    print("\nTesting different thread counts for parallel operations...")

    cpu_count = os.cpu_count() or 4
    thread_counts = [1, 2, 4, cpu_count // 2, cpu_count]
    # Remove duplicates and sort
    thread_counts = sorted(set(thread_counts))

    results = {}
    for threads in thread_counts:
        print(f"\nTesting {threads} thread(s)...", end=" ")
        config = {"intra_op_num_threads": threads}
        result = benchmark_config(f"{threads}_threads", config)
        if result:
            elapsed, rtf, samples = result
            results[threads] = (elapsed, rtf)
            print(f"✓ {elapsed:.2f}s (RTF: {rtf:.2f}x)")

    if results:
        print(f"\n{'Threads':<10} {'Time (s)':>10} {'RTF':>8} {'Speedup':>10}")
        print("-" * 50)
        baseline = max(results.values(), key=lambda x: x[0])[0]
        for threads, (elapsed, rtf) in sorted(results.items()):
            speedup = baseline / elapsed
            print(f"{threads:<10} {elapsed:>10.2f} {rtf:>8.2f}x {speedup:>9.1f}x")

        # Find optimal
        optimal = min(results.items(), key=lambda x: x[1][0])
        print(f"\n✓ Optimal thread count: {optimal[0]} ({optimal[1][0]:.2f}s)")
        return optimal[0]

    return None


def test_graph_optimization():
    """Test different graph optimization levels."""
    print("\n" + "=" * 70)
    print("TEST 2: Graph Optimization Levels")
    print("=" * 70)
    print("\nTesting different ONNX graph optimization levels...")

    try:
        import onnxruntime as rt

        levels = {
            "Disabled": rt.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "Basic": rt.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "Extended": rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "All": rt.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
    except ImportError:
        print("⚠ onnxruntime not available, using numeric values")
        levels = {
            "Disabled": 0,
            "Basic": 1,
            "Extended": 2,
            "All": 99,
        }

    results = {}
    for name, level in levels.items():
        print(f"\nTesting {name} optimization...", end=" ")
        config = {"graph_optimization_level": level}
        result = benchmark_config(f"opt_{name}", config)
        if result:
            elapsed, rtf, samples = result
            results[name] = (elapsed, rtf)
            print(f"✓ {elapsed:.2f}s (RTF: {rtf:.2f}x)")

    if results:
        print(f"\n{'Level':<15} {'Time (s)':>10} {'RTF':>8} {'Speedup':>10}")
        print("-" * 50)
        baseline = max(results.values(), key=lambda x: x[0])[0]
        for name, (elapsed, rtf) in sorted(results.items(), key=lambda x: x[1][0]):
            speedup = baseline / elapsed
            print(f"{name:<15} {elapsed:>10.2f} {rtf:>8.2f}x {speedup:>9.1f}x")

        # Find optimal
        optimal = min(results.items(), key=lambda x: x[1][0])
        print(f"\n✓ Optimal optimization level: {optimal[0]} ({optimal[1][0]:.2f}s)")
        return optimal[0]

    return None


def test_execution_mode():
    """Test sequential vs parallel execution mode."""
    print("\n" + "=" * 70)
    print("TEST 3: Execution Mode")
    print("=" * 70)
    print("\nTesting sequential vs parallel execution...")

    try:
        import onnxruntime as rt

        modes = {
            "Sequential": rt.ExecutionMode.ORT_SEQUENTIAL,
            "Parallel": rt.ExecutionMode.ORT_PARALLEL,
        }
    except ImportError:
        print("⚠ onnxruntime not available, using numeric values")
        modes = {
            "Sequential": 0,
            "Parallel": 1,
        }

    results = {}
    for name, mode in modes.items():
        print(f"\nTesting {name} mode...", end=" ")
        config = {"execution_mode": mode}
        result = benchmark_config(f"exec_{name}", config)
        if result:
            elapsed, rtf, samples = result
            results[name] = (elapsed, rtf)
            print(f"✓ {elapsed:.2f}s (RTF: {rtf:.2f}x)")

    if results:
        print(f"\n{'Mode':<15} {'Time (s)':>10} {'RTF':>8} {'Speedup':>10}")
        print("-" * 50)
        baseline = max(results.values(), key=lambda x: x[0])[0]
        for name, (elapsed, rtf) in sorted(results.items(), key=lambda x: x[1][0]):
            speedup = baseline / elapsed
            print(f"{name:<15} {elapsed:>10.2f} {rtf:>8.2f}x {speedup:>9.1f}x")

        # Find optimal
        optimal = min(results.items(), key=lambda x: x[1][0])
        print(f"\n✓ Optimal execution mode: {optimal[0]} ({optimal[1][0]:.2f}s)")
        return optimal[0]

    return None


def test_memory_settings():
    """Test different memory arena settings."""
    print("\n" + "=" * 70)
    print("TEST 4: Memory Arena Settings")
    print("=" * 70)
    print("\nTesting CPU memory arena configurations...")

    configs = {
        "Default (all enabled)": {
            "enable_cpu_mem_arena": True,
            "enable_mem_pattern": True,
        },
        "No mem arena": {
            "enable_cpu_mem_arena": False,
            "enable_mem_pattern": True,
        },
        "No mem pattern": {
            "enable_cpu_mem_arena": True,
            "enable_mem_pattern": False,
        },
        "Both disabled": {
            "enable_cpu_mem_arena": False,
            "enable_mem_pattern": False,
        },
    }

    results = {}
    for name, config in configs.items():
        print(f"\nTesting {name}...", end=" ")
        result = benchmark_config(name, config)
        if result:
            elapsed, rtf, samples = result
            results[name] = (elapsed, rtf)
            print(f"✓ {elapsed:.2f}s (RTF: {rtf:.2f}x)")

    if results:
        print(f"\n{'Configuration':<25} {'Time (s)':>10} {'RTF':>8} {'Speedup':>10}")
        print("-" * 60)
        baseline = max(results.values(), key=lambda x: x[0])[0]
        for name, (elapsed, rtf) in sorted(results.items(), key=lambda x: x[1][0]):
            speedup = baseline / elapsed
            print(f"{name:<25} {elapsed:>10.2f} {rtf:>8.2f}x {speedup:>9.1f}x")

        # Find optimal
        optimal = min(results.items(), key=lambda x: x[1][0])
        print(f"\n✓ Optimal memory config: {optimal[0]} ({optimal[1][0]:.2f}s)")


def test_combined_optimal():
    """Test combined optimal configuration."""
    print("\n" + "=" * 70)
    print("TEST 5: Combined Optimal Configuration")
    print("=" * 70)

    cpu_count = os.cpu_count() or 4

    # Build optimal config from previous tests
    try:
        import onnxruntime as rt

        optimal_config = {
            "intra_op_num_threads": cpu_count,
            "inter_op_num_threads": 1,
            "graph_optimization_level": rt.GraphOptimizationLevel.ORT_ENABLE_ALL,
            "execution_mode": rt.ExecutionMode.ORT_SEQUENTIAL,
            "enable_cpu_mem_arena": True,
            "enable_mem_pattern": True,
        }
    except ImportError:
        optimal_config = {
            "intra_op_num_threads": cpu_count,
            "inter_op_num_threads": 1,
            "graph_optimization_level": 99,  # ORT_ENABLE_ALL
            "execution_mode": 0,  # ORT_SEQUENTIAL
            "enable_cpu_mem_arena": True,
            "enable_mem_pattern": True,
        }

    print("\nTesting optimal combined configuration:")
    for key, value in optimal_config.items():
        print(f"  {key}: {value}")

    print("\nRunning benchmark...", end=" ")
    result = benchmark_config("optimal_combined", optimal_config, warmup=True)

    if result:
        elapsed, rtf, samples = result
        print(f"✓ {elapsed:.2f}s (RTF: {rtf:.2f}x)")

        # Save sample output
        pipe = KokoroPipeline(
            PipelineConfig(
                voice="af_bella",
                provider="cpu",
                provider_options=optimal_config,
                generation=GenerationConfig(lang="en-us", speed=1.0),
            )
        )
        res = pipe.run(TEXT)
        samples, sr = res.audio, res.sample_rate
        sf.write("cpu_benchmark_optimal.wav", samples, sr)
        print("  Saved sample to: cpu_benchmark_optimal.wav")

        return optimal_config

    return None


def main():
    """Run all CPU benchmarks."""
    print("=" * 70)
    print("PYKOKORO CPU PROVIDER BENCHMARK")
    print("Parameter Optimization Suite")
    print("=" * 70)

    cpu_count = os.cpu_count() or 4
    print("\nSystem Information:")
    print(f"  CPU cores available: {cpu_count}")
    print(f"  Test text length: {len(TEXT)} characters")

    # Run all tests
    optimal_threads = test_thread_counts()
    optimal_graph_opt = test_graph_optimization()
    optimal_exec_mode = test_execution_mode()
    test_memory_settings()
    optimal_config = test_combined_optimal()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)

    if optimal_config:
        print("\nRecommended CPU configuration:")
        print("```python")
        print("from pykokoro import KokoroPipeline, PipelineConfig")
        print("from pykokoro.generation_config import GenerationConfig")
        print()
        print("pipe = KokoroPipeline(")
        print("    PipelineConfig(")
        print('        voice="af_bella",')
        print('        generation=GenerationConfig(lang="en-us", speed=1.0),')
        print('        provider="cpu",')
        print("        provider_options={")
        for key, value in optimal_config.items():
            if isinstance(value, (bool, int)):
                print(f'            "{key}": {value},')
            else:
                print(f'            "{key}": {value!r},')
        print("        }")
        print("    )")
        print(")")
        print("```")

        print("\nAlternatively, save to config file (~/.config/pykokoro/config.json):")
        print("{")
        print('  "provider": "cpu",')
        print('  "provider_options": {')
        for i, (key, value) in enumerate(optimal_config.items()):
            comma = "," if i < len(optimal_config) - 1 else ""
            if isinstance(value, bool):
                print(f'    "{key}": {str(value).lower()}{comma}')
            elif isinstance(value, int):
                print(f'    "{key}": {value}{comma}')
            else:
                print(f'    "{key}": "{value}"{comma}')
        print("  }")
        print("}")

    print("\n" + "=" * 70)
    print("Key Findings:")
    print("=" * 70)
    if optimal_threads:
        print(f"  ✓ Optimal thread count: {optimal_threads}")
    if optimal_graph_opt:
        print(f"  ✓ Best graph optimization: {optimal_graph_opt}")
    if optimal_exec_mode:
        print(f"  ✓ Best execution mode: {optimal_exec_mode}")

    print("\n" + "=" * 70)
    print("Notes:")
    print("=" * 70)
    print("  • Results may vary based on your CPU architecture")
    print("  • Thread count close to CPU core count is usually optimal")
    print("  • Graph optimization 'All' is recommended for best quality")
    print("  • Sequential execution mode is typically faster for TTS")
    print("  • Memory arena settings have minimal impact on performance")
    print("\nFor GPU/accelerator optimization, run: python examples/gpu_benchmark.py")


if __name__ == "__main__":
    main()
