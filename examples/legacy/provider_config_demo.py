"""
Demonstration of ONNX Runtime provider configuration options.

This example shows different ways to configure execution providers
using SessionOptions and provider_options.
"""

import soundfile as sf

import pykokoro

# Example 1: Simple - Use defaults
print("=" * 60)
print("Example 1: Default Configuration (CPU)")
print("=" * 60)

kokoro_cpu = pykokoro.Kokoro(provider="cpu")
audio, sr = kokoro_cpu.create(
    "This uses the default CPU provider with automatic settings.",
    voice="af_bella",
    lang="en-us",
)
sf.write("provider_demo_cpu.wav", audio, sr)
print(f"✓ Generated audio with CPU provider: {len(audio)} samples at {sr} Hz\n")
kokoro_cpu.close()

# Example 2: Provider-specific options via dict
print("=" * 60)
print("Example 2: CPU with Custom Thread Count")
print("=" * 60)

kokoro_threads = pykokoro.Kokoro(
    provider="cpu",
    provider_options={
        "intra_op_num_threads": 4,  # Use 4 threads for operations
        "inter_op_num_threads": 1,  # Sequential execution
    },
)
audio, sr = kokoro_threads.create(
    "This uses CPU with custom thread configuration.",
    voice="af_bella",
    lang="en-us",
)
sf.write("provider_demo_cpu_threads.wav", audio, sr)
print(f"✓ Generated audio with 4 threads: {len(audio)} samples at {sr} Hz\n")
kokoro_threads.close()

# Example 3: OpenVINO with custom settings (if available)
print("=" * 60)
print("Example 3: OpenVINO with Custom Precision")
print("=" * 60)

try:
    import onnxruntime as rt

    if "OpenVINOExecutionProvider" in rt.get_available_providers():
        kokoro_openvino = pykokoro.Kokoro(
            provider="openvino",
            model_quality="fp16",  # Use FP16 model
            provider_options={
                "precision": "FP16",  # Auto-set from model_quality, but explicit here
                "num_of_threads": 8,
                "device_type": "CPU_FP32",
            },
        )
        audio, sr = kokoro_openvino.create(
            "This uses OpenVINO with FP16 precision.",
            voice="af_bella",
            lang="en-us",
        )
        sf.write("provider_demo_openvino.wav", audio, sr)
        print(f"✓ Generated audio with OpenVINO FP16: {len(audio)} samples at {sr} Hz\n")
        kokoro_openvino.close()
    else:
        print("⚠ OpenVINO provider not available")
        print("  Install with: pip install pykokoro[openvino]\n")
except ImportError:
    print("⚠ onnxruntime not available\n")

# Example 4: CUDA with memory limit (if available)
print("=" * 60)
print("Example 4: CUDA with GPU Memory Limit")
print("=" * 60)

try:
    import onnxruntime as rt

    if "CUDAExecutionProvider" in rt.get_available_providers():
        kokoro_cuda = pykokoro.Kokoro(
            provider="cuda",
            provider_options={
                "device_id": 0,  # Use GPU 0
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
                "arena_extend_strategy": "kNextPowerOfTwo",
            },
        )
        audio, sr = kokoro_cuda.create(
            "This uses CUDA with a 2 gigabyte memory limit.",
            voice="af_bella",
            lang="en-us",
        )
        sf.write("provider_demo_cuda.wav", audio, sr)
        print(f"✓ Generated audio with CUDA (2GB limit): {len(audio)} samples at {sr} Hz\n")
        kokoro_cuda.close()
    else:
        print("⚠ CUDA provider not available")
        print("  Install with: pip install pykokoro[gpu]\n")
except ImportError:
    print("⚠ onnxruntime not available\n")

# Example 5: Advanced - Full SessionOptions control
print("=" * 60)
print("Example 5: Advanced SessionOptions")
print("=" * 60)

try:
    import onnxruntime as rt

    # Create custom SessionOptions
    sess_opt = rt.SessionOptions()
    sess_opt.intra_op_num_threads = 8
    sess_opt.inter_op_num_threads = 1
    sess_opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_opt.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess_opt.enable_profiling = False  # Set to True for detailed profiling

    kokoro_advanced = pykokoro.Kokoro(
        provider="cpu",
        session_options=sess_opt,  # Pass pre-configured SessionOptions
    )
    audio, sr = kokoro_advanced.create(
        "This uses custom session options with extended graph optimization.",
        voice="af_bella",
        lang="en-us",
    )
    sf.write("provider_demo_advanced.wav", audio, sr)
    print(f"✓ Generated audio with advanced options: {len(audio)} samples at {sr} Hz\n")
    kokoro_advanced.close()
except ImportError:
    print("⚠ onnxruntime not available\n")

# Example 6: Auto-select with options
print("=" * 60)
print("Example 6: Auto-select Provider with Options")
print("=" * 60)

kokoro_auto = pykokoro.Kokoro(
    provider="auto",  # Automatically select best available
    provider_options={
        "intra_op_num_threads": 6,  # Applied to whichever provider is selected
        "precision": "FP16",  # Will be used if OpenVINO is selected
    },
)
audio, sr = kokoro_auto.create(
    "This automatically selects the best provider with custom options.",
    voice="af_bella",
    lang="en-us",
)
sf.write("provider_demo_auto.wav", audio, sr)
print(f"✓ Generated audio with auto-select: {len(audio)} samples at {sr} Hz\n")
kokoro_auto.close()

print("=" * 60)
print("All examples completed!")
print("=" * 60)
print("\nGenerated files:")
print("  - provider_demo_cpu.wav")
print("  - provider_demo_cpu_threads.wav")
print("  - provider_demo_openvino.wav (if OpenVINO available)")
print("  - provider_demo_cuda.wav (if CUDA available)")
print("  - provider_demo_advanced.wav")
print("  - provider_demo_auto.wav")
print("\nConfiguration options demonstrated:")
print("  1. Simple provider selection")
print("  2. Custom thread counts")
print("  3. Provider-specific options (OpenVINO precision)")
print("  4. GPU memory limits (CUDA)")
print("  5. Advanced SessionOptions control")
print("  6. Auto-selection with options")
