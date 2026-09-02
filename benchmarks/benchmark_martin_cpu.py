#!/usr/bin/env python3
"""Measure Martin German CPU model and pipeline runtime separately."""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from pykokoro import KokoroPipeline


def _german_assets() -> tuple[str, Any]:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples.german import TEXT, make_config

    return TEXT, make_config


def _execution_mode(value: str) -> Any:
    import onnxruntime as rt

    return {
        "sequential": rt.ExecutionMode.ORT_SEQUENTIAL,
        "parallel": rt.ExecutionMode.ORT_PARALLEL,
    }[value]


def _graph_optimization_level(value: str) -> Any:
    import onnxruntime as rt

    return {
        "disabled": rt.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": rt.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": rt.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[value]


def _provider_options(
    *,
    threads: int | None,
    execution_mode: str | None,
    memory_enabled: bool | None,
    graph_optimization: str | None = "all",
    baseline: bool = False,
) -> dict[str, Any]:
    if baseline:
        return {}
    options: dict[str, Any] = {"inter_op_num_threads": 1}
    if execution_mode is not None:
        options["execution_mode"] = _execution_mode(execution_mode)
    if graph_optimization is not None:
        options["graph_optimization_level"] = _graph_optimization_level(graph_optimization)
    if threads is not None:
        options["intra_op_num_threads"] = threads
    if memory_enabled is not None:
        options["enable_cpu_mem_arena"] = memory_enabled
        options["enable_mem_pattern"] = memory_enabled
    return options


def _clear_inference_cache(pipeline: KokoroPipeline) -> None:
    backend = getattr(pipeline, "_kokoro", None)
    generator = getattr(backend, "_audio_generator", None)
    clear = getattr(generator, "clear_inference_cache", None)
    if callable(clear):
        clear()


def _measure(pipeline: KokoroPipeline, text: str) -> dict[str, float | int]:
    started = time.perf_counter()
    result = pipeline.run(text)
    wall_seconds = time.perf_counter() - started
    audio_seconds = len(result.audio) / result.sample_rate if result.sample_rate else 0.0
    summary = result.trace.inference_summary() if result.trace is not None else {}
    runtime_ms = float(summary.get("onnx_runtime_ms", 0.0))
    retry_calls = int(summary.get("short_sentence_retry_calls", 0))
    fallback_calls = int(summary.get("fallback_onnx_calls", 0))
    return {
        "wall_seconds": wall_seconds,
        "audio_seconds": audio_seconds,
        "model_rtf": runtime_ms / 1000.0 / audio_seconds if audio_seconds else 0.0,
        "pipeline_rtf": wall_seconds / audio_seconds if audio_seconds else 0.0,
        "segments": len(result.phoneme_segments),
        "onnx_calls": int(summary.get("onnx_calls", 0)) - int(summary.get("onnx_cache_hits", 0)),
        "cache_hits": int(summary.get("onnx_cache_hits", 0)),
        "cache_misses": int(summary.get("onnx_cache_misses", 0)),
        "onnx_runtime_ms": runtime_ms,
        "retry_calls": retry_calls,
        "fallback_calls": fallback_calls,
    }


def _print_measurement(label: str, measurement: dict[str, float | int]) -> None:
    print(
        f"{label}: wall={measurement['wall_seconds']:.3f}s "
        f"audio={measurement['audio_seconds']:.3f}s "
        f"model_rtf={measurement['model_rtf']:.3f} "
        f"pipeline_rtf={measurement['pipeline_rtf']:.3f} "
        f"segments={measurement['segments']} "
        f"onnx_calls={measurement['onnx_calls']} "
        f"cache_hits={measurement['cache_hits']} "
        f"cache_misses={measurement['cache_misses']} "
        f"onnx_time={measurement['onnx_runtime_ms'] / 1000.0:.3f}s "
        f"retries={measurement['retry_calls']} "
        f"fallbacks={measurement['fallback_calls']}"
    )


def _run_case(
    label: str,
    *,
    lexicon: str,
    repeat: int,
    warmup: bool,
    cache_enabled: bool,
    single_segment: bool = False,
    provider_options: dict[str, Any],
) -> None:
    text, make_config = _german_assets()
    print(f"\n{label}: provider/session options: {provider_options}")
    config = replace(
        make_config(lexicons=(lexicon,)),
        provider="cpu",
        provider_options=provider_options,
        inference_cache_enabled=cache_enabled,
    )
    if single_segment:
        from examples.pipeline_g2p_onnx_minimal import PlainDocumentParser

        pipeline = KokoroPipeline(config, doc_parser=PlainDocumentParser())
    else:
        pipeline = KokoroPipeline(config)
    with pipeline:
        if warmup:
            pipeline.run("Warmup.")
        if not cache_enabled:
            _clear_inference_cache(pipeline)
        measurements = [_measure(pipeline, text) for _ in range(repeat)]

    for index, measurement in enumerate(measurements, start=1):
        _print_measurement(f"{label} run {index}", measurement)
    medians = {
        key: statistics.median(measurement[key] for measurement in measurements)
        for key in measurements[0]
    }
    _print_measurement(f"{label} median", medians)


def _matrix_cases(cpu_count: int) -> list[tuple[str, dict[str, Any]]]:
    thread_counts = [value for value in (1, 2, 4, 8, cpu_count // 2, cpu_count) if value > 0]
    cases: list[tuple[str, dict[str, Any]]] = [
        (
            "baseline",
            _provider_options(
                threads=None,
                execution_mode=None,
                memory_enabled=None,
                graph_optimization=None,
                baseline=True,
            ),
        )
    ]
    cases.extend(
        (
            f"threads-{threads}",
            _provider_options(
                threads=threads,
                execution_mode="sequential",
                memory_enabled=None,
            ),
        )
        for threads in sorted(set(thread_counts))
    )
    cases.extend(
        (
            f"mode-{mode}",
            _provider_options(
                threads=cpu_count,
                execution_mode=mode,
                memory_enabled=None,
            ),
        )
        for mode in ("sequential", "parallel")
    )
    cases.extend(
        (
            f"memory-{enabled}",
            _provider_options(
                threads=cpu_count,
                execution_mode="sequential",
                memory_enabled=enabled,
            ),
        )
        for enabled in (True, False)
    )
    cases.extend(
        (
            f"graph-{level}",
            _provider_options(
                threads=cpu_count,
                execution_mode="sequential",
                memory_enabled=None,
                graph_optimization=level,
            ),
        )
        for level in ("disabled", "basic", "extended", "all")
    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon", default="gold", choices=("gold", "crane", "espeak", "olaph"))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--cache", action="store_true", help="Keep the acoustic cache enabled")
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--execution-mode", choices=("sequential", "parallel"), default="sequential"
    )
    parser.add_argument("--matrix", action="store_true", help="Run the Martin CPU option matrix")
    parser.add_argument(
        "--control", action="store_true", help="Compare normal and single-segment document paths"
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    import onnxruntime as rt

    cpu_count = os.cpu_count() or 1
    print(f"CPU count: {cpu_count}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")
    print(f"ONNX Runtime: {rt.__version__}")
    print(f"Available providers: {', '.join(rt.get_available_providers())}")
    print("Selected provider: CPUExecutionProvider")
    print("Model variant: v1.2-de-martin")
    print("Model quality: fp32")
    print(f"Lexicon: {args.lexicon}")
    print(f"Cache enabled: {args.cache}")
    print(f"Measured repetitions: {args.repeat}")

    if args.matrix:
        for label, options in _matrix_cases(cpu_count):
            _run_case(
                label,
                lexicon=args.lexicon,
                repeat=args.repeat,
                warmup=not args.no_warmup,
                cache_enabled=args.cache,
                provider_options=options,
            )
        return

    options = _provider_options(
        threads=args.threads,
        execution_mode=args.execution_mode,
        memory_enabled=None,
    )
    print(f"Provider/session options: {options}")
    if args.control:
        _run_case(
            "normal",
            lexicon=args.lexicon,
            repeat=args.repeat,
            warmup=not args.no_warmup,
            cache_enabled=args.cache,
            provider_options=options,
        )
        _run_case(
            "single-segment",
            lexicon=args.lexicon,
            repeat=args.repeat,
            warmup=not args.no_warmup,
            cache_enabled=args.cache,
            provider_options=options,
            single_segment=True,
        )
        return

    _run_case(
        "Martin",
        lexicon=args.lexicon,
        repeat=args.repeat,
        warmup=not args.no_warmup,
        cache_enabled=args.cache,
        provider_options=options,
    )


if __name__ == "__main__":
    main()
