"""Benchmark warm short-sentence rendering and structural inference counters.

This benchmark intentionally reports wall time without using it as a CI threshold.
Use ``--dry-run`` to inspect the complete scenario and policy matrix without loading
model assets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from pykokoro import GenerationConfig, PipelineConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.short_sentence_handler import (
    PhraseResolveMode,
    ShortSentenceConfig,
)

DEMO_TEXTS = (
    "Hi!",
    "Why?",
    "Oh No.",
    "No!",
    "Yes!",
    "Help!",
    "Oh!",
    "Stop!",
    "What?",
    "Don't!",
    "One … step. In front.",
    "thing on her chest.",
    "Of.",
)

SCENARIOS = (
    ("short-declarative", "Oh No."),
    ("short-question", "Why?"),
    ("short-exclamation", "Help!"),
    ("one-word-interjection", "Oh!"),
    ("two-word-phrase", "No thanks."),
    ("fragment", "thing on her chest."),
    ("ellipsis", "Wait …"),
    ("near-min-phoneme-length", "One small step."),
)
POLICIES = ("disabled", "energy-valley", "timestamp-adaptive", "wrap")


def _short_sentence_config(policy: str) -> ShortSentenceConfig:
    if policy == "disabled":
        return ShortSentenceConfig(enabled=False)
    if policy == "wrap":
        return ShortSentenceConfig(resolve_mode="wrap")
    return ShortSentenceConfig(
        resolve_mode="phrase",
        resolve_modes={
            "phrase": PhraseResolveMode(cutter=policy),
        },
    )


def _summary(result: Any, *, policy: str, scenario: str, text: str, wall_ms: float) -> dict[str, Any]:
    trace = result.trace
    inference_summary = trace.inference_summary() if trace is not None else {}
    counters = trace.counters if trace is not None else {}
    audio_seconds = result.audio.size / result.sample_rate if result.sample_rate else 0.0
    boundary_metrics = _waveform_boundary_metrics(result.audio)
    if trace is not None:
        for event in trace.prosody:
            metadata = event.get("short_sentence") if isinstance(event, dict) else None
            if isinstance(metadata, dict):
                boundary_metrics.update(
                    {
                        key: value
                        for key, value in metadata.items()
                        if key.endswith("_distance_samples")
                        or key.endswith("_distance_ms")
                        or key == "retained_guard_duration_ms"
                    }
                )
    return {
        "policy": policy,
        "scenario": scenario,
        "text": text,
        "wall_ms": wall_ms,
        "audio_seconds": audio_seconds,
        "rtf": wall_ms / 1000.0 / audio_seconds if audio_seconds else None,
        "short_segments": counters.get("short_sentence_detected", 0),
        "onnx_calls": inference_summary.get("onnx_calls", 0),
        "onnx_runtime_ms": inference_summary.get("onnx_runtime_ms", 0.0),
        "initial_phrase_attempts": counters.get("short_sentence_phrase_initial", 0),
        "phrase_retries": counters.get("short_sentence_phrase_retry", 0),
        "wrap_fallbacks": counters.get("short_sentence_wrap_fallback", 0),
        "strict_cut_successes": counters.get("short_sentence_cut_strict_success", 0),
        "adaptive_cut_successes": counters.get("short_sentence_cut_adaptive_success", 0),
        "cut_failures": counters.get("short_sentence_cut_failure", 0),
        "boundary_metrics": boundary_metrics,
    }

def _waveform_boundary_metrics(audio: Any) -> dict[str, object]:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size < 2:
        return {}
    window = min(120, values.size)
    return {
        "left_endpoint_abs_amplitude": float(abs(values[0])),
        "right_endpoint_abs_amplitude": float(abs(values[-1])),
        "left_cross_boundary_slope": float(abs(values[1] - values[0])),
        "right_cross_boundary_slope": float(abs(values[-1] - values[-2])),
        "left_local_rms": float(np.sqrt(np.mean(np.square(values[:window])))),
        "right_local_rms": float(np.sqrt(np.mean(np.square(values[-window:])))),
    }


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-source", choices=("github", "huggingface"), default="huggingface")
    parser.add_argument("--model-variant", default="v1.0")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--voices-path", type=Path)
    parser.add_argument("--voices", default="af_sarah")
    parser.add_argument("--dry-run", action="store_true")


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    voices = tuple(voice.strip() for voice in args.voices.split(",") if voice.strip())
    policies = POLICIES
    if args.dry_run:
        return [
            {"policy": policy, "scenario": scenario, "text": text, "voice": voice}
            for voice in voices
            for policy in policies
            for scenario, text in (*SCENARIOS, ("demo-corpus", "|".join(DEMO_TEXTS)))
        ]

    pipelines: dict[str, KokoroPipeline] = {}
    try:
        for policy in policies:
            pipelines[policy] = KokoroPipeline(
                PipelineConfig(
                    voice=voices[0],
                    model_source=args.model_source,
                    model_variant=args.model_variant,
                    model_path=args.model_path,
                    voices_path=args.voices_path,
                    generation=GenerationConfig(lang="en-us", speed=1.0),
                    short_sentence_config=_short_sentence_config(policy),
                    return_trace=True,
                )
            )
        for pipeline in pipelines.values():
            pipeline.warmup()

        rows: list[dict[str, Any]] = []
        for voice in voices:
            for policy, pipeline in pipelines.items():
                for scenario, text in SCENARIOS:
                    started = time.perf_counter()
                    result = pipeline.run(text, voice=voice)
                    wall_ms = (time.perf_counter() - started) * 1000.0
                    row = _summary(result, policy=policy, scenario=scenario, text=text, wall_ms=wall_ms)
                    row["voice"] = voice
                    rows.append(row)
                for text in DEMO_TEXTS:
                    started = time.perf_counter()
                    result = pipeline.run(text, voice=voice)
                    wall_ms = (time.perf_counter() - started) * 1000.0
                    row = _summary(result, policy=policy, scenario="demo-corpus", text=text, wall_ms=wall_ms)
                    row["voice"] = voice
                    rows.append(row)
        return rows
    finally:
        for pipeline in pipelines.values():
            pipeline.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    args = parser.parse_args()
    for row in run_benchmark(args):
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
