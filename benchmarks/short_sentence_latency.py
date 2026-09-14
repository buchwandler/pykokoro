"""Benchmark warm short-sentence rendering and structural inference counters.

This is the engineering benchmark. It reports final case outcomes separately from
attempt-level costs and does not create listening artifacts. Use ``--dry-run`` to
inspect the complete matrix without loading model assets.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from pykokoro import GenerationConfig, PipelineConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.short_sentence_handler import PhraseResolveMode, ShortSentenceConfig

try:
    from benchmarks._short_sentence_reporting import (
        build_short_sentence_summary,
        format_short_sentence_table,
        normalize_short_sentence_row,
    )
except ModuleNotFoundError:
    from _short_sentence_reporting import (
        build_short_sentence_summary,
        format_short_sentence_table,
        normalize_short_sentence_row,
    )

logger = logging.getLogger(__name__)

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
    ("short-declarative", "Oh No.", "en-us"),
    ("short-question", "Why?", "en-us"),
    ("short-exclamation", "Help!", "en-us"),
    ("one-word-interjection", "Oh!", "en-us"),
    ("two-word-phrase", "No thanks.", "en-us"),
    ("fragment", "thing on her chest.", "en-us"),
    ("ellipsis", "Wait …", "en-us"),
    ("near-min-phoneme-length", "One small step.", "en-us"),
    ("zero-gap-fragment", "three", "en-us"),
    ("terminal-fragment", "tomato and dictionary.", "en-us"),
    ("german-declarative", "zwölf.", "de"),
    ("german-question", "wirklich?", "de"),
    ("german-exclamation", "Hilfe!", "de"),
)
POLICIES = ("disabled", "energy-valley", "timestamp-adaptive", "wrap")


def _short_sentence_config(policy: str) -> ShortSentenceConfig:
    if policy == "disabled":
        return ShortSentenceConfig(enabled=False)
    if policy == "wrap":
        return ShortSentenceConfig(resolve_mode="wrap")
    return ShortSentenceConfig(
        resolve_mode="phrase",
        resolve_modes={"phrase": PhraseResolveMode(cutter=policy)},
    )


def _short_sentence_metadata(trace: Any) -> dict[str, object]:
    if trace is None:
        return {}
    for event in trace.prosody:
        metadata = event.get("short_sentence") if isinstance(event, dict) else None
        if isinstance(metadata, dict):
            return dict(metadata)
    return {}


def _summary(
    result: Any,
    *,
    policy: str,
    scenario: str,
    text: str,
    wall_ms: float,
) -> dict[str, Any]:
    trace = result.trace
    inference_summary = trace.inference_summary() if trace is not None else {}
    counters = trace.counters if trace is not None else {}
    audio_seconds = result.audio.size / result.sample_rate if result.sample_rate else 0.0
    short_metadata = _short_sentence_metadata(trace)
    boundary_metrics = _waveform_boundary_metrics(result.audio)
    boundary_metrics.update(
        {
            key: value
            for key, value in short_metadata.items()
            if key.endswith("_distance_samples")
            or key.endswith("_distance_ms")
            or key == "retained_guard_duration_ms"
        }
    )
    row = {
        "policy": policy,
        "configured_policy": policy,
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
        "wrap_fallback_renders": counters.get("short_sentence_wrap_fallback", 0),
        "strict_cut_successes": counters.get("short_sentence_cut_strict_success", 0),
        "adaptive_cut_successes": counters.get("short_sentence_cut_adaptive_success", 0),
        "cut_failures": counters.get("short_sentence_cut_failure", 0),
        "boundary_metrics": boundary_metrics,
        "short_sentence_metadata": short_metadata,
        "cut_strategy": short_metadata.get("cut_strategy"),
        "actual_cut_strategy": short_metadata.get("cut_strategy"),
        "fallback_used": short_metadata.get("fallback_used"),
        "retry_attempts": short_metadata.get("retry_attempts", 0),
        "cut_failure_reason": short_metadata.get("cut_failure_reason"),
        "timing_failure_reason": short_metadata.get("timing_failure_reason"),
        "failure_stage": short_metadata.get("failure_stage"),
        "timing_token_count": short_metadata.get("timing_token_count"),
        "timing_target_token_count": short_metadata.get("timing_target_token_count"),
        "timing_model_position_count": short_metadata.get("timing_model_position_count"),
        "generated_token_count": short_metadata.get("generated_token_count"),
        "pred_duration_count": short_metadata.get("pred_duration_count"),
        "timing_alignment_complete": short_metadata.get("timing_alignment_complete"),
        "timestamp_join_complete": short_metadata.get("timestamp_join_complete"),
        "target_timestamp_count": short_metadata.get("target_timestamp_count"),
        "left_gap_samples": _metadata_gap(
            short_metadata, "previous_token_end_ts", "target_start_ts"
        ),
        "right_gap_samples": _metadata_gap(short_metadata, "target_end_ts", "next_token_start_ts"),
    }
    return normalize_short_sentence_row(row)


def _metadata_gap(
    metadata: dict[str, object],
    start_key: str,
    end_key: str,
) -> int | None:
    start = metadata.get(start_key)
    end = metadata.get(end_key)
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    return round((float(end) - float(start)) * 24000)


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
    parser.add_argument("--output-format", choices=("table", "jsonl", "both"), default="table")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("--dry-run", action="store_true")


def _dry_run_rows(voices: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for voice in voices:
        for policy in POLICIES:
            for scenario, text, language in SCENARIOS:
                rows.append(
                    {
                        "policy": policy,
                        "configured_policy": policy,
                        "scenario": scenario,
                        "text": text,
                        "language": language,
                        "voice": voice,
                    }
                )
            for text in DEMO_TEXTS:
                rows.append(
                    {
                        "policy": policy,
                        "configured_policy": policy,
                        "scenario": "demo-corpus",
                        "text": text,
                        "language": "en-us",
                        "voice": voice,
                    }
                )
    return rows


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    voices = tuple(voice.strip() for voice in args.voices.split(",") if voice.strip())
    languages = tuple(sorted({language for _, _, language in SCENARIOS}))
    if args.dry_run:
        return _dry_run_rows(voices)

    pipelines: dict[tuple[str, str], KokoroPipeline] = {}
    try:
        for policy in POLICIES:
            for language in languages:
                pipelines[(policy, language)] = KokoroPipeline(
                    PipelineConfig(
                        voice=voices[0],
                        model_source=args.model_source,
                        model_variant=args.model_variant,
                        model_path=args.model_path,
                        voices_path=args.voices_path,
                        generation=GenerationConfig(lang=language, speed=1.0),
                        short_sentence_config=_short_sentence_config(policy),
                        return_trace=True,
                    )
                )
        for pipeline in pipelines.values():
            pipeline.warmup()

        rows: list[dict[str, Any]] = []
        for voice in voices:
            for policy in POLICIES:
                for language in languages:
                    pipeline = pipelines[(policy, language)]
                    for scenario, text, scenario_language in SCENARIOS:
                        if scenario_language != language:
                            continue
                        started = time.perf_counter()
                        result = pipeline.run(text, voice=voice)
                        wall_ms = (time.perf_counter() - started) * 1000.0
                        row = _summary(
                            result,
                            policy=policy,
                            scenario=scenario,
                            text=text,
                            wall_ms=wall_ms,
                        )
                        row.update(
                            {
                                "voice": voice,
                                "language": language,
                                "model_source": args.model_source,
                                "model_variant": args.model_variant,
                            }
                        )
                        rows.append(row)
                    if language == "en-us":
                        for text in DEMO_TEXTS:
                            started = time.perf_counter()
                            result = pipeline.run(text, voice=voice)
                            wall_ms = (time.perf_counter() - started) * 1000.0
                            row = _summary(
                                result,
                                policy=policy,
                                scenario="demo-corpus",
                                text=text,
                                wall_ms=wall_ms,
                            )
                            row.update(
                                {
                                    "voice": voice,
                                    "language": language,
                                    "model_source": args.model_source,
                                    "model_variant": args.model_variant,
                                }
                            )
                            rows.append(row)
        return rows
    finally:
        for pipeline in pipelines.values():
            pipeline.close()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _print_dry_run(rows: list[dict[str, Any]]) -> None:
    voices = sorted({str(row["voice"]) for row in rows})
    languages = sorted({str(row["language"]) for row in rows})
    policies = sorted({str(row["configured_policy"]) for row in rows})
    scenarios = len({(row["scenario"], row["text"], row["language"]) for row in rows})
    print("Short-sentence benchmark dry run")
    print(f"Voices: {', '.join(voices)}")
    print(f"Languages: {', '.join(languages)}")
    print(f"Policies: {', '.join(policies)}")
    print(f"Scenario count: {scenarios}")
    print(f"Expected total renders: {len(rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    rows = run_benchmark(args)
    if args.dry_run:
        _print_dry_run(rows)
        return 0

    summary = build_short_sentence_summary(
        rows,
        metadata={
            "model_source": args.model_source,
            "model_variant": args.model_variant,
            "voices": [row.get("voice") for row in rows[:1]],
        },
    )
    if args.output_format in {"jsonl", "both"}:
        if args.jsonl is not None:
            _write_jsonl(args.jsonl, rows)
        else:
            for row in rows:
                print(json.dumps(row, sort_keys=True))
    elif args.jsonl is not None:
        _write_jsonl(args.jsonl, rows)
    if args.output_format in {"table", "both"}:
        print(format_short_sentence_table(summary))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
