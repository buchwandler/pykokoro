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


def _short_sentence_config(policy: str, phrase_fallback_tries: int = 1) -> ShortSentenceConfig:
    if phrase_fallback_tries < 0:
        raise ValueError("phrase_fallback_tries must be non-negative")
    if policy == "disabled":
        return ShortSentenceConfig(enabled=False)
    if policy == "wrap":
        return ShortSentenceConfig(resolve_mode="wrap", phrase_fallback_tries=phrase_fallback_tries)
    return ShortSentenceConfig(
        resolve_mode="phrase",
        resolve_modes={"phrase": PhraseResolveMode(cutter=policy)},
        phrase_fallback_tries=phrase_fallback_tries,
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
        "configured_cutter": short_metadata.get("cutter"),
        "phrase_template": short_metadata.get("phrase_template"),
        "phrase_language": short_metadata.get("phrase_language"),
        "retry_attempts": short_metadata.get("retry_attempts", 0),
        "cut_failure_reason": short_metadata.get("cut_failure_reason"),
        "fallback_retries": short_metadata.get("phrase_fallback_tries", 1),
        "max_phrase_attempts": int(short_metadata.get("phrase_fallback_tries", 1)) + 1,
        "phrase_attempt_count": len(short_metadata.get("short_sentence_attempts", []))
        if isinstance(short_metadata.get("short_sentence_attempts"), list)
        else 0,
        "attempt_history": short_metadata.get("short_sentence_attempts", []),
        "success_attempt_ordinal": next(
            (
                attempt.get("ordinal")
                for attempt in short_metadata.get("short_sentence_attempts", [])
                if isinstance(attempt, dict) and attempt.get("succeeded") is True
            ),
            None,
        )
        if isinstance(short_metadata.get("short_sentence_attempts"), list)
        else None,
        "success_template": short_metadata.get("phrase_template")
        if short_metadata.get("cut_strategy")
        in {"energy-valley", "timestamp-smooth", "timestamp-smooth-relaxed", "timestamp-anchor"}
        else None,
        "timing_failure_reason": short_metadata.get("timing_failure_reason"),
        "failure_stage": short_metadata.get("failure_stage"),
        "timing_token_count": short_metadata.get("timing_token_count"),
        "timing_target_token_count": short_metadata.get("timing_target_token_count"),
        "timing_model_position_count": short_metadata.get("timing_model_position_count"),
        "timing_model_position_delta": short_metadata.get("timing_model_position_delta"),
        "timing_first_unresolved_token_index": short_metadata.get(
            "timing_first_unresolved_token_index"
        ),
        "timing_first_unresolved_token_text": short_metadata.get(
            "timing_first_unresolved_token_text"
        ),
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

    parser.add_argument("--phrase-fallback-tries", type=int, default=1)
    parser.add_argument(
        "--retry-budgets",
        help="Comma-separated fallback retry budgets; overrides --phrase-fallback-tries.",
    )


def _retry_budgets(args: argparse.Namespace) -> tuple[int, ...]:
    if args.retry_budgets:
        raw_values = args.retry_budgets.split(",")
        try:
            budgets = tuple(int(value.strip()) for value in raw_values if value.strip())
        except ValueError as exc:
            raise ValueError("--retry-budgets must contain comma-separated integers") from exc
    else:
        budgets = (args.phrase_fallback_tries,)
    if not budgets or any(value < 0 for value in budgets):
        raise ValueError("retry budgets must contain at least one non-negative integer")
    return budgets


def _dry_run_rows(voices: tuple[str, ...], budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in budgets:
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
                            "fallback_retries": budget,
                            "max_phrase_attempts": budget + 1,
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
                            "fallback_retries": budget,
                            "max_phrase_attempts": budget + 1,
                        }
                    )
    return rows


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    voices = tuple(voice.strip() for voice in args.voices.split(",") if voice.strip())
    budgets = _retry_budgets(args)
    if args.dry_run:
        return _dry_run_rows(voices, budgets)
    languages = tuple(sorted({language for _, _, language in SCENARIOS}))
    pipelines: dict[tuple[str, str, int], KokoroPipeline] = {}
    try:
        for budget in budgets:
            for policy in POLICIES:
                for language in languages:
                    pipelines[(policy, language, budget)] = KokoroPipeline(
                        PipelineConfig(
                            voice=voices[0],
                            model_source=args.model_source,
                            model_variant=args.model_variant,
                            model_path=args.model_path,
                            voices_path=args.voices_path,
                            generation=GenerationConfig(lang=language, speed=1.0),
                            short_sentence_config=_short_sentence_config(policy, budget),
                            return_trace=True,
                        )
                    )
        for pipeline in pipelines.values():
            pipeline.warmup()
        rows: list[dict[str, Any]] = []
        for budget in budgets:
            for voice in voices:
                for policy in POLICIES:
                    for language in languages:
                        pipeline = pipelines[(policy, language, budget)]
                        for scenario, text, scenario_language in SCENARIOS:
                            if scenario_language != language:
                                continue
                            started = time.perf_counter()
                            result = pipeline.run(text, voice=voice)
                            wall_ms = (time.perf_counter() - started) * 1000.0
                            row = _summary(
                                result, policy=policy, scenario=scenario, text=text, wall_ms=wall_ms
                            )
                            row.update(
                                {
                                    "voice": voice,
                                    "language": language,
                                    "model_source": args.model_source,
                                    "model_variant": args.model_variant,
                                    "fallback_retries": budget,
                                    "max_phrase_attempts": budget + 1,
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
                                        "fallback_retries": budget,
                                        "max_phrase_attempts": budget + 1,
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
    budgets = sorted({int(row["fallback_retries"]) for row in rows if "fallback_retries" in row})
    for retries in budgets:
        print(f"Fallback retries: {retries}")
        print(f"Maximum phrase attempts per short segment: {retries + 1}")


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
            "fallback_retries": args.phrase_fallback_tries,
            "retry_budgets": list(_retry_budgets(args)),
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
