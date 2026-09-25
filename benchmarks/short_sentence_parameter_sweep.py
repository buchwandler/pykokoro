"""Generate one ordered listening WAV for a short-sentence parameter sweep."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from pykokoro.pipeline import KokoroPipeline

from pykokoro import GenerationConfig, PipelineConfig
from pykokoro.short_sentence_handler import PhraseResolveMode, ShortSentenceConfig

try:
    from benchmarks._short_sentence_reporting import classify_short_sentence_outcome
except ModuleNotFoundError:
    from _short_sentence_reporting import classify_short_sentence_outcome


SAMPLE_RATE = 24000
SHORT_PAUSE_SECONDS = 0.25
CANDIDATE_PAUSE_SECONDS = 0.70
SECTION_PAUSE_SECONDS = 1.20
DEFAULT_PHRASE_TEMPLATE = "The question was asked plainly: {segment}"


@dataclass(frozen=True)
class ParameterSpec:
    """Definition of one supported sweep parameter."""

    field: str | None
    parser: Callable[[str], int | float | str]
    cutters: frozenset[str]


PARAMETERS: dict[str, ParameterSpec] = {
    "energy-threshold": ParameterSpec(
        "energy_threshold", float, frozenset({"energy-valley", "timestamp-adaptive"})
    ),
    "frame-duration-ms": ParameterSpec(
        "frame_duration_ms", int, frozenset({"energy-valley", "timestamp-adaptive"})
    ),
    "min-silence-seconds": ParameterSpec(
        "min_silence_seconds", float, frozenset({"energy-valley", "timestamp-adaptive"})
    ),
    "search-radius-ms": ParameterSpec("search_radius_ms", float, frozenset({"timestamp-adaptive"})),
    "context-guard-ms": ParameterSpec("context_guard_ms", float, frozenset({"timestamp-adaptive"})),
    "analysis-window-ms": ParameterSpec(
        "analysis_window_ms", float, frozenset({"timestamp-adaptive"})
    ),
    "cutter": ParameterSpec(None, str, frozenset({"energy-valley", "timestamp-adaptive"})),
}


@dataclass(frozen=True)
class RenderedAudio:
    audio: np.ndarray
    sample_rate: int
    metadata: dict[str, object]
    trace: Any = None


def _default_phrase_template(text: str) -> str:
    if text.rstrip().endswith("?"):
        return DEFAULT_PHRASE_TEMPLATE
    if text.rstrip().endswith("!"):
        return "The speaker called out: {segment}"
    return "The conversation stopped after one last reply: {segment}"


def _parse_value(parameter: str, raw: str) -> int | float | str:
    if parameter == "silence-threshold":
        raise ValueError(
            "silence-threshold is present in compatibility metadata but is not used by the "
            "current short-sentence cutters and therefore cannot be optimized."
        )
    spec = PARAMETERS.get(parameter)
    if spec is None:
        choices = ", ".join(sorted([*PARAMETERS, "silence-threshold"]))
        raise ValueError(f"unsupported parameter {parameter!r}; choose from {choices}")
    try:
        return spec.parser(raw.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid value {raw!r} for parameter {parameter!r}") from exc


def parse_parameter_values(parameter: str, values: str) -> list[int | float | str]:
    """Parse ordered CLI values using the explicit parameter registry."""
    parsed = [_parse_value(parameter, raw) for raw in values.split(",") if raw.strip()]
    if not parsed:
        raise ValueError("at least one sweep value is required")
    return parsed


def validate_parameter(parameter: str, cutter: str, values: list[int | float | str]) -> None:
    """Validate parameter and cutter compatibility before model loading."""
    if parameter == "silence-threshold":
        _parse_value(parameter, "0")
    spec = PARAMETERS.get(parameter)
    if spec is None:
        _parse_value(parameter, "0")
        return
    if cutter not in {"energy-valley", "timestamp-adaptive"}:
        raise ValueError(f"unsupported cutter {cutter!r}")
    if cutter not in spec.cutters:
        raise ValueError(f"parameter {parameter!r} is incompatible with cutter {cutter!r}")
    if parameter == "cutter":
        invalid = [value for value in values if value not in spec.cutters]
        if invalid:
            raise ValueError(
                f"invalid cutter value(s) {invalid!r}; choose from {sorted(spec.cutters)}"
            )


def format_parameter_value(value: int | float | str) -> str:
    """Format a candidate deterministically for spoken labels and filenames."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, ".12g")
    return str(value)


def display_parameter_name(parameter: str) -> str:
    return parameter.replace("-", " ").capitalize()


def build_short_sentence_config(
    cutter: str,
    phrase_template: str,
    parameter: str,
    value: int | float | str,
    *,
    phrase_fallback_tries: int = 0,
) -> ShortSentenceConfig:
    """Build a candidate config with the same explicit phrase for every value."""
    effective_cutter = str(value) if parameter == "cutter" else cutter
    mode = PhraseResolveMode(
        phrase_selection="neutral",
        neutral_phrase=phrase_template,
        end_phrase=phrase_template,
        cutter=effective_cutter,
    )
    spec = PARAMETERS.get(parameter)
    if spec is not None and spec.field is not None and parameter != "cutter":
        mode = replace(mode, **{spec.field: value})
    return ShortSentenceConfig(
        resolve_mode="phrase",
        resolve_modes={"phrase": mode},
        phrase_fallback_tries=phrase_fallback_tries,
    )


def _disabled_config() -> ShortSentenceConfig:
    return ShortSentenceConfig(enabled=False)


def _extract_metadata(trace: Any) -> dict[str, object]:
    if trace is None:
        return {}
    for event in trace.prosody:
        metadata = event.get("short_sentence") if isinstance(event, dict) else None
        if isinstance(metadata, dict):
            return dict(metadata)
    return {}


def _render_result(result: Any) -> RenderedAudio:
    return RenderedAudio(
        np.asarray(result.audio, dtype=np.float32).reshape(-1),
        int(result.sample_rate),
        _extract_metadata(result.trace),
        result.trace,
    )


def _audio_with_pause(sample_rate: int, seconds: float) -> np.ndarray:
    return np.zeros(round(sample_rate * seconds), dtype=np.float32)


def assemble_sweep_audio(
    intro: np.ndarray,
    candidates: list[tuple[np.ndarray, np.ndarray]],
    sample_rate: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Concatenate intro and label/candidate pairs, returning candidate sample offsets."""
    sections = [np.asarray(intro, dtype=np.float32).reshape(-1)]
    sections.append(_audio_with_pause(sample_rate, SECTION_PAUSE_SECONDS))
    offsets: list[tuple[int, int]] = []
    offset = sum(len(section) for section in sections)
    for label, candidate in candidates:
        label_audio = np.asarray(label, dtype=np.float32).reshape(-1)
        candidate_audio = np.asarray(candidate, dtype=np.float32).reshape(-1)
        sections.extend(
            [
                label_audio,
                _audio_with_pause(sample_rate, SHORT_PAUSE_SECONDS),
            ]
        )
        start = offset + len(label_audio) + round(sample_rate * SHORT_PAUSE_SECONDS)
        end = start + len(candidate_audio)
        sections.extend([candidate_audio, _audio_with_pause(sample_rate, CANDIDATE_PAUSE_SECONDS)])
        offsets.append((start, end))
        offset = sum(len(section) for section in sections)
    return np.concatenate(sections).astype(np.float32, copy=False), offsets


def _candidate_manifest_entry(
    index: int,
    value: int | float | str,
    label: str,
    start: int,
    end: int,
    rendered: RenderedAudio,
    cutter: str,
) -> dict[str, object]:
    metadata = rendered.metadata
    outcome = classify_short_sentence_outcome(cutter, metadata)
    entry: dict[str, object] = {
        "index": index,
        "value": value,
        "label": label,
        "start_seconds": start / rendered.sample_rate,
        "end_seconds": end / rendered.sample_rate,
        "final_outcome": outcome,
        "actual_cut_strategy": metadata.get("cut_strategy"),
        "fallback_used": metadata.get("fallback_used"),
        "retry_attempts": metadata.get("retry_attempts", 0),
        "cut_failure_reason": metadata.get("cut_failure_reason"),
        "attempt_history": metadata.get("short_sentence_attempts", []),
        "success_attempt_ordinal": next(
            (
                attempt.get("ordinal")
                for attempt in metadata.get("short_sentence_attempts", [])
                if isinstance(attempt, dict) and attempt.get("succeeded") is True
            ),
            None,
        )
        if isinstance(metadata.get("short_sentence_attempts"), list)
        else None,
        "configured_cutter": metadata.get("cutter", cutter),
        "timing_model_position_count": metadata.get("timing_model_position_count"),
        "generated_token_count": metadata.get("generated_token_count"),
        "timing_model_position_delta": metadata.get("timing_model_position_delta"),
        "cutter_reached": metadata.get(
            "cutter_reached",
            metadata.get("failure_stage")
            not in {"timing-alignment", "timestamp-join", "target-boundary"},
        ),
        "parameter_evaluation": "valid"
        if metadata.get("failure_stage")
        not in {"timing-alignment", "timestamp-join", "target-boundary"}
        else "not-a-cutter-evaluation",
        "timing_failure_reason": metadata.get("timing_failure_reason"),
        "failure_stage": metadata.get("failure_stage"),
    }
    for key in (
        "left_endpoint_abs_amplitude",
        "right_endpoint_abs_amplitude",
        "left_cross_boundary_slope",
        "right_cross_boundary_slope",
        "left_local_rms",
        "right_local_rms",
        "left_anchor_distance_ms",
        "right_anchor_distance_ms",
        "retained_guard_duration_ms",
    ):
        if key in metadata:
            entry[key] = metadata[key]
    return entry


def _is_pre_cutter_failure(metadata: dict[str, object]) -> bool:
    """Return whether timing failed before any cutter parameter could matter."""
    return (
        metadata.get("failure_stage") in {"timing-alignment", "timestamp-join", "target-boundary"}
        or metadata.get("cutter_reached") is False
    )


def run_parameter_sweep(
    *,
    text: str,
    cutter: str,
    parameter: str,
    values: list[int | float | str],
    phrase_template: str,
    voice: str,
    language: str,
    model_source: str,
    model_variant: str,
    model_path: Path | None = None,
    voices_path: Path | None = None,
    speed: float = 1.0,
    phrase_fallback_tries: int = 0,
    random_seed: int = 0,
    keep_invalid: bool = False,
    label_template: str = "{parameter} is {value}.",
    render_text: Callable[[str, ShortSentenceConfig], RenderedAudio] | None = None,
    save_individual: bool = False,
    individual_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Render candidates and labels, returning the combined audio and manifest."""
    validate_parameter(parameter, cutter, values)
    if not phrase_template.count("{segment}") == 1:
        raise ValueError("phrase-template must contain exactly one '{segment}' placeholder")

    pipelines: list[KokoroPipeline] = []
    if render_text is None:

        def render_text(text_to_render: str, config: ShortSentenceConfig) -> RenderedAudio:
            pipeline = KokoroPipeline(
                PipelineConfig(
                    voice=voice,
                    model_source=model_source,
                    model_variant=model_variant,
                    model_path=model_path,
                    voices_path=voices_path,
                    generation=GenerationConfig(lang=language, speed=speed),
                    short_sentence_config=config,
                    return_trace=True,
                )
            )
            pipelines.append(pipeline)
            pipeline.warmup()
            return _render_result(pipeline.run(text_to_render, voice=voice))

    try:
        intro = render_text("Short sentence parameter sweep.", _disabled_config())
        candidates: list[tuple[np.ndarray, np.ndarray]] = []
        manifest_candidates: list[dict[str, object]] = []
        labels: list[str] = []
        rendered_candidates: list[RenderedAudio] = []
        for _index, value in enumerate(values, start=1):
            value_text = format_parameter_value(value)
            candidate_cutter = str(value) if parameter == "cutter" else cutter
            label = label_template.format(
                parameter=display_parameter_name(parameter),
                value=value_text,
            )
            label_audio = render_text(label, _disabled_config())
            config = build_short_sentence_config(
                candidate_cutter,
                phrase_template,
                parameter,
                value,
                phrase_fallback_tries=phrase_fallback_tries,
            )
            rendered = render_text(text, config)
            if _index == 1 and _is_pre_cutter_failure(rendered.metadata) and not keep_invalid:
                reason = rendered.metadata.get(
                    "timing_failure_reason", rendered.metadata.get("failure_stage")
                )
                raise ValueError(
                    "The carrier phrase did not produce valid target timing geometry. "
                    "Parameter sweep aborted because this parameter cannot affect the observed failure. "
                    f"Reason: {reason}"
                )
            if (
                rendered.sample_rate != intro.sample_rate
                or label_audio.sample_rate != intro.sample_rate
            ):
                raise ValueError("all rendered audio must use the same sample rate")
            labels.append(label)
            rendered_candidates.append(rendered)
            candidates.append((label_audio.audio, rendered.audio))
        combined, offsets = assemble_sweep_audio(intro.audio, candidates, intro.sample_rate)
        if save_individual:
            if individual_dir is None:
                raise ValueError("individual_dir is required when save_individual is enabled")
            individual_dir.mkdir(parents=True, exist_ok=True)
            import soundfile as sf

            for index, (value, rendered) in enumerate(
                zip(values, rendered_candidates, strict=True), start=1
            ):
                filename = f"{index:03d}_{parameter}_{format_parameter_value(value)}.wav"
                sf.write(
                    individual_dir / filename,
                    rendered.audio,
                    rendered.sample_rate,
                    subtype="PCM_16",
                )
        for index, (value, rendered, (start, end), label) in enumerate(
            zip(values, rendered_candidates, offsets, labels, strict=True), start=1
        ):
            manifest_candidates.append(
                _candidate_manifest_entry(
                    index,
                    value,
                    label,
                    start,
                    end,
                    rendered,
                    str(value) if parameter == "cutter" else cutter,
                )
            )
        manifest: dict[str, object] = {
            "schema": "pykokoro.short-sentence-sweep.v1",
            "model_source": model_source,
            "model_variant": model_variant,
            "voice": voice,
            "language": language,
            "speed": speed,
            "text": text,
            "phrase_template": phrase_template,
            "cutter": cutter,
            "parameter": parameter,
            "label_template": label_template,
            "candidates": manifest_candidates,
            "candidate_values": values,
            "phrase_fallback_tries": phrase_fallback_tries,
            "random_seed": random_seed,
            "max_phrase_attempts": phrase_fallback_tries + 1,
            "keep_invalid": keep_invalid,
            "sample_rate": intro.sample_rate,
            "pause_seconds": {
                "short": SHORT_PAUSE_SECONDS,
                "candidate": CANDIDATE_PAUSE_SECONDS,
                "section": SECTION_PAUSE_SECONDS,
            },
            "short_sentence_config": {"announcement_enabled": False, "candidate_enabled": True},
        }
        return combined, manifest
    finally:
        for pipeline in pipelines:
            pipeline.close()


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-source", choices=("github", "huggingface"), default="huggingface")
    parser.add_argument("--model-variant", default="v1.0")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--voices-path", type=Path)
    parser.add_argument("--voice", default="af_sarah")
    parser.add_argument("--lang", default="en-us")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--cutter", choices=("energy-valley", "timestamp-adaptive"), default="timestamp-adaptive"
    )
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--values", required=True)
    parser.add_argument("--phrase-template")
    parser.add_argument("--label-template", default="{parameter} is {value}.")
    parser.add_argument("--allow-retries", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Retain candidates after pre-cutter timing failures.",
    )
    parser.add_argument(
        "--output-wav", type=Path, default=Path("artifacts/short_sentence_parameter_sweep.wav")
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("artifacts/short_sentence_parameter_sweep.json")
    )
    parser.add_argument("--save-individual", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _print_dry_run(
    args: argparse.Namespace, values: list[int | float | str], phrase_template: str
) -> None:
    print(f"Text: {args.text}")
    print(f"Cutter: {args.cutter}")
    print(f"Fixed phrase: {phrase_template}")
    print(f"Retries: {args.allow_retries}")
    for index, value in enumerate(values, start=1):
        print(f"{index}  {args.parameter}={format_parameter_value(value)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    args = parser.parse_args()
    if args.allow_retries < 0:
        parser.error("--allow-retries must be non-negative")
    try:
        values = parse_parameter_values(args.parameter, args.values)
        validate_parameter(args.parameter, args.cutter, values)
    except ValueError as exc:
        parser.error(str(exc))
    phrase_template = args.phrase_template or _default_phrase_template(args.text)
    if args.dry_run:
        _print_dry_run(args, values, phrase_template)
        return 0

    combined, manifest = run_parameter_sweep(
        text=args.text,
        cutter=args.cutter,
        parameter=args.parameter,
        values=values,
        phrase_template=phrase_template,
        voice=args.voice,
        language=args.lang,
        model_source=args.model_source,
        model_variant=args.model_variant,
        model_path=args.model_path,
        voices_path=args.voices_path,
        speed=args.speed,
        phrase_fallback_tries=args.allow_retries,
        keep_invalid=args.keep_invalid,
        label_template=args.label_template,
        random_seed=args.random_seed,
        save_individual=args.save_individual,
        individual_dir=args.output_wav.parent / "short_sentence_sweep",
    )
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(args.output_wav, combined, int(manifest["sample_rate"]), subtype="PCM_16")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Candidates")
    print("Value    Outcome             Actual strategy             Retry  Failure")
    for candidate in manifest["candidates"]:
        assert isinstance(candidate, dict)
        print(
            f"{str(candidate['value']):<8} {str(candidate['final_outcome']):<19} "
            f"{str(candidate['actual_cut_strategy'] or '-'): <28} "
            f"{str(candidate['retry_attempts']):>5}  {candidate['cut_failure_reason'] or '-'}"
        )
    print("Created:")
    print(f"  {args.output_wav}")
    print(f"  {args.output_json}")
    print("Listen in printed order. Each sample is preceded by its spoken parameter value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
