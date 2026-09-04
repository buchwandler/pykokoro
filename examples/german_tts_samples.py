#!/usr/bin/env python3
"""Generate the three canonical German TTS benchmark sentences as separate WAV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import soundfile as sf

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.model_profiles import MODEL_PROFILES, normalize_language_code
from pykokoro.tokenizer import TokenizerConfig

SAMPLES = (
    (
        "sentence_1",
        "An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes "
        "besucht. Dabei war eigentlich immer sehr schönes Wetter gewesen.",
    ),
    (
        "sentence_2",
        "Dr. A. Smithe von der NATO (und nicht vom CIA) versorgt z.B. - meines "
        "Wissens nach - die Heroin seit dem 15.3.00 tgl. mit 13,84 Gramm Heroin "
        "zu 1,04 DM das Gramm.",
    ),
    (
        "sentence_3",
        "Die Manpowerdiskussion wird gecancelt, du kannst das File vom Server downloaden.",
    ),
)

DEFAULT_MODEL = "v1.2-de-martin"
DEFAULT_LEXICON = "gold"
LEXICON_CHOICES = ("gold", "crane", "espeak", "olaph")
MODEL_LABELS = {
    "v1.2-de-martin": "Martin v1.2",
    "de-crane": "Kerstin / Crane",
    "de-thorsten": "Thorsten",
}


def available_german_models() -> tuple[str, ...]:
    """Return deterministic IDs for locally runnable German model profiles."""
    variants: set[str] = set()
    for profile in MODEL_PROFILES.values():
        if not profile.runtime_available:
            continue
        languages = tuple(normalize_language_code(code) for code in profile.language_codes)
        if not any(code == "de" or code.startswith("de-") for code in languages):
            continue
        variants.add(profile.variant)

    ordered = sorted(variants)
    if DEFAULT_MODEL in ordered:
        ordered.remove(DEFAULT_MODEL)
        ordered.insert(0, DEFAULT_MODEL)
    return tuple(ordered)


def _profile_for_model(model_id: str) -> Any:
    """Find a selected model in the local runtime profile inventory."""
    for profile in MODEL_PROFILES.values():
        if profile.variant == model_id and profile.runtime_available:
            languages = tuple(normalize_language_code(code) for code in profile.language_codes)
            if any(code == "de" or code.startswith("de-") for code in languages):
                return profile
    raise ValueError(f"Unknown or unavailable German model: {model_id!r}")


def make_config(*, model_id: str, lexicon: str) -> PipelineConfig:
    """Build the explicit pipeline configuration for one German sample run."""
    if lexicon not in LEXICON_CHOICES:
        raise ValueError(f"Unknown German lexicon: {lexicon!r}")

    profile = _profile_for_model(model_id)
    return PipelineConfig(
        voice=profile.default_voice,
        model_source=profile.source,
        model_variant=profile.variant,
        model_quality="fp32",
        allow_experimental_frontend=profile.frontend_experimental,
        generation=GenerationConfig(lang="de", speed=1.0),
        tokenizer_config=TokenizerConfig(lexicons=(lexicon,)),
        return_trace=True,
    )


def _print_trace_warnings(result: Any) -> None:
    trace = getattr(result, "trace", None)
    warnings = getattr(trace, "warnings", None)
    if warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")


def synthesize_samples(*, model_id: str, lexicon: str) -> list[Path]:
    """Synthesize each canonical sentence through one shared pipeline."""
    config = make_config(model_id=model_id, lexicon=lexicon)
    outputs: list[Path] = []

    with KokoroPipeline(config) as pipeline:
        for sample_id, text in SAMPLES:
            print(f"[{sample_id}] ...")
            result = pipeline.run(text)
            output = artifact_path(
                Path("german_tts_samples") / model_id / lexicon / f"{sample_id}.wav"
            )
            sf.write(output, result.audio, result.sample_rate)
            outputs.append(output)
            print(f"  wrote: {output}")
            _print_trace_warnings(result)

    return outputs


def _model_help() -> str:
    lines = ["German models:"]
    for model_id in available_german_models():
        label = MODEL_LABELS.get(model_id, "locally supported German model")
        suffix = " (default)" if model_id == DEFAULT_MODEL else ""
        lines.append(f"  {model_id:<17} {label}{suffix}")
    lines.extend(["", "German lexicons:"])
    lines.extend(f"  {lexicon}" for lexicon in LEXICON_CHOICES)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without accessing the remote registry."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_model_help(),
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_ID",
        choices=available_german_models(),
        default=DEFAULT_MODEL,
        help="German acoustic model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--lexicon",
        metavar="LEXICON",
        choices=LEXICON_CHOICES,
        default=DEFAULT_LEXICON,
        help="German lexicon source (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate three separate WAV files from the canonical German sentences."""
    args = build_parser().parse_args(argv)
    model_label = MODEL_LABELS.get(args.model, "locally supported German model")
    print("Generating canonical German TTS samples:")
    print(f"  model:   {args.model} ({model_label})")
    print(f"  lexicon: {args.lexicon}")
    synthesize_samples(model_id=args.model, lexicon=args.lexicon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
