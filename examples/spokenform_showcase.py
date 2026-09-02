#!/usr/bin/env python3
"""Show automatic spoken-form preparation provided by Spokenform."""

from __future__ import annotations

import argparse
from pathlib import Path

import kokorog2p

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

TEXT = (
    "Dr. Smith will see you at 10:30 on 05/20/2023. "
    "The box weighs 5 kg and costs $10.99. "
    "The temperature is 98.6°F. "
    "She finished in 1st place. "
    "Call me at 555-123-4567. "
    "The final score was 3-1, and 2 + 2 = 4."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print the kokorog2p result without loading a synthesis model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spokenform_showcase.wav"),
        help="WAV path for normal synthesis mode.",
    )
    return parser.parse_args()


def inspect_spokenform() -> object:
    """Print and return the high-level kokorog2p result used for inspection."""
    prepared = kokorog2p.phonemize(
        TEXT,
        language="en-us",
        return_phonemes=True,
        return_ids=True,
    )

    print("SOURCE")
    print("------")
    print(TEXT)

    print("\nSPOKEN FORM")
    print("-----------")
    print(getattr(prepared, "extended_text", ""))

    print("\nPHONEMES")
    print("--------")
    phonemes = str(getattr(prepared, "phonemes", "") or "")
    preview = phonemes if len(phonemes) <= 240 else f"{phonemes[:237]}..."
    print(preview)

    warnings = list(getattr(prepared, "warnings", []) or [])
    if warnings:
        print("\nWARNINGS")
        print("--------")
        for warning in warnings:
            print(f"- {warning}")

    return prepared


def main() -> None:
    args = parse_args()
    inspect_spokenform()

    if args.inspect_only:
        return

    cfg = PipelineConfig(
        voice="af",
        generation=GenerationConfig(lang="en-us"),
        return_trace=True,
    )
    pipeline = KokoroPipeline(cfg)
    result = pipeline.run(TEXT)
    result.save_wav(str(args.output))
    print(f"\nWrote {args.output}")

    if result.trace is not None and result.trace.warnings:
        print("\nPYKOKORO TRACE WARNINGS")
        print("-----------------------")
        for warning in result.trace.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
