#!/usr/bin/env python3
"""Reproduce duplicate words with segment tracing."""

from __future__ import annotations

import argparse
import logging

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.debug.segment_invariants import check_segment_invariants
from pykokoro.generation_config import GenerationConfig
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.g2p.noop import NoopG2PAdapter
from pykokoro.types import Segment, Trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce duplicate-word audio with full segment tracing."
    )
    parser.add_argument(
        "--text",
        default="Hello world. Hello world.",
        help="Input text to synthesize.",
    )
    parser.add_argument("--voice", default="af", help="Voice name to use.")
    parser.add_argument("--lang", default="en-us", help="Language code.")
    parser.add_argument(
        "--pause-mode",
        default="tts",
        choices=("tts", "manual"),
        help="Pause handling mode.",
    )
    parser.add_argument(
        "--out",
        default=str(artifact_path("repro_dup_words.wav")),
        help="Path to write WAV output.",
    )
    parser.add_argument(
        "--noop-g2p",
        action="store_true",
        help="Replace g2p with a no-op adapter (forces no-op synth).",
    )
    parser.add_argument(
        "--noop-synth",
        action="store_true",
        help="Replace synth with silence output.",
    )
    return parser.parse_args()


def print_segments(segments: list[Segment]) -> None:
    print("Segments:")
    for seg in segments:
        print(f"  {seg.id}: {seg.char_start}:{seg.char_end} text={seg.text!r}")


def print_phoneme_segments(phoneme_segments: list) -> None:
    print("Phoneme Segments:")
    for seg in phoneme_segments:
        print(f"  {seg.char_start}:{seg.char_end} text={seg.text!r}")
        print(seg)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    generation = GenerationConfig(lang=args.lang, pause_mode=args.pause_mode)
    cfg = PipelineConfig(voice=args.voice, generation=generation, return_trace=True)

    doc_parser = SsmdDocumentParser()

    noop_synth = args.noop_synth
    if args.noop_g2p and not args.noop_synth:
        print("Forcing no-op synth because no-op g2p omits tokens.")
        noop_synth = True

    g2p = NoopG2PAdapter() if args.noop_g2p else None
    audio_generation = NoopAudioGenerationAdapter() if noop_synth else None
    audio_postprocessing = NoopAudioPostprocessingAdapter() if noop_synth else None

    pipeline = KokoroPipeline(
        cfg,
        doc_parser=doc_parser,
        g2p=g2p,
        audio_generation=audio_generation,
        audio_postprocessing=audio_postprocessing,
    )

    result = pipeline.run(args.text)
    result.save_wav(args.out)

    doc = doc_parser.parse(args.text, cfg, Trace())

    print(f"clean_text length: {len(doc.clean_text)}")
    print_segments(result.segments)
    print_phoneme_segments(result.phoneme_segments)
    check_segment_invariants(result.segments, doc.clean_text)

    if result.trace and result.trace.warnings:
        print("Warnings:")
        for warning in result.trace.warnings:
            print(f"  - {warning}")

    print(f"Wrote WAV to: {args.out}")


if __name__ == "__main__":
    main()
