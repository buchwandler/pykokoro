#!/usr/bin/env python3
"""
English abbreviations example using pykokoro.

This example demonstrates how pykokoro handles common English abbreviations
including titles, time references, locations, measurements, and more.

Usage:
    python examples/abbreviations.py

Output:
    abbreviations_demo.wav - Generated speech with various abbreviations
"""

import argparse
import logging

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.debug.segment_invariants import check_segment_invariants
from pykokoro.generation_config import GenerationConfig
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.g2p.noop import NoopG2PAdapter
from pykokoro.types import Segment, Trace

# Text with comprehensive abbreviations coverage
TEXT = """
Good morning! Let me introduce you to some people.

Meet Mr. Schmidt, Mrs. Johnson, Ms. Anderson, and Dr. Brown.
Prof. Williams and Rev. Martinez will join us at 3:00 p.m..
St. Patrick's Cathedral is located on 5th Ave. in New York, N.Y..

The meeting is scheduled for Mon., Jan. 15th at the company headquarters.
Please arrive by 9:30 a.m. and bring your I.D. card.

Our office is at 123 Main St., Apt. 4B, Washington, D.C., U.S.A..
For questions, contact us via email at info@example.com or call us ASAP.

The package weighs 5 lbs. and measures 10 ft. by 3 in..
The temperature reached 98°F, or approximately 37°C.
It takes .2 seconds.

Lt. Commander Harris served in the U.S. Navy for 15 yrs..
He earned a Ph.D. in Computer Science from MIT in Sept. 2010.

The company, founded in 1995 A.D., operates in the U.K., Canada, etc..
Our CEO, Mr. Thompson Jr., will present the Q&A session.

Please R.S.V.P. by Fri., Dec. 1st.
P.S. Don't forget to bring your laptop!

Sincerely,
Dr. Emily Clarke, M.D.
Vice President, Research & Development
ABC Corp., Inc.
"""

VOICE = "af_bella"  # American Female voice
LANG = "en-us"  # American English


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce duplicate-word audio with full segment tracing."
    )
    parser.add_argument("--voice", default=VOICE, help="Voice name to use.")
    parser.add_argument("--lang", default=LANG, help="Language code.")
    parser.add_argument(
        "--pause-mode",
        default="tts",
        choices=("tts", "manual"),
        help="Pause handling mode.",
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

    # List of abbreviations being tested
    abbreviations = [
        "Mr. (Mister)",
        "Mrs. (Missus)",
        "Ms. (Miss)",
        "Dr. (Doctor)",
        "Prof. (Professor)",
        "Rev. (Reverend)",
        "Lt. (Lieutenant)",
        "St. (Street/Saint)",
        "Ave. (Avenue)",
        "Apt. (Apartment)",
        "N.Y. (New York)",
        "D.C. (District of Columbia)",
        "U.S.A. (United States of America)",
        "U.K. (United Kingdom)",
        "Mon. (Monday)",
        "Jan. (January)",
        "Sept. (September)",
        "Dec. (December)",
        "Fri. (Friday)",
        "a.m. (ante meridiem)",
        "p.m. (post meridiem)",
        "I.D. (Identification)",
        "ASAP (As Soon As Possible)",
        "lbs. (pounds)",
        "ft. (feet)",
        "in. (inches)",
        "°F (degrees Fahrenheit)",
        "°C (degrees Celsius)",
        "yrs. (years)",
        "Ph.D. (Doctor of Philosophy)",
        "MIT (Massachusetts Institute of Technology)",
        "A.D. (Anno Domini)",
        "Jr. (Junior)",
        "CEO (Chief Executive Officer)",
        "Q&A (Questions and Answers)",
        "R.S.V.P. (Répondez s'il vous plaît)",
        "P.S. (Post Script)",
        "M.D. (Medical Doctor)",
        "Inc. (Incorporated)",
        "Corp. (Corporation)",
        "etc. (et cetera)",
    ]

    print("Abbreviations being tested:")
    for abbr in abbreviations:
        print(f"  - {abbr}")
    print()

    output_file = "abbreviations_demo.wav"
    result = pipeline.run(TEXT)
    result.save_wav(output_file)

    doc = doc_parser.parse(TEXT, cfg, Trace())

    print(f"clean_text length: {len(doc.clean_text)}")
    print_segments(result.segments)
    print_phoneme_segments(result.phoneme_segments)
    check_segment_invariants(result.segments, doc.clean_text)

    if result.trace and result.trace.warnings:
        print("Warnings:")
        for warning in result.trace.warnings:
            print(f"  - {warning}")

    print(f"Wrote WAV to: {output_file}")
    duration = len(result.audio) / result.sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")
    print("\nListen to verify that abbreviations are pronounced correctly!")


if __name__ == "__main__":
    main()
