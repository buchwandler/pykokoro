#!/usr/bin/env python3
"""Render one oversized request with engine-managed sentence splitting.

This synthesis may install model assets. The input is never manually split:
internal acoustic chunks still belong to one public request and one result.
"""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig, SynthesisSegment

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

LONG_TEXT = " ".join(
    (
        "A careful explorer studies the changing landscape before choosing a path.",
        "Each observation adds context, and each small decision shapes the journey.",
        "When the weather changes, the explorer pauses, checks the map, and adapts.",
        "The purpose is not to hurry through the story but to preserve its meaning.",
    )
    * 2
)


def make_request() -> SynthesisSegment:
    return SynthesisSegment(
        id="long-text",
        text=LONG_TEXT,
        language="en-us",
        voice="af_sarah",
    )


def make_config() -> SynthesisConfig:
    return SynthesisConfig(
        voice="af_sarah",
        generation=GenerationConfig(lang="en-us"),
        long_text_split="sentence",
        long_text_use_spacy=False,
        return_trace=True,
    )


def main() -> None:
    request = make_request()
    print("PyKokoro receives the complete original request; do not split it here.")
    with KokoroSynthesizer(make_config()) as synthesizer:
        results = tuple(synthesizer.synthesize_segments((request,)))
    if len(results) != 1 or results[0].text != request.text:
        raise RuntimeError("long-text rendering did not preserve one original request/result")
    result = results[0]
    output = artifact_path("long_text.wav")
    result.save_wav(output)
    print(f"Wrote one rendered request to {output}")
    if result.token_ids:
        print(f"Model token count: {len(result.token_ids)}")
    if result.word_timings:
        print(f"Word timings: {len(result.word_timings)}")
    if result.trace is not None:
        print(
            f"Internal acoustic chunks: {result.trace.model.get('acoustic_chunk_count', 'available')}"
        )
    print("Internal chunks are not separate public requests or results.")


if __name__ == "__main__":
    main()
