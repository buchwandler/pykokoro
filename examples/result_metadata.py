#!/usr/bin/env python3
"""Inspect request results, reproducibility identity, timing, and inference trace.

Synthesis may install model assets. Fields are printed only when available for
the selected frontend and runtime.
"""

from __future__ import annotations

from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path


def make_config() -> SynthesisConfig:
    return SynthesisConfig(
        voice="af_sarah",
        generation=GenerationConfig(lang="en-us", random_seed=7),
        return_trace=True,
    )


def main() -> None:
    with KokoroSynthesizer(make_config()) as synthesizer:
        result = synthesizer.synthesize_text(
            "A reproducible request can include useful timing and diagnostics.",
            language="en-us",
        )
    output = artifact_path("result_metadata.wav")
    result.save_wav(output)

    print(f"ID: {result.id}")
    print(f"Sample rate: {result.sample_rate}")
    print(f"Text: {result.text}")
    print(f"Language: {result.language}")
    print(f"Voice: {result.voice}")
    print(f"Phonemes: {result.phonemes}")
    print(f"Token IDs: {result.token_ids}")
    print(f"Diagnostics: {result.diagnostics}")
    print(f"Short-sentence mode: {result.short_sentence_mode}")
    print(f"Voice-level applications: {result.voice_level_applications}")
    if result.synthesis_identity is not None:
        print(f"Synthesis cache key: {result.synthesis_identity.cache_key}")
    for timing in result.word_timings:
        print(
            timing.text,
            timing.start_seconds(result.sample_rate),
            timing.end_seconds(result.sample_rate),
        )
    if result.trace is not None:
        print(f"Inference summary: {result.trace.inference_summary()}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
