#!/usr/bin/env python3
"""
Demonstrate automatic pauses at high-confidence English clausal commas.

The example compares normal TTS-controlled punctuation with pause_mode="auto".
The auto case must detect exactly one high-confidence clausal comma and assign
pause_clause to the preceding phoneme segment.

Prerequisites:
    pip install -e ".[cpu]"
    python -m spacy download en_core_web_sm

Usage:
    python examples/english_clausal_comma_pause.py

Output:
    english_clausal_comma_tts.wav
    english_clausal_comma_auto.wav
"""

from __future__ import annotations

from typing import Any

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig, with_spacy_model_size

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

TEXT = "It had picked up the sound of a explosion, direction suggested it was behind."
LANG = "en-us"
VOICE = "af_heart"
PAUSE_CLAUSE = 0.30


def _clausal_comma_diagnostic(result: Any) -> Any | None:
    if result.trace is None:
        return None
    return next(
        (event for event in result.trace.events if event.name == "clausal_comma_boundaries"),
        None,
    )


def _print_segment_pauses(label: str, result: Any) -> None:
    print(f"\n{label}:")
    for segment in result.phoneme_segments:
        print(
            f"  {segment.text!r} "
            f"pause_before={segment.pause_before:.3f}s "
            f"pause_after={segment.pause_after:.3f}s"
        )


def _trace_warnings(result: Any) -> str:
    warnings = result.trace.warnings if result.trace is not None else []
    return "\n".join(f"  - {warning}" for warning in warnings) or "  (none)"


def main() -> None:
    """Generate an audible A/B comparison for one clausal comma."""
    base_config = PipelineConfig(
        voice=VOICE,
        generation=GenerationConfig(lang=LANG),
        return_trace=True,
        retain_segment_audio=False,
    )

    # High-confidence comma detection requires dependency-capable analysis.
    config = with_spacy_model_size(base_config)
    pipe = KokoroPipeline(config)

    baseline = GenerationConfig(
        lang=LANG,
        pause_mode="tts",
        pause_variance=0.0,
    )
    automatic = GenerationConfig(
        lang=LANG,
        pause_mode="auto",
        pause_clause=PAUSE_CLAUSE,
        pause_sentence=0.0,
        pause_paragraph=0.0,
        pause_variance=0.0,
    )

    print("High-confidence clausal-comma pause demo")
    print(f"Text: {TEXT}")
    print(f"Configured automatic clause pause: {PAUSE_CLAUSE:.2f}s")

    tts_result = pipe.run(TEXT, generation=baseline)
    auto_result = pipe.run(TEXT, generation=automatic)

    diagnostic = _clausal_comma_diagnostic(auto_result)
    added = 0 if diagnostic is None else int(diagnostic.details.get("added", 0))
    if added != 1:
        raise RuntimeError(
            "Expected exactly one high-confidence clausal-comma boundary, "
            f"but PyKokoro added {added}.\n"
            "Install Phrasplit 0.3.8+ and a compatible local English spaCy model.\n"
            f"Trace warnings:\n{_trace_warnings(auto_result)}"
        )

    tts_path = artifact_path("english_clausal_comma_tts.wav")
    auto_path = artifact_path("english_clausal_comma_auto.wav")
    tts_result.save_wav(str(tts_path))
    auto_result.save_wav(str(auto_path))

    _print_segment_pauses("TTS-controlled punctuation", tts_result)
    _print_segment_pauses("Automatic clausal-comma pause", auto_result)

    paused = [
        segment
        for segment in auto_result.phoneme_segments
        if abs(segment.pause_after - PAUSE_CLAUSE) < 1e-6
    ]
    if len(paused) != 1:
        raise RuntimeError(
            "Expected exactly one preceding phoneme segment to own the configured "
            f"{PAUSE_CLAUSE:.2f}s clausal-comma pause, found {len(paused)}."
        )

    print("\nDetected clausal-comma boundary:")
    print(f"  after {paused[0].text!r}: {paused[0].pause_after:.3f}s")
    print(f"\nCreated baseline: {tts_path}")
    print(f"Created automatic: {auto_path}")
    print("Listen to the same sentence in both files; the automatic version inserts")
    print("one deterministic pause after the high-confidence clausal comma.")


if __name__ == "__main__":
    main()
