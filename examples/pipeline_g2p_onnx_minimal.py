#!/usr/bin/env python3
"""
Minimal pipeline example: G2P + ONNX only (no SSMD).

This example wires a custom document parser so the pipeline:
- skips SSMD parsing
- produces a single segment for the full paragraph
- runs G2P and ONNX synthesis only

Usage:
    python examples/pipeline_g2p_onnx_minimal.py

Output:
    pipeline_g2p_onnx_minimal.wav
"""

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig


def main() -> None:
    text = (
        "This paragraph is synthesized without SSMD parsing or sentence splitting. "
        "The pipeline uses a single segment for the full text and runs only G2P "
        "and ONNX synthesis."
    )
    text = (
        "'That's ridiculous!' I protested. 'I'm not gonna stand here and "
        "let you insult me! What's your problem anyway?'"
    )

    cfg = PipelineConfig(
        voice="af_sarah",
        generation=GenerationConfig(lang="en-us"),
        return_trace=True,
    )
    pipeline = KokoroPipeline(
        cfg,
    )
    result = pipeline.run(text)
    output_path = artifact_path("pipeline_g2p_onnx_minimal.wav")
    result.save_wav(output_path)
    print(f"Wrote {output_path}")

    trace = result.trace
    if trace is not None:
        if trace.warnings:
            print("Warnings:")
            for warning in trace.warnings:
                print(f"- {warning}")
        if trace.events:
            print("Trace events:")
            for event in trace.events:
                print(f"- {event.stage}:{event.name} {event.ms:.2f}ms")

    print(f"Text: {result.clean_text}")
    print("Segments:")
    for segment in result.segments:
        print(segment)
    print("Phonemes:")
    for segment in result.phoneme_segments:
        print(segment.phonemes)


if __name__ == "__main__":
    main()
