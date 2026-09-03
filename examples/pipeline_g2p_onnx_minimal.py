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
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import Segment, Trace


class PlainDocumentParser:
    def parse(self, text: str, cfg: PipelineConfig, trace: Trace) -> DocumentResult:
        _ = (cfg, trace)
        return DocumentResult(
            clean_text=text,
            segments=[
                Segment(
                    id="p0_s0_c0_seg0",
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    paragraph_idx=0,
                    sentence_idx=0,
                    clause_idx=0,
                )
            ],
        )


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
        voice="af",
        generation=GenerationConfig(lang="en-us"),
        return_trace=True,
    )
    pipeline = KokoroPipeline(
        cfg,
        doc_parser=PlainDocumentParser(),
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

    doc = pipeline.doc_parser.parse(text, cfg, Trace())
    segments = doc.segments
    print(segments)
    phoneme_segments = pipeline.g2p.phonemize(segments, doc, cfg, Trace())
    print(f"Text: {text}")
    print("Phonemes:")
    for segment in phoneme_segments:
        print(segment.phonemes)


if __name__ == "__main__":
    main()
