"""Pipeline stage showcase with explicit stage wiring.

Usage:
    python examples/pipeline_stage_showcase.py
"""

from __future__ import annotations

from pathlib import Path

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.onnx_backend import Kokoro
from pykokoro.stages.audio_generation.onnx import OnnxAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
from pykokoro.stages.doc_parsers.plain import PlainTextDocumentParser
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.phoneme_processing.noop import NoopPhonemeProcessorAdapter
from pykokoro.stages.phoneme_processing.onnx import OnnxPhonemeProcessorAdapter

SSMD_TEXT = """
[Hello]{voice="af_sarah"} world...500ms This is a full SSMD pipeline demo.
""".strip()

PARAGRAPH_TEXT = """
This is the first paragraph. It has multiple sentences.

This is the second paragraph. It uses paragraph boundaries.
""".strip()

SIMPLE_TEXT = "This is a minimal pipeline with only g2p and audio generation."


def build_kokoro(cfg: PipelineConfig) -> Kokoro:
    model_path = Path(cfg.model_path) if cfg.model_path else None
    voices_path = Path(cfg.voices_path) if cfg.voices_path else None
    return Kokoro(
        model_path=model_path,
        voices_path=voices_path,
        model_quality=cfg.model_quality,
        model_source=cfg.model_source,
        model_variant=cfg.model_variant,
        provider=cfg.provider,
        provider_options=cfg.provider_options,
        session_options=cfg.session_options,
        tokenizer_config=cfg.tokenizer_config,
        espeak_config=cfg.espeak_config,
        short_sentence_config=cfg.short_sentence_config,
    )


def run_pipeline(label: str, pipeline: KokoroPipeline, text: str, output_path: str) -> None:
    result = pipeline.run(text)
    result.save_wav(output_path)
    print(f"[{label}] Wrote {output_path}")


def main() -> None:
    cfg = PipelineConfig(
        voice="af_heart",
        generation=GenerationConfig(lang="en-us"),
    )
    kokoro = build_kokoro(cfg)

    full_pipeline = KokoroPipeline(
        cfg,
        doc_parser=SsmdDocumentParser(),
        g2p=KokoroG2PAdapter(),
        phoneme_processing=OnnxPhonemeProcessorAdapter(kokoro),
        audio_generation=OnnxAudioGenerationAdapter(kokoro),
        audio_postprocessing=OnnxAudioPostprocessingAdapter(kokoro),
    )

    paragraph_pipeline = KokoroPipeline(
        cfg,
        doc_parser=PlainTextDocumentParser(),
        g2p=KokoroG2PAdapter(),
        phoneme_processing=OnnxPhonemeProcessorAdapter(kokoro),
        audio_generation=OnnxAudioGenerationAdapter(kokoro),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )

    minimal_pipeline = KokoroPipeline(
        cfg,
        doc_parser=PlainTextDocumentParser(),
        g2p=KokoroG2PAdapter(),
        phoneme_processing=NoopPhonemeProcessorAdapter(),
        audio_generation=OnnxAudioGenerationAdapter(kokoro),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )

    run_pipeline(
        "ssmd-full",
        full_pipeline,
        SSMD_TEXT,
        artifact_path("pipeline_stage_showcase_ssmd.wav"),
    )
    run_pipeline(
        "paragraph",
        paragraph_pipeline,
        PARAGRAPH_TEXT,
        artifact_path("pipeline_stage_showcase_paragraph.wav"),
    )
    run_pipeline(
        "minimal",
        minimal_pipeline,
        SIMPLE_TEXT,
        artifact_path("pipeline_stage_showcase_minimal.wav"),
    )


if __name__ == "__main__":
    main()
