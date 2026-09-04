from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.doc_parsers.plain import PlainTextDocumentParser
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.phoneme_processing.noop import NoopPhonemeProcessorAdapter
from pykokoro.stages.segmentation.phrasplit import PhrasplitSentenceSegmenter
from pykokoro.stages.text_preparation.spokenform import SpokenformTextPreparer
from pykokoro.tokenizer import TokenizerConfig

LANGUAGE_TO_G2P = {"en-US": "en-us", "en-GB": "en-gb", "de-DE": "de"}


@dataclass(frozen=True, slots=True)
class FrontendVariant:
    """A named frontend configuration, independent of the acoustic model."""

    id: str = "default"
    language: str = "en-us"
    options: Mapping[str, Any] = field(default_factory=dict)

    def tokenizer_config(self, backend: str = "kokorog2p") -> TokenizerConfig:
        options = dict(self.options)
        options.setdefault("backend", backend)
        return TokenizerConfig(**options)


@dataclass(frozen=True, slots=True)
class FrontendResult:
    text: str
    clean_text: str
    source_text: str | None
    segments: tuple[Any, ...]
    phoneme_segments: tuple[Any, ...]
    trace: Any
    document_metadata: Mapping[str, Any]
    audio: Any = None
    sample_rate: int | None = None


class NoOnnxFrontend:
    """Reusable PyKokoro frontend whose downstream adapters never load ONNX."""

    def __init__(
        self,
        locale: str,
        *,
        backend: str = "kokorog2p",
        ssmd: bool = False,
        variant: FrontendVariant | None = None,
        pause_mode: str = "tts",
    ) -> None:
        if locale not in LANGUAGE_TO_G2P:
            raise ValueError(f"unsupported locale: {locale!r}")
        language = LANGUAGE_TO_G2P[locale]
        selected = variant or FrontendVariant(language=language)
        if selected.language != language:
            raise ValueError("frontend variant language does not match locale")
        tokenizer = selected.tokenizer_config(backend)
        config = PipelineConfig(
            generation=GenerationConfig(lang=language, pause_mode=pause_mode),
            tokenizer_config=tokenizer,
            return_trace=True,
        )
        self.locale = locale
        self.variant = selected
        self.pipeline = KokoroPipeline(
            config,
            doc_parser=SsmdDocumentParser() if ssmd else PlainTextDocumentParser(),
            text_preparer=SpokenformTextPreparer(),
            sentence_segmenter=PhrasplitSentenceSegmenter(),
            phoneme_processing=NoopPhonemeProcessorAdapter(),
            audio_generation=NoopAudioGenerationAdapter(seconds_per_segment=0.0),
            audio_postprocessing=NoopAudioPostprocessingAdapter(),
        )

    def run(self, text: str) -> FrontendResult:
        result = self.pipeline.run(text)
        return FrontendResult(
            text=text,
            clean_text=result.clean_text,
            source_text=result.source_text,
            segments=tuple(result.segments),
            phoneme_segments=tuple(result.phoneme_segments),
            trace=result.trace,
            document_metadata=result.document_metadata,
            audio=result.audio,
            sample_rate=result.sample_rate,
        )

    def close(self) -> None:
        self.pipeline.close()

    def __enter__(self) -> NoOnnxFrontend:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


PyKokoroFrontend = NoOnnxFrontend

__all__ = [
    "FrontendResult",
    "FrontendVariant",
    "LANGUAGE_TO_G2P",
    "NoOnnxFrontend",
    "PyKokoroFrontend",
]
