from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..ssmd_config import SSMDDiagnostic
from ..types import (
    AnnotationSpan,
    BoundaryEvent,
    PhonemeSegment,
    Segment,
    TextPreparationInfo,
    Trace,
)

if TYPE_CHECKING:
    from ..pipeline_config import PipelineConfig
    from ..runtime.linguistics import LinguisticRequestState


@dataclass
class DocumentResult:
    clean_text: str
    annotation_spans: list[AnnotationSpan] = field(default_factory=list)
    boundary_events: list[BoundaryEvent] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    header: dict[str, object] = field(default_factory=dict)
    body: str | None = None
    diagnostics: list[SSMDDiagnostic] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    structural_clean_text: str | None = None
    preparation: TextPreparationInfo | None = None

    # Ephemeral request state; never part of serialized public results.
    linguistic_state: LinguisticRequestState | None = field(default=None, repr=False, compare=False)


class DocumentParser(Protocol):
    def parse(self, text: str, cfg: PipelineConfig, trace: Trace) -> DocumentResult: ...


class TextPreparer(Protocol):
    def prepare(self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace) -> DocumentResult: ...


class SentenceSegmenter(Protocol):
    def split(self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace) -> list[Segment]: ...


class G2PAdapter(Protocol):
    def phonemize(
        self,
        segments: list[Segment],
        doc: DocumentResult,
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]: ...


class PhonemeProcessor(Protocol):
    def process(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]: ...


class AudioGeneratorStage(Protocol):
    def generate(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]: ...


class AudioPostprocessor(Protocol):
    def postprocess(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> np.ndarray: ...


class Synthesizer(Protocol):
    def synthesize(
        self, phoneme_segments: list[PhonemeSegment], cfg: PipelineConfig, trace: Trace
    ) -> np.ndarray: ...
