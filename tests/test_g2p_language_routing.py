from __future__ import annotations

import re

from pykokoro.generation_config import GenerationConfig
from pykokoro.language_detection import LanguageDetectionConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.runtime.language_plan import LanguageRun
from pykokoro.runtime.linguistics import (
    LinguisticRequestState,
    PreparedRunAnalysis,
    TokenAnnotation,
)
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.tokenizer import TokenizerConfig
from pykokoro.types import AnnotationSpan, Segment, Trace

SOURCE = "Die Manpowerdiskussion wird gecancelt, du kannst das File downloaden."


def _segment(text: str, language: str | None = None) -> Segment:
    return Segment(
        id="segment-0",
        text=text,
        char_start=0,
        char_end=len(text),
        meta={"language": language} if language else {},
        paragraph_idx=0,
        sentence_idx=0,
        clause_idx=0,
    )


def _analyzed_doc(text: str) -> DocumentResult:
    annotations = tuple(
        TokenAnnotation(
            start=match.start(),
            end=match.end(),
            text=match.group(),
            language="de",
        )
        for match in re.finditer(r"\w+", text)
    )
    doc = DocumentResult(clean_text=text, segments=[_segment(text)])
    doc.linguistic_state = LinguisticRequestState(
        prepared_analysis=[
            PreparedRunAnalysis(
                run=LanguageRun(0, len(text), "de"),
                text=text,
                doc=None,
                annotations=annotations,
            )
        ]
    )
    return doc


def _route_signature(doc: DocumentResult) -> list[tuple[object, ...]]:
    return [
        (
            route["char_start"],
            route["char_end"],
            route["text"],
            route["reason"],
            tuple(
                (
                    fragment["char_start"],
                    fragment["char_end"],
                    fragment["text"],
                    fragment["language"],
                    fragment["kind"],
                    fragment.get("evidence_lexicon_id"),
                )
                for fragment in route["fragments"]
            ),
        )
        for route in doc.metadata["g2p_language_routes"]
    ]


def _run(text: str, doc: DocumentResult, cache_dir: str) -> list[tuple[object, ...]]:
    config = PipelineConfig(
        cache_dir=cache_dir,
        generation=GenerationConfig(lang="de"),
        language_detection=LanguageDetectionConfig(mode="auto", languages=("de", "en")),
        tokenizer_config=TokenizerConfig(use_spacy=False),
    )
    segments = doc.segments or [_segment(text)]
    KokoroG2PAdapter().phonemize(segments, doc, config, Trace())
    return _route_signature(doc)


def test_prepared_analysis_does_not_disable_auto_routing(tmp_path) -> None:
    without_analysis = DocumentResult(clean_text=SOURCE, segments=[_segment(SOURCE)])
    with_analysis = _analyzed_doc(SOURCE)
    with_default_span = DocumentResult(
        clean_text=SOURCE,
        annotation_spans=[AnnotationSpan(0, len(SOURCE), {"lang": "de"})],
        segments=[_segment(SOURCE, "de")],
    )

    plain_routes = _run(SOURCE, without_analysis, str(tmp_path / "plain"))
    default_routes = _run(SOURCE, with_default_span, str(tmp_path / "default"))
    analyzed_routes = _run(SOURCE, with_analysis, str(tmp_path / "analyzed"))

    assert default_routes == plain_routes
    assert analyzed_routes == plain_routes
    routed_text = {route[2] for route in analyzed_routes}
    assert {"Manpowerdiskussion", "gecancelt", "File", "downloaden"} <= routed_text
