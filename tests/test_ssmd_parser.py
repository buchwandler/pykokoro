from __future__ import annotations

from types import SimpleNamespace

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.ssmd_config import ResolvedPauseDefaults
from pykokoro.ssmd_parser import SSMDSegment, parse_ssmd_document
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.types import Segment, Trace


def test_parse_ssmd_document_forwards_spacy_settings(monkeypatch):
    captured: dict[str, object] = {}

    def fake_body_parser(*args: object, **kwargs: object):
        captured.update(kwargs)
        return 0.0, []

    monkeypatch.setattr("pykokoro.ssmd_parser._parse_ssmd_body_to_segments", fake_body_parser)

    parsed = parse_ssmd_document(
        "Hello.",
        lang="en-us",
        spacy_model="en_core_web_sm",
        model_size="sm",
        use_spacy=True,
    )

    assert parsed.segments == ()
    assert captured["spacy_model"] == "en_core_web_sm"
    assert captured["model_size"] == "sm"
    assert captured["use_spacy"] is True


def test_parse_ssmd_document_keeps_sentence_diagnostics(monkeypatch):
    diagnostic = SimpleNamespace(selected_model="en_core_web_lg", selected_model_size="lg")

    def fake_body_parser(*args: object, **kwargs: object):
        sink = kwargs["sentence_diagnostics"]
        sink.append(diagnostic)
        return 0.0, []

    monkeypatch.setattr("pykokoro.ssmd_parser._parse_ssmd_body_to_segments", fake_body_parser)

    parsed = parse_ssmd_document("Hello.")

    assert parsed.sentence_diagnostics is diagnostic


def test_ssmd_auto_sentence_pause_uses_generation_config() -> None:
    segments = [
        Segment("s0", "First.", 0, 6, {}, 0, 0),
        Segment("s1", "Second.", 8, 15, {}, 0, 1),
    ]
    candidates = SsmdDocumentParser()._sentence_candidates(
        segments,
        None,
        "auto",
        0.23,
    )

    assert [candidate.duration_s for candidate in candidates] == [0.23]


def test_ssmd_zero_duration_header_default_is_preserved() -> None:
    segments = [
        Segment("s0", "First.", 0, 6, {}, 0, 0),
        Segment("s1", "Second.", 8, 15, {}, 0, 1),
    ]
    candidates = SsmdDocumentParser()._sentence_candidates(
        segments,
        ResolvedPauseDefaults(enabled=True, sentence=0.0),
        "auto",
        0.23,
    )

    assert [candidate.duration_s for candidate in candidates] == [0.0]


def test_ssmd_zero_duration_paragraph_header_default_is_preserved() -> None:
    parser = SsmdDocumentParser()
    segments = [
        # The two paragraph segments force the paragraph-default branch.
        # Voice metadata is intentionally absent so only paragraph behavior is tested.
        SSMDSegment("First.", paragraph=0, sentence=0),
        SSMDSegment("Second.", paragraph=1, sentence=0),
    ]
    clean_text, _spans, boundaries, _doc_segments = parser._build_document(
        segments,
        0.0,
        Trace(),
        ResolvedPauseDefaults(enabled=True, paragraph=0.0),
        PipelineConfig(generation=GenerationConfig(pause_mode="auto")),
    )

    assert clean_text == "First.\n\nSecond."
    paragraph_candidates = [
        boundary for boundary in boundaries if getattr(boundary, "kind", None) == "paragraph"
    ]
    assert [candidate.duration_s for candidate in paragraph_candidates] == [0.0]
